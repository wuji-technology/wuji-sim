# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from wuji_sim import conventions, framing, mit, scene
from wuji_sim.platforms import motrixsim as platform
from wuji_sim.profiles import ProfileError

HANDS = ("left", "right")
PREFIX = {"left": "l_", "right": "r_"}


@pytest.fixture(params=HANDS)
def hand(request):
    return request.param


@pytest.fixture
def mjcf(description_root, hand):
    return description_root / "mjcf" / f"{hand}_with_glove.xml"


@pytest.fixture
def right_mjcf(description_root):
    return description_root / "mjcf" / "right_with_glove.xml"


def test_official_profile_loads():
    profile = platform.load_profile()
    assert profile["platform"] == "motrixsim"
    assert profile["path"] == str(platform.PROFILE_PATH)
    assert profile["sha256"] == hashlib.sha256(platform.PROFILE_PATH.read_bytes()).hexdigest()
    assert set(profile["params"]) == {"option", "contact", "joint_coulomb_friction", "joint_damping"}
    option = profile["params"]["option"]
    assert {name: option[name]["value"] for name in platform.OPTION_FIELDS} == {
        "timestep": 0.005,
        "solver": "Newton",
        "iterations": 100,
    }
    # MuJoCo-only option parameters must stay out of this profile: MotrixSim rejects them as unknown groups.
    assert set(option) == set(platform.OPTION_FIELDS)
    assert profile["overrides"] == {}


@pytest.mark.parametrize("table", ["option.integrator", "option.ls_iterations"])
def test_profile_mujoco_only_option_field_is_rejected(tmp_path, table):
    text = platform.PROFILE_PATH.read_text(encoding="utf-8") + f"\n[{table}]\nvalue = 1\nunit = '1'\n"
    path = tmp_path / "motrixsim.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProfileError, match=table.split(".")[-1]):
        platform.load_profile(path)


def test_load_builds_a_backend_model_matching_the_patched_spec(mjcf, hand):
    view = framing.view(mjcf, hand, platform.PLATFORM, "tracking")
    loaded = platform.load(mjcf, scene=lambda spec: scene.add_scene(spec, hand, view))
    assert loaded.report["platform"] == "motrixsim"
    assert loaded.joint_names == tuple(PREFIX[hand] + suffix for suffix in conventions.JOINT_SUFFIXES)
    assert tuple(loaded.scene_model.joint_names) == loaded.joint_names
    assert tuple(loaded.scene_model.actuator_names) == tuple(f"{name}_actuator" for name in loaded.joint_names)
    assert loaded.scene_model.num_actuators == len(loaded.joint_names) == 20
    json.dumps(loaded.report, allow_nan=False)


def test_load_reports_backend_audit_without_impositions(right_mjcf):
    loaded = platform.load(right_mjcf)
    assert loaded.report["backend_imposed"] == []
    checked = loaded.report["backend_checked"]
    assert "option.timestep" in checked and "option.solver" in checked
    assert any(entry.startswith("joint.r_index_finger_pip.") for entry in checked)
    assert "actuator.r_index_finger_pip_actuator.ctrlrange" in checked
    assert loaded.report["import_options"]["motrixsim"] == "1.0.0"
    assert loaded.report["import_options"]["solver"] == "Newton"
    # The runtime option the backend steps matches the profile request; tolerance is backend-managed.
    assert loaded.scene_model.options.timestep == pytest.approx(0.005, rel=1e-6)
    assert loaded.scene_model.options.max_iterations == 100
    assert loaded.scene_model.options.solver_tolerance == pytest.approx(1e-4, rel=1e-6)
    assert not any(entry["parameter"] == "option.tolerance" for entry in loaded.report["backend_imposed"])


def test_mit_stability_uses_the_integrated_bound(right_mjcf):
    loaded = platform.load(right_mjcf)
    kp = np.full(len(loaded.joint_names), scene.DEMO_KP)
    kd = np.full(len(loaded.joint_names), scene.DEMO_KD)
    report = platform.mit_stability(loaded, kp, kd)
    assert report["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert report["limit"] == mit.MIT_STABILITY_LIMIT == 4.0
    assert report["criterion"] == "kp_j*dt^2/I_j < 4 for every integrated actuator joint; I_j is the diagonal of the compiled joint-space " \
        "inertia matrix (mj_fullM, armature included) at the initial pose qpos0"
    assert report["max_ratio"] < report["limit"]


def test_set_mit_gains_changes_backend_and_host(right_mjcf):
    loaded = platform.load(right_mjcf)
    kp = [scene.DEMO_KP * (1 + 0.01 * j) for j in range(20)]
    kd = [scene.DEMO_KD] * 20
    platform.set_mit_gains(loaded, kp, kd)
    assert loaded.gains == tuple(zip(kp, kd))
    # Host twin keeps the same bias so its gains track the backend controller.
    for actuator_id, gain_kp in zip(loaded.actuator_ids, kp):
        assert loaded.model.actuator_biasprm[actuator_id, 1] == pytest.approx(-gain_kp)
    # The backend override writes the bias only: the unit command gain is untouched, matching the MuJoCo form.
    gains = [float(actuator.get_gain_override(loaded.scene_data)) for actuator in loaded.actuators]
    assert gains == [1.0] * 20


def test_apply_mit_and_step_track_the_reference(right_mjcf):
    loaded = platform.load(right_mjcf)
    kp = np.full(len(loaded.joint_names), scene.DEMO_KP)
    kd = np.full(len(loaded.joint_names), scene.DEMO_KD)
    platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    timestep = float(loaded.model.opt.timestep)
    platform.set_state(loaded, [0.0] * int(loaded.scene_model.num_dof_pos))
    errors = []
    for step in range(round(1.0 / timestep)):
        q_des, qd_des = scene.tracking_reference(loaded.joint_names, step * timestep)
        platform.apply_mit(loaded, q_des, qd_des)
        q = [float(value) for value in loaded.scene_data.dof_pos]
        errors.append(max(abs(q_des[j] - q[j]) for j in range(len(q_des))))
        platform.step(loaded)
    assert max(errors) < 0.35  # rad; the preset poses sweep most of the joint range


def test_set_state_installs_the_full_dof_state(right_mjcf):
    loaded = platform.load(right_mjcf)
    n = int(loaded.scene_model.num_dof_pos)
    platform.set_state(loaded, [0.0] * n, [0.0] * n)
    assert [round(float(value), 6) for value in loaded.scene_data.dof_pos] == [0.0] * n
    with pytest.raises(ValueError, match="do not match"):
        platform.set_state(loaded, [0.0] * (n - 1))


def test_apply_mit_rejects_mismatched_shapes(right_mjcf):
    loaded = platform.load(right_mjcf)
    with pytest.raises(ValueError, match="do not match"):
        platform.apply_mit(loaded, [0.0] * 19, [0.0] * 20)


def test_code_override_sets_joint_friction(right_mjcf):
    name = "r_index_finger_pip"
    overrides = {"joint_coulomb_friction": {name: {"value": 0.02, "reason": "test friction sweep"}}}
    loaded = platform.load(right_mjcf, overrides=overrides)
    # The patched spec carries it into the backend through the XML route.
    joint = dict(zip(loaded.scene_model.joint_names, loaded.scene_model.joints))[name]
    assert joint.frictionloss == pytest.approx(0.02, abs=1e-9)
    assert f"with override: joint_coulomb_friction.{name}" in loaded.report["labels"]
    entry = next(e for e in loaded.report["overrides"] if e.get("parameter") == f"joint_coulomb_friction.{name}")
    assert (entry["kind"], entry["origin"], entry["value"]) == ("profile_override", "code", 0.02)


def test_code_override_for_joint_of_other_hand_is_rejected(right_mjcf):
    overrides = {"joint_coulomb_friction": {"l_index_finger_pip": {"value": 0.02, "reason": "test"}}}
    with pytest.raises(ProfileError, match="l_index_finger_pip"):
        platform.load(right_mjcf, overrides=overrides)


def test_tau_limit_file_replaces_backend_agnostic_clamps(tmp_path, right_mjcf):
    from wuji_sim import profiles as profiles_module

    values = {f"r_{suffix}": round(0.105 + 0.01 * j, 3) for j, suffix in enumerate(conventions.JOINT_SUFFIXES)}
    path = tmp_path / "tau_limit.toml"
    path.write_text('source = "synthetic test limits"\nunit = "N*m"\n[tau_limit]\n'
                    + "".join(f"{n} = {v}\n" for n, v in values.items()), encoding="utf-8")
    loaded = platform.load(right_mjcf, tau_limits=profiles_module.load_tau_limits(path))
    assert loaded.tau_limit == tuple(values[name] for name in loaded.joint_names)
    assert loaded.report["labels"][-1] == "with override: tau_limit (synthetic test limits)"


def test_link_pose_returns_wxyz_world_pose(right_mjcf):
    import numpy as np
    import mujoco

    loaded = platform.load(right_mjcf)
    platform.set_state(loaded, [0.0] * int(loaded.scene_model.num_dof_pos))
    pose = platform.link_pose(loaded, "r_middle_finger_proximal")
    # Cross-check position and quaternion against the compiled MuJoCo twin at the same (zeros) state.
    loaded.data.qpos[:] = 0.0
    mujoco.mj_forward(loaded.model, loaded.data)
    body = loaded.model.body("r_middle_finger_proximal")
    assert np.allclose(pose[:3], loaded.data.xpos[body.id], atol=1e-5)
    assert abs(np.dot(pose[3:], loaded.data.xquat[body.id])) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="no link"):
        platform.link_pose(loaded, "no_such_link")


def test_audit_backend_detects_a_tampered_option(right_mjcf):
    loaded = platform.load(right_mjcf)
    loaded.scene_model.options.timestep = 0.123  # tamper after load
    option = dict(zip(platform.OPTION_FIELDS, (0.005, "Newton", 100)))
    imposed, _ = platform.audit_backend(loaded._build, loaded.scene_model, option)
    assert any(e["parameter"] == "option.timestep" and e["note"] == "backend-imposed" for e in imposed)
    assert any(e["parameter"] == "option.timestep" and e["effective"] == pytest.approx(0.123) for e in imposed)
