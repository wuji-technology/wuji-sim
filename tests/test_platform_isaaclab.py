# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import math

import re
import sys
from types import SimpleNamespace

import pytest
import mujoco
import numpy as np



from wuji_sim import assets, conventions, framing, mit, profiles, records, scene, scene_isaaclab
from wuji_sim.platforms import isaaclab as platform
from wuji_sim.platforms import mujoco as mujoco_platform



JOINTS = tuple(f"r_{suffix}" for suffix in conventions.JOINT_SUFFIXES)
CONTACT_TARGET = {
    "r_index_finger_mcp_abd": 0.698,
    "r_middle_finger_mcp_abd": -0.698,
}
FK_TARGET = {"r_index_finger_mcp_flex": 0.2}

CONTACT_DURATION_S = 2.0
# Ignoring the crossing target leaves both fingers at the open pose; in an Isaac Lab 3.0 run they travelled
# +0.10 rad (index) and -0.18 rad (middle) before stopping short of +-0.35.
CONTACT_MIN_TRAVEL_RAD = 0.05


def _contact_target(joint_names):
    return [CONTACT_TARGET.get(name, 0.0) for name in joint_names]


def _fk_target(joint_names):
    return [FK_TARGET.get(name, 0.0) for name in joint_names]





def _mujoco_contact_run(description_root, target):
    mjcf = description_root / "mjcf" / "right_with_glove.xml"
    view = framing.view(mjcf, "right", mujoco_platform.PLATFORM, "tracking")
    loaded = mujoco_platform.load(
        mjcf,
        scene=lambda spec: scene.add_scene(spec, "right", view),
    )
    kp = [loaded.report["effective_values"]["joints"][name]["kp"] for name in loaded.joint_names]
    kd = [loaded.report["effective_values"]["joints"][name]["kd"] for name in loaded.joint_names]
    mujoco_platform.set_mit_gains(loaded, kp, kd)
    loaded.data.ctrl[:] = target
    for _ in range(round(CONTACT_DURATION_S / loaded.model.opt.timestep)):
        mujoco.mj_step(loaded.model, loaded.data)
    return loaded, kp, kd


def _mujoco_fk_and_mass(description_root, q):
    mjcf = description_root / "mjcf" / "right_with_glove.xml"
    view = framing.view(mjcf, "right", mujoco_platform.PLATFORM, "tracking")
    loaded = mujoco_platform.load(
        mjcf,
        scene=lambda spec: scene.add_scene(spec, "right", view),
    )
    loaded.data.qpos[list(loaded.qpos_adr)] = q
    loaded.data.qvel[:] = 0.0
    mujoco.mj_forward(loaded.model, loaded.data)
    # Isaac Lab keeps the hand root at the env origin; MuJoCo lifts it to view.hand_roots[0] and adds scene_ bodies.
    root = np.asarray(view.hand_roots[0])
    positions = {
        loaded.model.body(body_id).name: (loaded.data.xpos[body_id] - root).tolist()
        for body_id in range(loaded.model.nbody)
        if loaded.model.body(body_id).name and not loaded.model.body(body_id).name.startswith("scene_")
    }
    mass = np.zeros((loaded.model.nv, loaded.model.nv), dtype=np.float64)
    mujoco.mj_fullM(loaded.model, loaded.data, mass)
    armature = np.asarray(
        [loaded.model.dof_armature[index] for index in loaded.dof_adr],
        dtype=np.float64,
    )
    mass_link = mass.copy()
    for index, value in zip(loaded.dof_adr, armature):
        mass_link[index, index] -= value
    inverse_mass = {
        scale: np.linalg.inv(mass_link + scale * np.diag(armature))
        for scale in (0.0, 1.0, 2.0)
    }
    return (
        loaded,
        positions,
        [float(mass[index, index]) for index in loaded.dof_adr],
        mass_link.tolist(),
        armature.tolist(),
        {str(scale): value.tolist() for scale, value in inverse_mass.items()},
    )


def _isaac_fk(loaded, q):
    platform.set_joint_state(loaded, q, [0.0] * len(q))
    transforms = loaded.articulation.data.body_link_pose_w.torch[0].detach().cpu().tolist()
    positions = [transform[:3] for transform in transforms]
    return dict(zip(loaded.articulation.body_names, positions))


TAU_LIMIT_HEAD = 'source = "synthetic test limits"\nunit = "N*m"\n'


def write_tau_limits(tmp_path, values):
    path = tmp_path / "tau-limit.toml"
    path.write_text(TAU_LIMIT_HEAD + "[tau_limit]\n" + "".join(
        f"{name} = {value}\n" for name, value in values.items()
    ), encoding="utf-8")
    return path


def test_official_profile_loads():
    profile = platform.load_profile()
    assert profile["platform"] == platform.PLATFORM
    assert profile["path"] == str(platform.PROFILE_PATH)
    assert profile["sha256"] == hashlib.sha256(platform.PROFILE_PATH.read_bytes()).hexdigest()
    assert set(profile["params"]) == {
        "option", "contact", "joint_friction", "joint_velocity_limit"
    }
    assert profile["params"]["option"]["timestep"]["value"] == 0.005
    assert profile["params"]["contact"] == {
        name: {"source": "choice", "value": 0.7, "unit": "1"} for name in platform.CONTACT_ATTRIBUTES
    }
    assert set(profile["params"]["joint_friction"]) == set(
        f"{prefix}{suffix}" for prefix in ("l_", "r_") for suffix in conventions.JOINT_SUFFIXES
    )
    assert {entry["unit"] for entry in profile["params"]["joint_friction"].values()} == {"N*m"}
    assert set(profile["params"]["joint_velocity_limit"]) == set(
        f"{prefix}{suffix}" for prefix in ("l_", "r_") for suffix in conventions.JOINT_SUFFIXES
    )
    assert profile["overrides"] == {}


def test_profile_validation_rejects_invalid_timestep(tmp_path):
    text = platform.PROFILE_PATH.read_text(encoding="utf-8").replace(
        "value = 0.005\nunit = \"s\"", "value = -0.005\nunit = \"s\"", 1
    )
    path = tmp_path / "isaaclab.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match=re.escape("option.timestep")):
        platform.load_profile(path)


def test_profile_removed_gain_scale_group_is_rejected(tmp_path):
    path = tmp_path / "isaaclab.toml"
    path.write_text(
        platform.PROFILE_PATH.read_text(encoding="utf-8") + '\n[mit_fw_gain_scale]\nsource = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(profiles.ProfileError, match=r"unknown group.*mit_fw_gain_scale"):
        platform.load_profile(path)


@pytest.mark.parametrize("name", sorted(platform.CONTACT_ATTRIBUTES))
def test_profile_validation_rejects_negative_default_material_friction(tmp_path, name):
    text = platform.PROFILE_PATH.read_text(encoding="utf-8")
    changed = text.replace(f"{name} = {{ value = 0.7", f"{name} = {{ value = -0.7", 1)
    assert changed != text
    path = tmp_path / "isaaclab.toml"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match=re.escape(f"contact.{name}")):
        platform.load_profile(path)


def test_profile_without_the_contact_group_is_rejected(tmp_path):
    text = platform.PROFILE_PATH.read_text(encoding="utf-8")
    start, end = text.index("[contact]"), text.index("[joint_friction]")
    path = tmp_path / "isaaclab.toml"
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match=re.escape("contact: missing group")):
        platform.load_profile(path)


def test_tau_limits_are_loaded_with_digest(tmp_path):
    values = {name: 0.1 + index * 0.01 for index, name in enumerate(JOINTS)}
    path = write_tau_limits(tmp_path, values)
    assert profiles.load_tau_limits(path) == {
        "file": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "synthetic test limits",
        "unit": "N*m",
        "values": values,
    }


def test_mit_stability_uses_integrated_actuator_bound():
    loaded = SimpleNamespace(
        joint_names=("r_a", "r_b"),
        joint_inertia=(1.0, 2.0),
        dt=0.005,
        control_path=mit.INTEGRATED_CONTROL_PATH,
        report={"control_path": mit.INTEGRATED_CONTROL_PATH},
    )
    report = platform.mit_stability(loaded, [1.0, 2.0], [0.1, 0.2])
    assert report["limit"] == 4.0
    assert report["criterion"] == platform.MIT_STABILITY_CRITERION
    assert report["max_joint"] == "r_a"
    assert report["joints"]["r_a"]["ratio"] == pytest.approx(0.000025)


def test_mit_stability_rejects_gain_at_bound():
    loaded = SimpleNamespace(
        joint_names=("r_a",),
        joint_inertia=(1.0,),
        dt=0.005,
        control_path=mit.INTEGRATED_CONTROL_PATH,
        report={"control_path": mit.INTEGRATED_CONTROL_PATH},
    )
    with pytest.raises(mit.MitStabilityError, match="integrated actuator"):
        platform.mit_stability(loaded, [mit.MIT_STABILITY_LIMIT / loaded.dt**2], [0.0])


def test_root_quaternion_matches_scene():
    w, x, y, z = scene.HAND_ROOT_QUAT
    assert platform.HAND_ROOT_QUAT == [x, y, z, w]
    assert assets.USD_ENTRIES["right"] == "usd/right_with_glove/wujihand2_2_glove.usd"

def test_platform_records_are_registered():
    assert "isaaclab" in records.PLATFORMS
    assert records.REQUIRED_DISTRIBUTIONS["isaaclab"] == (
        "isaaclab", "isaacsim", "torch", "numpy"
    )
    assert assets.USD_ENTRIES["left"] == "usd/left_with_glove/wujihand2_2_glove.usd"


@pytest.mark.parametrize("env_spacing", [0, -0.3, math.nan, math.inf, True])
def test_load_rejects_invalid_env_spacing_before_starting_isaac(description_root, env_spacing):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    with pytest.raises(profiles.ProfileError, match="env_spacing"):
        platform.load(usd, hand="right", num_envs=4, env_spacing=env_spacing, floor_z=-0.1, scene=scene_isaaclab.Scene())


@pytest.mark.parametrize("floor_z", [math.nan, math.inf, True, None, "-0.1"])
def test_load_rejects_invalid_floor_z_before_starting_isaac(description_root, floor_z):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    with pytest.raises(profiles.ProfileError, match="floor_z"):
        platform.load(usd, hand="right", floor_z=floor_z, scene=scene_isaaclab.Scene())


def test_load_requires_the_scene_argument(description_root):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    with pytest.raises(TypeError, match="scene"):
        platform.load(usd, hand="right", floor_z=-0.1)


def test_load_rejects_a_scene_without_the_hooks_before_starting_isaac(description_root):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    with pytest.raises(profiles.ProfileError, match="scene must provide"):
        platform.load(usd, hand="right", floor_z=-0.1, scene=object())


def test_load_accepts_scene_none_past_the_scene_check(description_root):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    with pytest.raises(ValueError, match="hand must be one of"):
        platform.load(usd, hand="middle", floor_z=-0.1, scene=None)


def test_load_accepts_a_contact_override_past_the_joint_name_check(description_root, tmp_path):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    overrides = {"contact": {"default_material_static_friction": {"value": 1.0, "reason": "test"}}}
    with pytest.raises(profiles.ProfileError, match="cannot load model override file"):
        platform.load(usd, hand="right", floor_z=-0.1, scene=None, overrides=overrides,
                      model_overrides_path=tmp_path / "missing.toml")


def test_load_rejects_a_joint_override_for_the_other_hand(description_root):
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    overrides = {"joint_friction": {"l_thumb_ip": {"value": 0.1, "reason": "test"}}}
    with pytest.raises(profiles.ProfileError, match="is not a joint of 'right' hand"):
        platform.load(usd, hand="right", floor_z=-0.1, scene=None, overrides=overrides)


@pytest.mark.parametrize("group", sorted(profiles.MODEL_OVERRIDE_UNITS))
def test_load_rejects_partial_model_overrides_before_starting_isaac(description_root, tmp_path, group):
    text = profiles.MODEL_OVERRIDES_PATH.read_text(encoding="utf-8")
    start = text.index(f"[{group}.r_thumb_ip]\n")
    path = tmp_path / "model_overrides.toml"
    path.write_text(text[:start] + text[text.index("\n\n", start) + 2:], encoding="utf-8")
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    with pytest.raises(profiles.ProfileError, match=re.escape(f"{group}: no entry for model joint(s) ['r_thumb_ip']")):
        platform.load(usd, hand="right", floor_z=-0.1, scene=None, model_overrides_path=path)


def _tracking_view(description_root):
    return framing.view(description_root / assets.MJCF_ENTRIES["right"], "right", platform.PLATFORM, "tracking")


@pytest.fixture(scope="session")
def mujoco_reference(description_root):
    names = JOINTS
    target = _contact_target(names)
    _, kp, kd = _mujoco_contact_run(description_root, [0.0] * len(names))
    fk_positions = []
    mujoco_mass = None
    mujoco_mass_link = None
    mujoco_armature = None
    mujoco_inverse_mass = None
    for q in ([0.0] * len(names), _fk_target(names)):
        _, positions, current_mass, current_mass_link, current_armature, current_inverse = _mujoco_fk_and_mass(
            description_root, q
        )
        fk_positions.append(positions)
        if mujoco_mass is None:
            mujoco_mass = current_mass
            mujoco_mass_link = current_mass_link
            mujoco_armature = current_armature
            mujoco_inverse_mass = current_inverse
    return {
        "target": target,
        "kp": kp,
        "kd": kd,
        "fk_positions": fk_positions,
        "mass": mujoco_mass,
        "mass_link": mujoco_mass_link,
        "armature": mujoco_armature,
        "mass_inverse": mujoco_inverse_mass,
    }

@pytest.fixture(scope="session")
def isaac_app(request):
    pytest.importorskip("isaaclab")
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        stack = contextlib.ExitStack()
        app = stack.enter_context(platform.launch(headless=True, device="cuda:0"))
    finally:
        sys.argv = saved_argv
    request.config._isaac_kit_stack = stack
    yield app


@pytest.fixture(scope="session")
def isaac_loaded(mujoco_reference, isaac_app, description_root):
    del mujoco_reference
    usd = description_root / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"
    return platform.load(usd, hand="right", device="cuda:0", floor_z=_tracking_view(description_root).floor_z,
                         scene=scene_isaaclab.Scene())


# One articulation serves the session (reloading rebuilds the stage); zeroing it keeps the tests order-independent.
@pytest.fixture(autouse=True)
def reset_isaac_loaded(request):
    if "isaac_loaded" in request.fixturenames:
        loaded = request.getfixturevalue("isaac_loaded")
        zeros = [0.0] * len(loaded.joint_names)
        platform.set_mit_gains(loaded, zeros, zeros)
        platform.set_command(loaded, zeros, zeros, zeros)
        platform.set_joint_state(loaded, zeros, zeros)


@pytest.mark.isaac
def test_isaac_scene_has_the_preview_look(isaac_loaded, description_root):
    import carb.settings
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom, UsdLux

    stage = get_current_stage()
    ground = stage.GetPrimAtPath("/World/ground")
    assert UsdGeom.Imageable(ground).ComputeVisibility() == UsdGeom.Tokens.inherited
    ground_z = UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
    assert ground_z == pytest.approx(_tracking_view(description_root).floor_z, abs=1e-6)
    shader = stage.GetPrimAtPath(scene_isaaclab.GROUND_SHADER_PRIM)
    assert tuple(shader.GetAttribute("inputs:texture_scale").Get()) == scene_isaaclab.GROUND_TEXTURE_SCALE
    sky = UsdLux.LightAPI(stage.GetPrimAtPath("/World/scene_skybox"))
    assert sky.GetIntensityAttr().Get() == pytest.approx(scene_isaaclab.SCENE_LIGHT_INTENSITY)
    assert tuple(sky.GetColorAttr().Get()) == pytest.approx(scene_isaaclab.SCENE_SKY_RGB)
    for name in scene_isaaclab.SCENE_LIGHTS:
        light = UsdLux.LightAPI(stage.GetPrimAtPath(f"/World/{name}"))
        assert light.GetIntensityAttr().Get() == pytest.approx(scene_isaaclab.SCENE_LIGHT_INTENSITY)
    settings = carb.settings.get_settings()
    assert {path: settings.get(path) for path in scene_isaaclab.RTX_SETTINGS} == scene_isaaclab.RTX_SETTINGS


@pytest.mark.isaac
def test_isaac_runtime_properties(isaac_loaded):
    loaded = isaac_loaded
    data = loaded.articulation.data

    def selected(values):
        return values.torch[0, list(loaded.joint_ids)].cpu().tolist()

    assert loaded.dt == pytest.approx(0.005)
    assert loaded.joint_names == JOINTS
    assert selected(data.joint_stiffness) == [0.0] * 20
    assert selected(data.joint_damping) == [0.0] * 20
    assert selected(data.joint_armature) == pytest.approx(
        [loaded.report["effective_values"]["joints"][name]["armature"] for name in loaded.joint_names]
    )
    assert selected(data.joint_effort_limits) == pytest.approx(loaded.tau_limit)
    assert selected(data.joint_vel_limits) == [100.0] * 20


@pytest.mark.isaac
def test_isaac_runtime_self_collision_is_enabled(isaac_loaded):
    readback = isaac_loaded.report["self_collision_readback"]
    assert readback["configured"] is True
    assert readback["attribute"] == "physxArticulation:enabledSelfCollisions"
    assert readback["value"] is True


@pytest.mark.isaac
def test_isaac_default_material_friction_is_the_profile_value(isaac_loaded):
    import warp as wp

    contact = isaac_loaded.report["effective_values"]["contact"]
    assert contact["default_material_prim"] == "/physicsScene/defaultMaterial"
    assert {name: contact[name] for name in platform.CONTACT_ATTRIBUTES} == pytest.approx(
        dict.fromkeys(platform.CONTACT_ATTRIBUTES, 0.7))
    shapes = wp.to_torch(isaac_loaded.articulation.root_view.get_material_properties())[0, :, :2].cpu()
    assert shapes.numel() > 0
    assert shapes.flatten().tolist() == pytest.approx([0.7] * shapes.numel())


@pytest.mark.isaac
def test_isaac_self_collision_blocks_crossing_fingers(isaac_loaded, mujoco_reference):
    target = mujoco_reference["target"]
    kp = mujoco_reference["kp"]
    kd = mujoco_reference["kd"]
    platform.set_mit_gains(isaac_loaded, kp, kd)

    platform.set_command(isaac_loaded, [0.0] * 20, [0.0] * 20, [0.0] * 20)
    for _ in range(round(CONTACT_DURATION_S / isaac_loaded.dt)):
        platform.step(isaac_loaded)
    q_open, _ = platform.joint_state(isaac_loaded)
    open_index = q_open[isaac_loaded.joint_names.index("r_index_finger_mcp_abd")]
    open_middle = q_open[isaac_loaded.joint_names.index("r_middle_finger_mcp_abd")]
    assert abs(open_index) <= 0.05
    assert abs(open_middle) <= 0.05

    platform.set_command(isaac_loaded, target, [0.0] * 20, [0.0] * 20)
    for _ in range(round(CONTACT_DURATION_S / isaac_loaded.dt)):
        platform.step(isaac_loaded)
    q_positive, _ = platform.joint_state(isaac_loaded)
    index_abd = q_positive[isaac_loaded.joint_names.index("r_index_finger_mcp_abd")]
    middle_abd = q_positive[isaac_loaded.joint_names.index("r_middle_finger_mcp_abd")]
    print(f"self_collision_final_abd=index:{index_abd:.9g},middle:{middle_abd:.9g}")
    assert index_abd - open_index >= CONTACT_MIN_TRAVEL_RAD
    assert open_middle - middle_abd >= CONTACT_MIN_TRAVEL_RAD
    assert index_abd < 0.35
    assert middle_abd > -0.35

@pytest.mark.isaac
def test_isaac_fk_and_generalized_mass_parity(isaac_loaded, mujoco_reference):
    platform.set_mit_gains(isaac_loaded, [0.0] * 20, [0.0] * 20)
    max_diffs = []
    for q, mujoco_positions in zip(
        ([0.0] * 20, _fk_target(isaac_loaded.joint_names)),
        mujoco_reference["fk_positions"],
    ):
        isaac_positions = _isaac_fk(isaac_loaded, q)
        missing = sorted(set(mujoco_positions) - {"world"} - set(isaac_positions))
        assert not missing
        common = sorted(set(mujoco_positions) - {"world"})
        max_diff = max(
            math.sqrt(sum((isaac_positions[name][axis] - mujoco_positions[name][axis]) ** 2 for axis in range(3)))
            for name in common
        )
        max_diffs.append(max_diff)
        assert max_diff <= 1e-5
    relative = [
        abs(isaac - mujoco) / abs(mujoco)
        for isaac, mujoco in zip(isaac_loaded.joint_inertia, mujoco_reference["mass"])
    ]
    assert all(math.isfinite(value) and value <= 0.01 for value in relative)
    print(f"fk_max_diff_m={max(max_diffs):.9g}")
    print(f"generalized_mass_relative={dict(zip(isaac_loaded.joint_names, relative))}")
    print(f"generalized_mass_over_1pct={[name for name, value in zip(isaac_loaded.joint_names, relative) if value > 0.01]}")

@pytest.mark.isaac
def test_isaac_armature_is_used_by_dynamics(isaac_loaded, mujoco_reference):
    import torch

    loaded = isaac_loaded
    platform.set_mit_gains(loaded, [0.0] * 20, [0.0] * 20)
    tau = 1.0e-3
    expected_inverse = np.asarray(mujoco_reference["mass_inverse"]["1.0"])
    errors = {}
    for name in ("r_thumb_ip", "r_index_finger_dip", "r_index_finger_mcp_flex"):
        joint_index = loaded.joint_names.index(name)
        platform.set_joint_state(loaded, [0.0] * 20, [0.0] * 20)
        platform.set_command(loaded, [0.0] * 20, [0.0] * 20, [0.0] * 20)
        platform.step(loaded)
        _, qd0 = platform.joint_state(loaded)

        platform.set_joint_state(loaded, [0.0] * 20, [0.0] * 20)
        platform.set_command(loaded, [0.0] * 20, [0.0] * 20, [0.0] * 20)
        effort = torch.zeros((1, 20), dtype=torch.float32, device=loaded.device)
        effort[0, joint_index] = tau
        loaded.articulation.actuators.target_command.set_effort_index(value=effort, joint_ids=list(loaded.joint_ids))
        platform.step(loaded)
        _, qd1 = platform.joint_state(loaded)

        column = (np.asarray(qd1) - np.asarray(qd0)) / (tau * loaded.dt)
        expected = expected_inverse[joint_index, joint_index]
        error = abs(column[joint_index] - expected) / abs(expected)
        errors[name] = float(error)
        assert error <= 0.02
    print(f"armature_probe_diagonal_relative={errors}")




@pytest.mark.isaac
def test_isaac_runtime_controls(isaac_loaded):
    import torch

    loaded = isaac_loaded
    gains = [1.0] * 20
    platform.set_mit_gains(loaded, gains, [0.03] * 20)
    assert loaded.articulation.data.joint_stiffness.torch[0].cpu().tolist() == pytest.approx(gains)
    assert loaded.articulation.data.joint_damping.torch[0].cpu().tolist() == pytest.approx([0.03] * 20)
    with pytest.raises(ValueError, match="tau_ff"):
        platform.set_command(loaded, [0.0] * 20, [0.0] * 20, [0.01] + [0.0] * 19)
    platform.set_command(loaded, torch.zeros(20, device=loaded.device), torch.zeros(20, device=loaded.device), [0.0] * 20)
    for _ in range(round(CONTACT_DURATION_S / loaded.dt)):
        platform.step(loaded)


@pytest.mark.isaac
def test_isaac_mit_gains_readback_matches_root_view(isaac_loaded):
    import warp as wp

    loaded = isaac_loaded
    kp = [1.5] * 20
    kd = [0.07] * 20
    platform.set_mit_gains(loaded, kp, kd)
    joint_ids = list(loaded.joint_ids)
    stiffness = wp.to_torch(loaded.articulation.root_view.get_dof_stiffnesses())[0, joint_ids].cpu().tolist()
    damping = wp.to_torch(loaded.articulation.root_view.get_dof_dampings())[0, joint_ids].cpu().tolist()
    assert stiffness == pytest.approx(kp)
    assert damping == pytest.approx(kd)


@pytest.mark.isaac
def test_isaac_applied_effort_matches_pd_formula(isaac_loaded):
    loaded = isaac_loaded
    kp = [2.0] * 20
    kd = [0.05] * 20
    platform.set_mit_gains(loaded, kp, kd)
    q0, qd0 = platform.joint_state(loaded)
    q_des = _fk_target(loaded.joint_names)
    qd_des = [0.0] * 20
    platform.set_command(loaded, q_des, qd_des, [0.0] * 20)
    target_command = loaded.articulation.actuators.target_command
    effort_readback = target_command.effort.torch[0, list(loaded.joint_ids)].cpu().tolist()
    assert effort_readback == pytest.approx([0.0] * 20, abs=1e-9)
    platform.step(loaded)
    applied = platform.applied_effort(loaded)
    expected = [
        max(-limit, min(limit, gain_kp * (q_target - q) + gain_kd * (qd_target - qd)))
        for q, qd, q_target, qd_target, gain_kp, gain_kd, limit in zip(
            q0, qd0, q_des, qd_des, kp, kd, loaded.tau_limit
        )
    ]
    assert applied == pytest.approx(expected, abs=1e-5)
