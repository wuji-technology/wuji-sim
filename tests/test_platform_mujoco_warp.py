# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import re
import tomllib

import mujoco
import numpy as np
import pytest

mjw = pytest.importorskip("mujoco_warp")
import warp as wp

from wuji_sim import assets, conventions, framing, mit, profiles, scene
from wuji_sim.platforms import _mjcf
from wuji_sim.platforms import mujoco as cpu_platform
from wuji_sim.platforms import mujoco_warp as platform
from wuji_sim.profiles import ProfileError

HANDS = ("left", "right")
PREFIX = {"left": "l_", "right": "r_"}
DEVICE = "cuda:0"


def official_model_tau_limits():
    document = tomllib.loads(profiles.MODEL_OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {
        name: entry["value"]
        for name, entry in document["joint_torque_limit"].items()
        if name != "source"
    }


MIT_FIELDS = ("q_des", "qd_des", "tau_ff")
OFFICIAL_CONTACT = {
    "geom_friction": [0.7, 0.005, 0.0001],
    "geom_solref": [0.02, 1.0],
    "geom_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    "geom_condim": 4,
}
BACKEND_CHECKED = [
    "option.timestep",
    "option.integrator",
    "option.solver",
    "option.iterations",
    "option.tolerance",
    "option.ls_iterations",
    "option.jacobian",
    "dof_frictionloss",
    "dof_damping",
    "dof_armature",
    "jnt_actfrcrange",
    "actuator_ctrlrange",
    "actuator_gaintype",
    "actuator_biastype",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcelimited",
    "actuator_forcerange",
    "actuator_ctrllimited",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_condim",
]
CAPACITY_SOURCE = "choice"


ONE_HINGE_XML = """
<mujoco>
  <option tolerance="{tolerance}"/>
  <worldbody>
    <body name="link">
      <joint name="hinge" axis="1 0 0" range="-1 1" actuatorfrcrange="-0.5 0.5"/>
      <geom type="capsule" size="0.01 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" ctrlrange="-0.5 0.5"/>
  </actuator>
</mujoco>
"""


def official_text():
    return platform.PROFILE_PATH.read_text(encoding="utf-8")


def edit_line(text, table, key, new=None):
    lines = text.splitlines(keepends=True)
    start = lines.index(f"[{table}]\n")
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("["):
            break
        if lines[i].startswith(f"{key} ="):
            lines[i : i + 1] = [] if new is None else [new + "\n"]
            return "".join(lines)
    raise AssertionError(f"{table}.{key} not found in the profile text")


def drop_tables(text, prefix):
    kept, skipping = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("["):
            skipping = line.startswith(f"[{prefix}")
        if not skipping:
            kept.append(line)
    return "".join(kept)


def write_profile(tmp_path, text):
    path = tmp_path / "mujoco_warp.toml"
    path.write_text(text, encoding="utf-8")
    return path


def with_scene(hand):
    view = framing.view(assets.resolve(hand).mjcf, hand, platform.PLATFORM, "tracking")
    return lambda spec: scene.add_scene(spec, hand, view)


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
    assert platform.PLATFORM == "mujoco_warp"
    assert platform.PROFILE_PATH.name == "mujoco_warp.toml"
    assert profile["platform"] == "mujoco_warp"
    assert profile["path"] == str(platform.PROFILE_PATH)
    assert profile["sha256"] == hashlib.sha256(platform.PROFILE_PATH.read_bytes()).hexdigest()
    assert set(profile["params"]) == {"option", "contact", "joint_coulomb_friction", "joint_damping", "capacity"}
    option = profile["params"]["option"]
    assert {name: option[name]["value"] for name in _mjcf.OPTION_FIELDS} == {
        "timestep": 0.005,
        "integrator": "implicitfast",
        "solver": "Newton",
        "iterations": 100,
        "tolerance": 1e-6,
        "ls_iterations": 50,
    }
    assert option["timestep"] == {
        "value": 0.005,
        "unit": "s",
        "source": "choice",
    }
    assert option["tolerance"] == {"value": 1e-6, "unit": "1", "source": "default: MuJoCo Warp 3.11.0"}
    assert {option[name]["source"] for name in ("integrator", "solver", "iterations", "ls_iterations")} == {
        "default: mjlab 1.6.0"
    }
    assert profile["params"]["capacity"] == {
        "nconmax": {"value": 256, "unit": "contacts per world", "source": CAPACITY_SOURCE},
        "njmax": {"value": 1024, "unit": "constraint rows per world", "source": CAPACITY_SOURCE},
    }
    contact = profile["params"]["contact"]
    assert {name: contact[name]["value"] for name in _mjcf.CONTACT_FIELDS} == OFFICIAL_CONTACT
    friction = profile["params"]["joint_coulomb_friction"]
    assert {(p["value"], p["unit"], p["source"]) for p in friction.values()} == {
        (0.0, "N*m", "estimate")
    }

    assert profile["overrides"] == {}


def test_profile_equals_mujoco_profile_except_timestep_tolerance_and_capacity():
    warp_params = platform.load_profile()["params"]
    cpu_params = cpu_platform.load_profile()["params"]
    differs = {"timestep", "tolerance"}
    assert {k: v for k, v in warp_params["option"].items() if k not in differs} == {
        k: v for k, v in cpu_params["option"].items() if k not in differs
    }
    assert (warp_params["option"]["timestep"]["value"], cpu_params["option"]["timestep"]["value"]) == (0.005, 0.005)
    assert warp_params["contact"] == cpu_params["contact"]
    assert warp_params["joint_coulomb_friction"] == cpu_params["joint_coulomb_friction"]
    assert warp_params["joint_damping"] == cpu_params["joint_damping"]



def test_profile_joint_friction_covers_both_description_hands(description_root):
    names = set()
    for hand in HANDS:
        spec = mujoco.MjSpec.from_file(str(description_root / "mjcf" / f"{hand}_with_glove.xml"))
        names |= {j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE}
    assert len(names) == 40
    assert set(platform.load_profile()["params"]["joint_coulomb_friction"]) == names



@pytest.mark.parametrize("key", ["value", "unit"])
@pytest.mark.parametrize(
    "table",
    [
        "option.tolerance",
        "contact.geom_solref",
        "joint_coulomb_friction.l_pinky_dip",
        "capacity.nconmax",
        "capacity.njmax",

    ],
)
def test_profile_missing_field_is_rejected(tmp_path, table, key):
    path = write_profile(tmp_path, edit_line(official_text(), table, key))
    with pytest.raises(ProfileError, match=re.escape(f"{table}: missing '{key}'")):
        platform.load_profile(path)


@pytest.mark.parametrize("group", ["option", "contact", "joint_coulomb_friction", "joint_damping", "capacity"])
def test_profile_missing_group_source_is_rejected(tmp_path, group):
    path = write_profile(tmp_path, edit_line(official_text(), group, "source"))
    with pytest.raises(ProfileError, match=re.escape(f"{group}: missing 'source'")):
        platform.load_profile(path)


@pytest.mark.parametrize(
    "table, key",
    [
        ("option.solver", "unit"),
        ("option.timestep", "source"),
        ("capacity.nconmax", "unit"),
        ("capacity", "source"),
    ],
)
def test_profile_empty_string_field_is_rejected(tmp_path, table, key):
    path = write_profile(tmp_path, edit_line(official_text(), table, key, f'{key} = ""'))
    with pytest.raises(ProfileError, match=re.escape(f"{table}: '{key}' must be a non-empty string")):
        platform.load_profile(path)


def test_profile_unknown_group_is_rejected(tmp_path):
    text = official_text() + '\n[bogus.x]\nvalue = 1.0\nunit = "1"\nsource = "estimate"\n'
    with pytest.raises(ProfileError, match="bogus"):
        platform.load_profile(write_profile(tmp_path, text))


def test_profile_removed_gain_scale_group_is_rejected(tmp_path):
    text = official_text() + '\n[mit_fw_gain_scale]\nsource = "x"\n'
    with pytest.raises(ProfileError, match=r"unknown group.*mit_fw_gain_scale"):
        platform.load_profile(write_profile(tmp_path, text))


@pytest.mark.parametrize("group", ["option", "contact", "joint_coulomb_friction", "joint_damping", "capacity"])
def test_profile_missing_group_is_rejected(tmp_path, group):
    with pytest.raises(ProfileError, match=re.escape(f"{group}: missing group")):
        platform.load_profile(write_profile(tmp_path, drop_tables(official_text(), group)))


@pytest.mark.parametrize("name", ["nconmax", "njmax"])
def test_profile_missing_capacity_field_is_rejected(tmp_path, name):
    with pytest.raises(ProfileError, match=re.escape(f"capacity.{name}: missing parameter")):
        platform.load_profile(write_profile(tmp_path, drop_tables(official_text(), f"capacity.{name}")))


def test_profile_unknown_capacity_field_is_rejected(tmp_path):
    text = official_text() + '\n[capacity.naconmax]\nvalue = 512\nunit = "contacts"\nsource = "estimate"\n'
    with pytest.raises(ProfileError, match=re.escape("capacity.naconmax: unknown parameter")):
        platform.load_profile(write_profile(tmp_path, text))


@pytest.mark.parametrize("key", ["limit", "note"])
def test_profile_unknown_parameter_key_is_rejected(tmp_path, key):
    text = edit_line(official_text(), "capacity.njmax", "unit", f'unit = "constraint rows per world"\n{key} = "x"')
    with pytest.raises(ProfileError, match=re.escape(f"capacity.njmax: unknown key(s) ['{key}']")):
        platform.load_profile(write_profile(tmp_path, text))


@pytest.mark.parametrize(
    "table, line",
    [
        ("option.tolerance", "value = -1e-6"),
        ("option.timestep", "value = 0.0"),
        ("option.solver", 'value = "newton"'),
        ("contact.geom_solref", "value = [0.02]"),
        ("contact.geom_condim", "value = 5"),
        ("joint_coulomb_friction.l_index_finger_pip", "value = -0.01"),
        ("capacity.nconmax", "value = 0"),
        ("capacity.nconmax", "value = 128.0"),
        ("capacity.nconmax", 'value = "128"'),
        ("capacity.njmax", "value = -80"),
        ("capacity.njmax", "value = true"),
    ],
)
def test_profile_invalid_value_is_rejected(tmp_path, table, line):
    path = write_profile(tmp_path, edit_line(official_text(), table, "value", line))
    with pytest.raises(ProfileError, match=re.escape(table)):
        platform.load_profile(path)


@pytest.mark.parametrize(
    "override, message",
    [
        (
            '[overrides.joint_coulomb_friction.l_no_such_joint]\nvalue = 0.1\nreason = "test"\n',
            "joint_coulomb_friction.l_no_such_joint",
        ),
        ('[overrides.capacity.naconmax]\nvalue = 512\nreason = "test"\n', "capacity.naconmax"),
        ("[overrides.capacity.nconmax]\nvalue = 256\n", "reason"),
        ('[overrides.capacity.nconmax]\nvalue = 256\nreason = ""\n', "reason"),
        ('[overrides.capacity.nconmax]\nvalue = 0\nreason = "test"\n', "capacity.nconmax"),
        ('[overrides.option.tolerance]\nvalue = -1.0\nreason = "test"\n', "option.tolerance"),
        ('[overrides.bogus.x]\nvalue = 0.1\nreason = "test"\n', "bogus"),
    ],
)
def test_profile_bad_override_is_rejected(tmp_path, override, message):
    with pytest.raises(ProfileError, match=re.escape(message)):
        platform.load_profile(write_profile(tmp_path, official_text() + "\n" + override))


def test_profile_schema_is_the_mujoco_one_plus_capacity():
    groups = dict(platform.SCHEMA.groups)
    assert groups.pop("capacity") == ("nconmax", "njmax")
    assert groups == dict(cpu_platform.SCHEMA.groups)
    rules = {key: rule for key, rule in platform.SCHEMA.rules.items() if key[0] != "capacity"}
    assert rules == dict(cpu_platform.SCHEMA.rules)


@pytest.mark.gpu
@pytest.mark.parametrize("bias_rows", [1, 3])
def test_mit_ctrl_kernel_matches_reference(bias_rows):
    rng = np.random.default_rng(20261110)
    nworld, n, nq, nv, nu = 3, 20, 23, 22, 22
    qpos_adr = rng.permutation(nq)[:n]
    dof_adr = rng.permutation(nv)[:n]
    act_id = rng.permutation(nu)[:n]
    f32 = {
        "tau_limit": rng.uniform(0.2, 2.0, n),
        "qpos": rng.uniform(-1.5, 1.5, (nworld, nq)),
        "qvel": rng.uniform(-5.0, 5.0, (nworld, nv)),
        "q_des": rng.uniform(-1.5, 1.5, (nworld, n)),
        "qd_des": rng.uniform(-5.0, 5.0, (nworld, n)),
        "tau_ff": rng.uniform(-0.5, 0.5, (nworld, n)),
    }
    f32 = {name: value.astype(np.float32) for name, value in f32.items()}
    arrays = {name: wp.array(value, dtype=float, device=DEVICE) for name, value in f32.items()}
    biasprm = rng.uniform(-1.0, 1.0, (bias_rows, nu, mujoco.mjNBIAS)).astype(np.float32)
    biasprm[..., 1] = -rng.uniform(0.0, 3.0, (bias_rows, nu))
    biasprm[..., 2] = -rng.uniform(0.0, 0.1, (bias_rows, nu))
    actuator_biasprm = wp.array(biasprm, dtype=wp.types.vector(length=mujoco.mjNBIAS, dtype=float), device=DEVICE)
    ctrl = wp.array(np.full((nworld, nu), 7.0, dtype=np.float32), dtype=float, device=DEVICE)
    index = {
        name: wp.array(value.astype(np.int32), dtype=int, device=DEVICE)
        for name, value in (("qpos_adr", qpos_adr), ("dof_adr", dof_adr), ("act_id", act_id))
    }
    wp.launch(
        platform.mit_ctrl_kernel,
        dim=(nworld, n),
        inputs=[
            arrays["qpos"],
            arrays["qvel"],
            index["qpos_adr"],
            index["dof_adr"],
            index["act_id"],
            arrays["tau_limit"],
            actuator_biasprm,
            *(arrays[name] for name in MIT_FIELDS),
        ],
        outputs=[ctrl],
        device=DEVICE,
    )
    got = ctrl.numpy()

    world = np.arange(nworld) % bias_rows
    kp, kd = -biasprm[world][:, act_id, 1], -biasprm[world][:, act_id, 2]
    expected = (
        kp * f32["q_des"]
        + kd * f32["qd_des"]
        + f32["tau_ff"]
    )
    np.testing.assert_allclose(got[:, act_id], expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(expected).all()
    untouched = np.setdiff1d(np.arange(nu), act_id)
    assert (got[:, untouched] == 7.0).all()


@pytest.mark.gpu
def test_audit_backend_reports_tolerance_raised_by_put_model():
    mjm = mujoco.MjModel.from_xml_string(ONE_HINGE_XML.format(tolerance="1e-10"))
    imposed, checked = platform.audit_backend(mjm, mjw.put_model(mjm))
    assert imposed == [
        {
            "parameter": "option.tolerance",
            "requested": 1e-10,
            "effective": pytest.approx(1e-6, rel=1e-6),
            "note": "backend-imposed",
        }
    ]
    assert checked == BACKEND_CHECKED

    mjm = mujoco.MjModel.from_xml_string(ONE_HINGE_XML.format(tolerance="1e-6"))
    assert platform.audit_backend(mjm, mjw.put_model(mjm)) == ([], BACKEND_CHECKED)


@pytest.mark.gpu
def test_audit_backend_reports_model_fields_changed_on_the_device():
    mjm = mujoco.MjModel.from_xml_string(ONE_HINGE_XML.format(tolerance="1e-6"))
    m = mjw.put_model(mjm)
    m.dof_armature.assign(np.array([[0.123]], dtype=np.float32))
    assert platform.audit_backend(mjm, m) == (
        [
            {
                "parameter": "dof_armature[0]",
                "requested": float(mjm.dof_armature[0]),
                "effective": pytest.approx(0.123, rel=1e-6),
                "note": "backend-imposed",
            }
        ],
        BACKEND_CHECKED,
    )

    m = mjw.put_model(mjm)
    m.geom_friction.assign(np.array([[[0.5, 0.01, 0.001]]], dtype=np.float32))
    assert platform.audit_backend(mjm, m) == (
        [
            {
                "parameter": "geom_friction[0]",
                "requested": mjm.geom_friction[0].tolist(),
                "effective": pytest.approx([0.5, 0.01, 0.001], rel=1e-6),
                "note": "backend-imposed",
            }
        ],
        BACKEND_CHECKED,
    )


@pytest.mark.gpu
def test_model_load_steps_finite_without_backend_changes(mjcf, hand):
    loaded = platform.load(mjcf, nworld=2, scene=with_scene(hand))
    peak = {}
    for _ in range(100):
        mjw.step(loaded.m, loaded.d)
        platform.check_capacity(loaded, peak)
    for name in ("qpos", "qvel", "qacc", "ctrl", "qfrc_actuator"):
        assert np.isfinite(getattr(loaded.d, name).numpy()).all(), name
    assert loaded.nworld == 2
    assert loaded.d.qpos.shape == (2, loaded.model.nq)
    assert loaded.model.opt.tolerance == 1e-6
    assert loaded.m.opt.tolerance.numpy()[0] == pytest.approx(1e-6, rel=1e-6)
    assert peak["naconmax"] == 512 and peak["njmax"] == 1024
    assert 0 < peak["ncollision"] < 512 and peak["nefc_max"] < 1024

    report = loaded.report
    assert report["platform"] == "mujoco_warp"
    assert report["mjcf"] == str(mjcf)
    assert report["profile"] == {
        "platform": "mujoco_warp",
        "file": str(platform.PROFILE_PATH),
        "sha256": hashlib.sha256(platform.PROFILE_PATH.read_bytes()).hexdigest(),
    }
    assert report["backend_imposed"] == []
    assert report["backend_checked"] == BACKEND_CHECKED
    assert report["import_options"] == {
        "nworld": 2,
        "nconmax": 256,
        "njmax": 1024,
        "naconmax": 512,
        "broadphase": "NXN",
        "is_sparse": True,
        "precision": "float32",
        "device": DEVICE,
    }
    option = report["option"]
    assert option["tolerance"] == {
        "unit": "1",
        "model_source": 1e-10,
        "profile": 1e-6,
        "effective": 1e-6,
        "backend": pytest.approx(1e-6, rel=1e-6),
    }
    assert option["integrator"] == {
        "unit": "enum",
        "model_source": "RK4",
        "profile": "implicitfast",
        "effective": "implicitfast",
        "backend": "implicitfast",
    }
    assert option["timestep"]["backend"] == pytest.approx(0.005, rel=1e-6)
    assert {name: option[name]["backend"] for name in ("solver", "iterations", "ls_iterations")} == {
        "solver": "Newton",
        "iterations": 100,
        "ls_iterations": 50,
    }
    assert option["jacobian"] == {
        "unit": "enum",
        "model_source": "sparse",
        "profile": None,
        "effective": "sparse",
        "backend": "sparse",
    }
    override_entries = [entry for entry in report["overrides"] if entry["kind"] == "model_override"]
    armature_entries = [entry for entry in override_entries if entry["parameter"].startswith("joint_armature.")]
    torque_entries = [entry for entry in override_entries if entry["parameter"].startswith("joint_torque_limit.")]
    assert len(armature_entries) == len(torque_entries) == 20
    assert report["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert report["labels"] == [
        _mjcf.REPLACEMENT_LABEL,
        mit.CONTROL_PATH_LABEL,
        *(f"with override: {entry['parameter']}" for entry in override_entries),
    ]
    assert len(report["overrides"]) == 41
    replacement = report["overrides"][0]
    assert replacement["kind"] == "actuator_replacement"
    assert replacement["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert [entry["joint"] for entry in replacement["joints"]] == list(loaded.joint_names)
    for entry in replacement["joints"]:
        assert entry["model_actuator"]["type"] == "position"
        assert entry["model_actuator"]["kp"] > 0
        assert entry["model_actuator"]["kv"] > 0
        actuator = entry["actuator"]
        assert actuator["type"] == "general"
        assert actuator["forcerange"] == entry["model_actfrcrange"]
        assert actuator["forcelimited"] is True
        assert actuator["ctrllimited"] is False
        assert actuator["gainprm"] == [1.0, 0.0, 0.0]
        assert actuator["biasprm"][0] == 0.0
        assert actuator["biasprm"][1] == pytest.approx(-entry["kp"])
        assert actuator["biasprm"][2] == pytest.approx(-entry["kd"])
    effective = report["effective_values"]
    p = PREFIX[hand]
    joints = effective["joints"]
    assert set(joints) == set(loaded.joint_names)
    assert {j["frictionloss"] for j in joints.values()} == {0.0}
    expected_tau = official_model_tau_limits()[f"{p}index_finger_mcp_flex"]
    joint_entry = joints[f"{p}index_finger_mcp_flex"]
    assert joint_entry["model_armature"] == 0.0005
    assert joint_entry["armature"] == pytest.approx(0.00011304740105532539)
    assert joint_entry["actfrcrange"] == pytest.approx([-expected_tau, expected_tau], rel=1e-6)
    assert joint_entry["kp"] > 0
    assert joint_entry["kd"] > 0
    assert joint_entry["tau_limit"] == pytest.approx(expected_tau, rel=1e-6)
    assert joint_entry["control_path"] == mit.INTEGRATED_CONTROL_PATH

    actuator = effective["actuators"][f"{p}index_finger_mcp_flex_actuator"]
    assert actuator["forcerange"] == pytest.approx([-expected_tau, expected_tau], rel=1e-6)
    assert actuator["forcelimited"] is True
    assert actuator["ctrllimited"] is False
    assert actuator["gainprm0"] == 1.0
    assert actuator["biasprm"] == pytest.approx([0.0, -joint_entry["kp"], -joint_entry["kd"]])
    file_values = official_model_tau_limits()
    assert len(file_values) == 40
    for entry in torque_entries:
        name = entry["parameter"].split(".", 1)[1]
        value = file_values[name]
        assert entry["effective"] == pytest.approx(value, rel=1e-6)
        assert joints[name]["actfrcrange"] == pytest.approx([-value, value], rel=1e-6)
        assert effective["actuators"][f"{name}_actuator"]["forcerange"] == pytest.approx(
            [-value, value], rel=1e-6
        )
    for name, value in OFFICIAL_CONTACT.items():
        assert effective["contact"][name]["geoms"] == 49
        assert len(effective["contact"][name]["values"]) == 1
        assert effective["contact"][name]["values"][0] == pytest.approx(value, rel=1e-6)
    json.dumps(report, allow_nan=False)


@pytest.mark.gpu
def test_apply_mit_writes_reference_torque_to_ctrl(mjcf, hand):
    loaded = platform.load(mjcf, nworld=2, scene=with_scene(hand))
    qpos_adr, dof_adr, act_ids = list(loaded.qpos_adr), list(loaded.dof_adr), list(loaded.actuator_ids)
    assert loaded.mit_index.qpos_adr.numpy().tolist() == qpos_adr
    assert loaded.mit_index.dof_adr.numpy().tolist() == dof_adr
    assert loaded.mit_index.act_id.numpy().tolist() == act_ids
    assert loaded.mit_index.tau_limit.numpy().tolist() == pytest.approx(list(loaded.tau_limit), rel=1e-6)

    qpos = loaded.d.qpos.numpy()
    qpos[1, qpos_adr] = scene.pose_vector(loaded.joint_names, "pinch")
    loaded.d.qpos.assign(qpos)
    for _ in range(50):
        mjw.step(loaded.m, loaded.d)

    n = len(loaded.joint_names)
    buffers = platform.mit_buffers(loaded)
    for name in MIT_FIELDS:
        array = getattr(buffers, name)
        assert array.shape == (2, n) and array.dtype == wp.float32 and array.device == loaded.d.ctrl.device
        assert not array.numpy().any()
    rng = np.random.default_rng(7)
    kp = rng.uniform(0.0, 3.0, n)
    kd = rng.uniform(0.0, 0.1, n)
    platform.set_mit_gains(loaded, kp, kd)
    command = {
        "q_des": rng.uniform(-0.5, 1.5, (2, n)),
        "qd_des": rng.uniform(-2.0, 2.0, (2, n)),
        "tau_ff": rng.uniform(-0.3, 0.3, (2, n)),
    }
    command = {name: value.astype(np.float32) for name, value in command.items()}
    for name, value in command.items():
        getattr(buffers, name).assign(value)
    platform.apply_mit(loaded, buffers)
    ctrl, qpos, qvel = loaded.d.ctrl.numpy(), loaded.d.qpos.numpy(), loaded.d.qvel.numpy()
    assert not np.allclose(qpos[0, qpos_adr], qpos[1, qpos_adr])

    expected = (
        kp.astype(np.float32) * command["q_des"]
        + kd.astype(np.float32) * command["qd_des"]
        + command["tau_ff"]
    )
    np.testing.assert_allclose(ctrl[:, act_ids], expected, rtol=1e-5, atol=1e-6)

    mjw.forward(loaded.m, loaded.d)
    tau_limit = np.array(loaded.tau_limit)
    tau = kp * (command["q_des"] - qpos[:, qpos_adr]) + kd * (command["qd_des"] - qvel[:, dof_adr]) + command["tau_ff"]
    np.testing.assert_allclose(
        loaded.d.qfrc_actuator.numpy()[:, dof_adr], np.clip(tau, -tau_limit, tau_limit), rtol=1e-5, atol=1e-5
    )


@pytest.mark.gpu
def test_apply_mit_rejects_buffers_of_wrong_shape(right_mjcf):
    loaded = platform.load(right_mjcf, nworld=2)
    buffers = platform.mit_buffers(loaded)
    buffers.qd_des = wp.zeros((1, len(loaded.joint_names)), dtype=float, device=loaded.d.ctrl.device)
    with pytest.raises(ValueError, match="qd_des"):
        platform.apply_mit(loaded, buffers)


def test_mit_buffers_hold_only_the_command_and_reject_a_gain_field():
    assert platform.MIT_FIELDS == MIT_FIELDS
    buffers = platform.MitBuffers(**dict.fromkeys(MIT_FIELDS))
    with pytest.raises(AttributeError):
        buffers.kp = None


@pytest.mark.gpu
def test_set_mit_gains_writes_the_same_gains_to_the_host_and_device_models(right_mjcf):
    loaded = platform.load(right_mjcf, nworld=2)
    act_ids = list(loaded.actuator_ids)
    n = len(loaded.joint_names)
    kp = np.linspace(0.5, 2.0, n)
    kd = np.linspace(0.01, 0.05, n)
    # The hand model's own position-actuator gains differ, so a host model left untouched fails below.
    assert not np.allclose(loaded.model.actuator_biasprm[act_ids, 1], -kp)
    platform.set_mit_gains(loaded, kp, kd)
    host = loaded.model.actuator_biasprm[act_ids, 1:3]
    np.testing.assert_array_equal(host, np.stack([-kp, -kd], axis=1))
    np.testing.assert_array_equal(loaded.m.actuator_biasprm.numpy()[0, act_ids, 1:3], host.astype(np.float32))


@pytest.mark.gpu
def test_host_data_render_matches_cpu_forward(mjcf, hand):
    loaded = platform.load(mjcf, nworld=1, scene=with_scene(hand))
    for _ in range(20):
        mjw.step(loaded.m, loaded.d)
    stale = mujoco.MjData(loaded.model)
    mjw.get_data_into(stale, loaded.model, loaded.d)

    data = platform.host_data(loaded, 0)
    assert data is loaded.data
    np.testing.assert_array_equal(data.qpos, loaded.d.qpos.numpy()[0])
    reference = mujoco.MjData(loaded.model)
    reference.qpos[:] = data.qpos
    reference.qvel[:] = data.qvel
    mujoco.mj_forward(loaded.model, reference)
    # Without host_data's recomputation the copied poses lag qpos by one step while the fingers move.
    assert np.abs(stale.geom_xpos - reference.geom_xpos).max() > 1e-7
    np.testing.assert_array_equal(data.geom_xpos, reference.geom_xpos)
    np.testing.assert_array_equal(data.cam_xmat, reference.cam_xmat)
    # Multisampling makes repeated renders differ by a grey level in a few pixels; without it identical poses match.
    loaded.model.vis.quality.offsamples = 0
    with scene.SceneRenderer(loaded.model) as renderer:
        warp_image = renderer.render(data, "front")
        cpu_image = renderer.render(reference, "front")
    np.testing.assert_array_equal(warp_image, cpu_image)


@pytest.mark.gpu
def test_host_data_rejects_world_outside_batch(right_mjcf):
    loaded = platform.load(right_mjcf, nworld=2)
    for world_id in (-1, 2):
        with pytest.raises(ValueError, match="world_id"):
            platform.host_data(loaded, world_id)


@pytest.mark.gpu
def test_add_world_geoms_adds_every_hand_geom_shifted_and_no_floor(right_mjcf):
    loaded = platform.load(right_mjcf, nworld=2, scene=with_scene("right"))
    model = loaded.model
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    scn = mujoco.MjvScene(model, maxgeom=2 * model.ngeom)
    offset = np.array([0.0, 0.3, 0.0])
    platform.add_world_geoms(loaded, 1, offset, option, mujoco.MjvPerturb(), scn)

    data = platform.host_data(loaded, 1)
    shown = [g for g in range(model.ngeom) if option.geomgroup[model.geom_group[g]]]
    hand_geoms = {g for g in shown if model.geom_bodyid[g] != 0}
    assert any(model.geom_bodyid[g] == 0 for g in shown)
    assert any(model.body(model.geom_bodyid[g]).name == "r_wrist" for g in hand_geoms)
    added = [scn.geoms[i] for i in range(scn.ngeom)]
    assert sorted(int(geom.objid) for geom in added) == sorted(hand_geoms)
    for geom in added:
        np.testing.assert_allclose(geom.pos, data.geom_xpos[geom.objid] + offset, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("origin", ["code", "profile-file"])
def test_capacity_override_reaches_put_data(tmp_path, right_mjcf, origin):
    reason = f"test {origin} capacity override"
    if origin == "code":
        loaded = platform.load(
            right_mjcf, nworld=2, overrides={"capacity": {"nconmax": {"value": 512, "reason": reason}}}
        )
    else:
        text = official_text() + f'\n[overrides.capacity.nconmax]\nvalue = 512\nreason = "{reason}"\n'
        loaded = platform.load(right_mjcf, nworld=2, profile_path=write_profile(tmp_path, text))
    assert loaded.d.naconmax == 1024
    assert loaded.report["import_options"]["nconmax"] == 512
    assert loaded.report["import_options"]["naconmax"] == 1024
    assert loaded.report["labels"][:2] == [_mjcf.REPLACEMENT_LABEL, mit.CONTROL_PATH_LABEL]
    assert "with override: capacity.nconmax" in loaded.report["labels"]
    assert loaded.report["overrides"][1] == {
        "kind": "profile_override",
        "parameter": "capacity.nconmax",
        "baseline": 256,
        "value": 512,
        "effective": 512,
        "unit": "contacts per world",
        "reason": reason,
        "origin": origin,
    }


@pytest.mark.gpu
def test_override_for_joint_of_other_hand_is_rejected(right_mjcf):
    overrides = {"joint_coulomb_friction": {"l_index_finger_pip": {"value": 0.02, "reason": "test"}}}
    with pytest.raises(ProfileError, match="l_index_finger_pip"):
        platform.load(right_mjcf, nworld=1, overrides=overrides)


@pytest.mark.gpu
def test_code_override_sets_joint_friction_on_the_warp_model(right_mjcf):
    overrides = {"joint_coulomb_friction": {"r_index_finger_pip": {"value": 0.02, "reason": "test sweep"}}}
    loaded = platform.load(right_mjcf, nworld=1, overrides=overrides)
    dof = int(loaded.model.jnt_dofadr[loaded.model.joint("r_index_finger_pip").id])
    assert loaded.m.dof_frictionloss.numpy()[0, dof] == pytest.approx(0.02, rel=1e-6)
    entry = loaded.report["overrides"][1]
    assert (entry["parameter"], entry["origin"], entry["effective"]) == (
        "joint_coulomb_friction.r_index_finger_pip",
        "code",
        0.02,
    )
    assert entry["backend"] == pytest.approx(0.02, rel=1e-6)
    assert loaded.report["effective_values"]["joints"]["r_index_finger_pip"]["frictionloss"] == pytest.approx(0.02)
    assert loaded.report["backend_imposed"] == []


@pytest.mark.parametrize("nworld", [0, -1, 1.0, True])
def test_load_rejects_invalid_nworld(right_mjcf, nworld):
    with pytest.raises(ValueError, match="nworld"):
        platform.load(right_mjcf, nworld=nworld)


@pytest.mark.gpu
def test_check_capacity_tracks_peak_and_raises_at_capacity(right_mjcf):
    loaded = platform.load(right_mjcf, nworld=2, scene=with_scene("right"))
    peak = {}
    for _ in range(10):
        mjw.step(loaded.m, loaded.d)
        platform.check_capacity(loaded, peak)
    usage = platform.capacity_usage(loaded)
    assert set(usage) == {"ncollision", "nacon", "naconmax", "nefc_max", "njmax", "overflow"}
    assert usage["overflow"] == 0
    assert (usage["naconmax"], usage["njmax"]) == (512, 1024)
    assert set(peak) == set(usage)
    assert all(peak[key] >= usage[key] for key in ("ncollision", "nacon", "nefc_max"))

    # Overflowing for real is unsafe in-process (MJWarp may hit CUDA error 700), so set the counters directly.
    counters = {
        "ncollision": (loaded.d.ncollision, [512], [511], "ncollision=512 >= naconmax=512"),
        "nacon": (loaded.d.nacon, [512], [511], "nacon=512 >= naconmax=512"),
        "nefc": (loaded.d.nefc, [0, 1024], [1023, 0], "nefc=1024 >= njmax=1024 in world 1"),
        # 256 is the EPA_HORIZON bit, which MJWarp only warns about while dropping the contact.
        "overflow": (loaded.d.overflow, [0, 256], [0, 0], "overflow bitmap 256 in world 1"),
    }
    for array, full, below, message in counters.values():
        saved = array.numpy().copy()
        array.assign(np.array(below, dtype=np.int32))
        platform.check_capacity(loaded, peak)
        array.assign(np.array(full, dtype=np.int32))
        with pytest.raises(RuntimeError, match=re.escape(message)):
            platform.check_capacity(loaded, peak)
        array.assign(saved)
    assert (peak["ncollision"], peak["nacon"], peak["nefc_max"], peak["overflow"]) == (512, 512, 1024, 256)


@pytest.mark.gpu
def test_mit_stability_at_the_profile_timestep(right_mjcf):
    loaded = platform.load(right_mjcf, nworld=1)
    n = len(loaded.joint_names)
    kp = np.full(n, scene.DEMO_KP)
    kd = np.full(n, scene.DEMO_KD)
    report = platform.mit_stability(loaded, kp, kd)
    assert report["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert report["dt"] == 0.005
    assert report["criterion"] == _mjcf.MIT_STABILITY_CRITERION
    expected_ratios = np.array(kp) * report["dt"] ** 2 / np.array(loaded.joint_inertia)
    np.testing.assert_allclose(
        report["max_ratio"], expected_ratios.max(), rtol=1e-6, atol=1e-8
    )
    expected_index = max(range(n), key=expected_ratios.__getitem__)
    assert report["max_joint"] == loaded.joint_names[expected_index]
    assert list(report["joints"]) == list(loaded.joint_names)
    json.dumps(report, allow_nan=False)
    unstable_kp = np.array(loaded.joint_inertia) * (report["limit"] + 0.01) / report["dt"] ** 2
    with pytest.raises(mit.MitStabilityError, match=r"MIT integrated actuator stability bound violated"):
        platform.mit_stability(loaded, unstable_kp, kd)

@pytest.mark.gpu
def test_model_override_shared_file_applies_only_loaded_hand(tmp_path, description_root):
    csv_thumb = 0.0001457162066834878
    body = "".join(
        f"[joint_armature.{prefix}{suffix}]\n"
        f"value = {csv_thumb if suffix == 'thumb_cmc_flex' else 0.001}\n"
        'unit = "kg*m^2"\n'
        for prefix in ("l_", "r_")
        for suffix in conventions.JOINT_SUFFIXES
    )
    model_overrides_path = tmp_path / "model_overrides.toml"
    model_overrides_path.write_text(
        'product = "hand2_2"\n'
        'description_version = "v2026.11.18"\n\n'
        "[joint_armature]\nsource = \"test source\"\n\n"
        f"{body}"
        "[joint_torque_limit]\nsource = \"test source\"\n",
        encoding="utf-8",
    )
    loaded = platform.load(description_root / "mjcf" / "left_with_glove.xml", nworld=1, model_overrides_path=model_overrides_path)
    applied = [entry for entry in loaded.report["overrides"] if entry["kind"] == "model_override"]
    assert len(applied) == 20
    assert all(entry["parameter"].startswith("joint_armature.l_") for entry in applied)
    assert loaded.report["effective_values"]["joints"]["l_thumb_cmc_flex"]["armature"] == pytest.approx(csv_thumb)


@pytest.mark.gpu
def test_model_override_unknown_joint_fails_loudly(tmp_path, description_root):
    model_overrides_path = tmp_path / "model_overrides.toml"
    model_overrides_path.write_text(
        'product = "hand2_2"\n'
        'description_version = "v2026.11.18"\n\n'
        '[joint_armature]\nsource = "test source"\n\n'
        '[joint_armature.r_no_such_joint]\nvalue = 0.001\nunit = "kg*m^2"\n\n'
        '[joint_torque_limit]\nsource = "test source"\n',
        encoding="utf-8",
    )
    with pytest.raises(ProfileError, match="r_no_such_joint"):
        platform.load(description_root / "mjcf" / "right_with_glove.xml", nworld=1, model_overrides_path=model_overrides_path)


@pytest.mark.gpu
def test_tau_limit_file_supersedes_matching_model_override(tmp_path, right_mjcf):
    name = "r_index_finger_pip"
    model_overrides_path = tmp_path / "model_overrides.toml"
    model_overrides_path.write_text(
        'product = "hand2_2"\n'
        'description_version = "v2026.11.18"\n\n'
        '[joint_armature]\nsource = "test source"\n\n'
        f"[joint_torque_limit.{name}]\nvalue = 0.071\nunit = \"N*m\"\n\n"
        '[joint_torque_limit]\nsource = "test source"\n',
        encoding="utf-8",
    )
    tau_values = {f"r_{suffix}": round(0.109 + 0.01 * j, 3) for j, suffix in enumerate(conventions.JOINT_SUFFIXES)}
    tau_values[name] = 0.173
    tau_path = tmp_path / "tau_limit.toml"
    tau_path.write_text(
        'source = "synthetic test limits"\nunit = "N*m"\n[tau_limit]\n'
        + "".join(f"{joint} = {value}\n" for joint, value in tau_values.items()),
        encoding="utf-8",
    )
    loaded = platform.load(
        right_mjcf,
        nworld=1,
        model_overrides_path=model_overrides_path,
        tau_limits=profiles.load_tau_limits(tau_path),
    )

    joint_id = loaded.model.joint(name).id
    assert loaded.model.jnt_actfrcrange[joint_id].tolist() == [-tau_values[name], tau_values[name]]
    override_entry = next(item for item in loaded.report["overrides"] if item.get("parameter") == f"joint_torque_limit.{name}")
    tau_entry = next(item for item in loaded.report["overrides"] if item["kind"] == "tau_limit_override")
    assert override_entry["superseded"] is True
    assert "effective" not in override_entry
    assert tau_entry["source"] == "synthetic test limits"
    assert tau_entry["joints"][name]["effective"] == [-tau_values[name], tau_values[name]]
    assert f"with override: joint_torque_limit.{name}" not in loaded.report["labels"]
    assert "with override: tau_limit (synthetic test limits)" in loaded.report["labels"]
