# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import re
import tomllib
from collections import Counter

import mujoco
import numpy as np
import pytest

from wuji_sim import conventions, framing, mit, profiles, scene
from wuji_sim.platforms import _mjcf
from wuji_sim.platforms import mujoco as platform
from wuji_sim.profiles import ProfileError

HANDS = ("left", "right")
PREFIX = {"left": "l_", "right": "r_"}
MUJOCO_DEFAULT_CONTACT = {
    "geom_friction": [1.0, 0.005, 0.0001],
    "geom_solref": [0.02, 1.0],
    "geom_solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
    "geom_condim": 3,
}
OFFICIAL_CONTACT = {**MUJOCO_DEFAULT_CONTACT, "geom_friction": [0.7, 0.005, 0.0001], "geom_condim": 4}
# Deliberately not the MuJoCo defaults, so a profile value that never reaches the geoms is caught.
PROFILE_CONTACT = {
    "geom_friction": [0.6, 0.004, 0.0003],
    "geom_solref": [0.015, 1.2],
    "geom_solimp": [0.85, 0.97, 0.002, 0.45, 3.0],
    "geom_condim": 6,
}
MODEL_TAU_LIMIT = (0.6, 0.6, 0.3, 0.3) + (2.0, 0.2, 0.3, 0.3) * 4
# Synthetic values, all different from the hand model; the ring and pinky abd ones exceed it, so a clamp left at the hand model value shows up.
SYNTHETIC_TAU_LIMIT = tuple(round(0.105 + 0.01 * j, 3) for j in range(20))
TAU_LIMIT_SOURCE = "synthetic test limits"
TAU_LIMIT_HEAD = f'source = "{TAU_LIMIT_SOURCE}"\nunit = "N*m"\n'


def official_model_tau_limits():
    document = tomllib.loads(profiles.MODEL_OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {
        name: entry["value"]
        for name, entry in document["joint_torque_limit"].items()
        if name != "source"
    }


def hand_model_tau_limits(hand):
    values = official_model_tau_limits()
    prefix = PREFIX[hand]
    return {f"{prefix}{suffix}": values[f"{prefix}{suffix}"] for suffix in conventions.JOINT_SUFFIXES}

TWO_JOINT_XML = """
<mujoco>
  <worldbody>
    <body name="b1">
      <joint name="j1" axis="1 0 0" range="-1 1" actuatorfrcrange="-0.5 0.5"/>
      <geom type="capsule" size="0.01 0.05"/>
      <body name="b2" pos="0 0 -0.1">
        <joint name="j2" axis="1 0 0" range="-1 1" actuatorfrcrange="{j2_frc}"/>
        <geom type="capsule" size="0.01 0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="2" kv="0.1" ctrlrange="-1 1" forcerange="-0.5 0.5"/>
    {a2}
  </actuator>
</mujoco>
"""


def two_joint_spec(j2_frc="-0.25 0.25", a2='<position name="a2" joint="j2" kp="3" kv="0.2"/>'):
    return mujoco.MjSpec.from_string(TWO_JOINT_XML.format(j2_frc=j2_frc, a2=a2))


def actuator_kinds(spec):
    kinds = Counter()
    for act in spec.actuators:
        if act.name.endswith("_actuator"):
            kinds["general"] += 1
        elif act.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
            kinds["position"] += 1
        elif (
            act.gaintype == mujoco.mjtGain.mjGAIN_FIXED
            and act.biastype == mujoco.mjtBias.mjBIAS_NONE
            and act.gainprm[0] == 1.0
        ):
            kinds["motor"] += 1
        else:
            kinds["other"] += 1
    return dict(kinds)


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
    path = tmp_path / "mujoco.toml"
    path.write_text(text, encoding="utf-8")
    return path


def hinge_names(spec):
    return [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]


def right_tau_limits():
    return {f"r_{suffix}": value for suffix, value in zip(conventions.JOINT_SUFFIXES, SYNTHETIC_TAU_LIMIT)}


def tau_limit_text(values, head=TAU_LIMIT_HEAD):
    return head + "[tau_limit]\n" + "".join(f"{name} = {value}\n" for name, value in values.items())


def write_tau_limits(tmp_path, text):
    path = tmp_path / "tau_limit.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(params=HANDS)
def hand(request):
    return request.param


@pytest.fixture
def mjcf(description_root, hand):
    return description_root / "mjcf" / f"{hand}_with_glove.xml"


@pytest.fixture
def right_mjcf(description_root):
    return description_root / "mjcf" / "right_with_glove.xml"


def test_replacement_turns_position_actuators_into_integrated_general_actuators():
    spec = two_joint_spec()
    entry = _mjcf.replace_position_actuators(spec)
    assert actuator_kinds(spec) == {"general": 2}
    model = spec.compile()
    for joint, frc in (("j1", [-0.5, 0.5]), ("j2", [-0.25, 0.25])):
        jid = model.joint(joint).id
        aid = model.actuator(f"{joint}_actuator").id
        assert model.actuator_trnid[aid, 0] == jid
        assert model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
        assert model.actuator_forcelimited[aid] == 1
        assert model.actuator_ctrllimited[aid] == 0
        assert model.actuator_forcerange[aid].tolist() == model.jnt_actfrcrange[jid].tolist() == frc
    assert entry["kind"] == "actuator_replacement"
    assert entry["label"] == _mjcf.REPLACEMENT_LABEL
    assert entry["temporary"] is True
    assert entry["reason"]
    assert entry["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert entry["joints"][0] == {
        "joint": "j1",
        "model_actuator": {
            "name": "a1",
            "type": "position",
            "kp": 2.0,
            "kv": 0.1,
            "ctrlrange": [-1.0, 1.0],
            "forcerange": [-0.5, 0.5],
        },
        "model_actfrcrange": [-0.5, 0.5],
        "kp": 2.0,
        "kd": 0.1,
        "tau_limit": 0.5,
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "actuator": {
            "name": "j1_actuator",
            "type": "general",
            "gaintype": "fixed",
            "biastype": "affine",
            "gainprm": [1.0, 0.0, 0.0],
            "biasprm": [0.0, -2.0, -0.1],
            "forcerange": [-0.5, 0.5],
            "forcelimited": True,
            "ctrllimited": False,
        },
    }
    assert entry["joints"][1]["model_actuator"]["kp"] == 3.0
    assert entry["joints"][1]["model_actuator"]["kv"] == 0.2

def test_replacement_rejects_non_position_actuator():
    spec = two_joint_spec(a2='<motor name="a2" joint="j2"/>')
    with pytest.raises(ProfileError, match="a2"):
        _mjcf.replace_position_actuators(spec)


@pytest.mark.parametrize("j2_frc", ["-0.25 0.2", "0 0"])
def test_replacement_rejects_asymmetric_or_zero_force_limit(j2_frc):
    with pytest.raises(ProfileError, match="j2"):
        _mjcf.replace_position_actuators(two_joint_spec(j2_frc=j2_frc))


def test_model_replacement_uses_general_actuator_with_joint_force_range(mjcf):
    spec = mujoco.MjSpec.from_file(str(mjcf))
    assert actuator_kinds(spec) == {"position": 20}
    entry = _mjcf.replace_position_actuators(spec)
    assert actuator_kinds(spec) == {"general": 20}
    assert len(entry["joints"]) == 20
    model = spec.compile()
    assert model.nu == 20
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        assert model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
        assert model.actuator(aid).name == f"{model.joint(jid).name}_actuator"
        assert model.actuator_forcelimited[aid] == 1
        assert model.actuator_ctrllimited[aid] == 0
        assert model.actuator_forcerange[aid].tolist() == model.jnt_actfrcrange[jid].tolist()

def test_official_profile_loads():
    profile = platform.load_profile()
    assert profile["platform"] == "mujoco"
    assert profile["path"] == str(platform.PROFILE_PATH)
    assert profile["sha256"] == hashlib.sha256(platform.PROFILE_PATH.read_bytes()).hexdigest()
    assert set(profile["params"]) == {"option", "contact", "joint_coulomb_friction", "joint_damping"}
    option = profile["params"]["option"]
    assert {name: option[name]["value"] for name in _mjcf.OPTION_FIELDS} == {
        "timestep": 0.005,
        "integrator": "implicitfast",
        "solver": "Newton",
        "iterations": 100,
        "tolerance": 1e-8,
        "ls_iterations": 50,
    }
    assert {p["source"] for name, p in option.items() if name != "timestep"} == {"default: mjlab 1.6.0"}
    contact = profile["params"]["contact"]
    assert {name: contact[name]["value"] for name in _mjcf.CONTACT_FIELDS} == OFFICIAL_CONTACT
    assert {name: p["source"] for name, p in contact.items()} == {
        "geom_friction": "choice",
        "geom_solref": "default: MuJoCo 3.11.0",
        "geom_solimp": "default: MuJoCo 3.11.0",
        "geom_condim": "choice",
    }
    friction = profile["params"]["joint_coulomb_friction"]
    assert {(p["value"], p["unit"], p["source"]) for p in friction.values()} == {
        (0.0, "N*m", "estimate")
    }
    assert profile["overrides"] == {}


def test_profile_joint_friction_covers_both_description_hands(description_root):
    names = set()
    for hand in HANDS:
        names |= set(hinge_names(mujoco.MjSpec.from_file(str(description_root / "mjcf" / f"{hand}_with_glove.xml"))))
    assert len(names) == 40
    assert set(platform.load_profile()["params"]["joint_coulomb_friction"]) == names


@pytest.mark.parametrize("key", ["value", "unit"])
@pytest.mark.parametrize(
    "table",
    [
        "option.timestep",
        "contact.geom_solimp",
        "joint_coulomb_friction.r_index_finger_pip",
    ],
)
def test_profile_missing_field_is_rejected(tmp_path, table, key):
    path = write_profile(tmp_path, edit_line(official_text(), table, key))
    with pytest.raises(ProfileError, match=re.escape(f"{table}: missing '{key}'")):
        platform.load_profile(path)


@pytest.mark.parametrize("group", ["option", "contact", "joint_coulomb_friction", "joint_damping"])
def test_profile_missing_group_source_is_rejected(tmp_path, group):
    path = write_profile(tmp_path, edit_line(official_text(), group, "source"))
    with pytest.raises(ProfileError, match=re.escape(f"{group}: missing 'source'")):
        platform.load_profile(path)


@pytest.mark.parametrize("table, key", [("option.solver", "unit"), ("option", "source")])
def test_profile_empty_string_field_is_rejected(tmp_path, table, key):
    path = write_profile(tmp_path, edit_line(official_text(), table, key, f'{key} = ""'))
    with pytest.raises(ProfileError, match=re.escape(f"{table}: '{key}' must be a non-empty string")):
        platform.load_profile(path)


def test_profile_parameter_source_replaces_the_group_source(tmp_path):
    text = edit_line(official_text(), "option.solver", "unit", 'unit = "enum"\nsource = "test source"')
    option = platform.load_profile(write_profile(tmp_path, text))["params"]["option"]
    assert option["solver"]["source"] == "test source"
    assert option["iterations"]["source"] == "default: mjlab 1.6.0"


def test_profile_unknown_group_is_rejected(tmp_path):
    text = official_text() + '\n[bogus.x]\nvalue = 1.0\nunit = "1"\nsource = "estimate"\n'
    with pytest.raises(ProfileError, match="bogus"):
        platform.load_profile(write_profile(tmp_path, text))


def test_profile_removed_gain_scale_group_is_rejected(tmp_path):
    text = official_text() + '\n[mit_fw_gain_scale]\nsource = "x"\n'
    with pytest.raises(ProfileError, match=r"unknown group.*mit_fw_gain_scale"):
        platform.load_profile(write_profile(tmp_path, text))


def test_profile_missing_group_is_rejected(tmp_path):
    with pytest.raises(ProfileError, match=re.escape("contact: missing group")):
        platform.load_profile(write_profile(tmp_path, drop_tables(official_text(), "contact")))


def test_profile_missing_option_field_is_rejected(tmp_path):
    with pytest.raises(ProfileError, match="option.ls_iterations"):
        platform.load_profile(write_profile(tmp_path, drop_tables(official_text(), "option.ls_iterations")))


def test_profile_unknown_option_field_is_rejected(tmp_path):
    text = official_text() + '\n[option.gravity]\nvalue = 0.0\nunit = "m/s^2"\nsource = "estimate"\n'
    with pytest.raises(ProfileError, match="option.gravity"):
        platform.load_profile(write_profile(tmp_path, text))


@pytest.mark.parametrize("key", ["limit", "note"])
def test_profile_unknown_parameter_key_is_rejected(tmp_path, key):
    text = edit_line(official_text(), "option.timestep", "unit", f'unit = "s"\n{key} = "x"')
    with pytest.raises(ProfileError, match=re.escape(f"option.timestep: unknown key(s) ['{key}']")):
        platform.load_profile(write_profile(tmp_path, text))


@pytest.mark.parametrize(
    "table, line",
    [
        ("option.timestep", "value = -0.002"),
        ("option.timestep", "value = true"),
        ("option.integrator", 'value = "RK45"'),
        ("option.solver", 'value = "newton"'),
        ("option.iterations", "value = 1.5"),
        ("option.ls_iterations", "value = 0"),
        ("option.tolerance", "value = -1e-8"),
        ("contact.geom_friction", "value = [1.0, 0.005]"),
        ("contact.geom_solimp", 'value = [0.9, 0.95, 0.001, 0.5, "2"]'),
        ("contact.geom_condim", "value = 2"),
        ("contact.geom_condim", "value = true"),
        ("joint_coulomb_friction.r_index_finger_pip", "value = -0.01"),
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
            '[overrides.joint_coulomb_friction.r_no_such_joint]\nvalue = 0.1\nreason = "test"\n',
            "joint_coulomb_friction.r_no_such_joint",
        ),
        ('[overrides.joint_coulomb_friction.r_index_finger_pip]\nvalue = 0.1\n', "reason"),
        ('[overrides.joint_coulomb_friction.r_index_finger_pip]\nvalue = 0.1\nreason = ""\n', "reason"),
        ('[overrides.option.timestep]\nvalue = -0.1\nreason = "test"\n', "option.timestep"),
        ('[overrides.option.timestep]\nvalue = 0.001\nreason = "test"\nunit = "s"\n', "option.timestep"),
        ('[overrides.bogus.x]\nvalue = 0.1\nreason = "test"\n', "bogus"),
    ],
)
def test_profile_bad_override_is_rejected(tmp_path, override, message):
    with pytest.raises(ProfileError, match=re.escape(message)):
        platform.load_profile(write_profile(tmp_path, official_text() + "\n" + override))


def test_load_applies_profile_and_reports_model_sources(mjcf, hand):
    loaded = platform.load(mjcf)
    model, report = loaded.model, loaded.report
    assert model.opt.timestep == 0.005
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.iterations == 100
    assert model.opt.tolerance == 1e-8
    assert model.opt.ls_iterations == 50
    option = report["option"]
    profile_values = {
        "timestep": 0.005,
        "integrator": "implicitfast",
        "solver": "Newton",
        "iterations": 100,
        "tolerance": 1e-8,
        "ls_iterations": 50,
    }
    assert {name: option[name]["effective"] for name in _mjcf.OPTION_FIELDS} == profile_values
    assert {name: option[name]["profile"] for name in _mjcf.OPTION_FIELDS} == profile_values
    assert {name: option[name]["model_source"] for name in (*_mjcf.OPTION_FIELDS, "jacobian")} == {
        "timestep": 0.002,
        "integrator": "RK4",
        "solver": "Newton",
        "iterations": 50,
        "tolerance": 1e-10,
        "ls_iterations": 50,
        "jacobian": "sparse",
    }
    assert option["jacobian"]["profile"] is None
    assert option["jacobian"]["effective"] == "sparse"
    assert option["timestep"]["unit"] == "s"

    dofs = list(loaded.dof_adr)
    assert model.dof_frictionloss[dofs].tolist() == [0.0] * 20
    armature_entries = [
        entry for entry in report["overrides"]
        if entry["kind"] == "model_override" and entry["parameter"].startswith("joint_armature.")
    ]
    torque_entries = [
        entry for entry in report["overrides"]
        if entry["kind"] == "model_override" and entry["parameter"].startswith("joint_torque_limit.")
    ]
    assert len(armature_entries) == len(torque_entries) == 20
    assert all(entry["source"].startswith("estimate") for entry in (*armature_entries, *torque_entries))
    assert model.dof_armature[dofs].tolist() == pytest.approx([entry["effective"] for entry in armature_entries])

    assert report["platform"] == "mujoco"
    assert report["mjcf"] == str(mjcf)
    assert report["profile"] == {
        "platform": "mujoco",
        "file": str(platform.PROFILE_PATH),
        "sha256": hashlib.sha256(platform.PROFILE_PATH.read_bytes()).hexdigest(),
    }
    assert report["labels"] == [
        _mjcf.REPLACEMENT_LABEL,
        mit.CONTROL_PATH_LABEL,
        *(f"with override: {entry['parameter']}" for entry in (*armature_entries, *torque_entries)),
    ]
    replacement = report["overrides"][0]
    assert replacement["kind"] == "actuator_replacement"
    assert replacement["label"] == _mjcf.REPLACEMENT_LABEL
    assert replacement["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert [entry["joint"] for entry in replacement["joints"]] == list(loaded.joint_names)
    tau_limits = official_model_tau_limits()
    for entry in replacement["joints"]:
        assert entry["model_actuator"]["type"] == "position"
        assert entry["model_actuator"]["kp"] > 0
        assert entry["model_actuator"]["kv"] > 0
        actuator = entry["actuator"]
        expected_tau = tau_limits[entry["joint"]]
        assert actuator["type"] == "general"
        assert actuator["forcerange"] == entry["model_actfrcrange"]
        assert actuator["forcerange_effective"] == pytest.approx([-expected_tau, expected_tau])
        assert actuator["forcelimited_effective"] is True
        assert actuator["ctrllimited_effective"] is False
    assert len(report["overrides"]) == 41
    assert loaded.tau_limit == tuple(tau_limits[name] for name in loaded.joint_names)
    assert report["backend_imposed"] == []
    assert report["backend_checked"] == []
    assert report["import_options"] == {}
    json.dumps(report, allow_nan=False)

    p = PREFIX[hand]
    effective = report["effective_values"]
    assert set(effective) == {"units", "joints", "actuators", "contact"}
    expected_tau = tau_limits[f"{p}index_finger_mcp_flex"]
    joint_entry = effective["joints"][f"{p}index_finger_mcp_flex"]
    assert joint_entry["model_armature"] == 0.0005
    assert joint_entry["frictionloss"] == 0.0
    assert joint_entry["damping"] == 0.0
    assert joint_entry["armature"] == 0.00011304740105532539
    assert joint_entry["model_actfrcrange"] == [-2.0, 2.0]
    assert joint_entry["actfrcrange"] == [-expected_tau, expected_tau]
    assert joint_entry["range"] == [-1.047, 1.57]
    assert joint_entry["kp"] > 0.0
    assert joint_entry["kd"] > 0.0
    assert joint_entry["tau_limit"] == expected_tau
    assert joint_entry["control_path"] == mit.INTEGRATED_CONTROL_PATH
    actuator = effective["actuators"][f"{p}index_finger_mcp_flex_actuator"]
    assert actuator["joint"] == f"{p}index_finger_mcp_flex"
    assert actuator["forcerange"] == [-expected_tau, expected_tau]
    assert actuator["forcelimited"] is True
    assert actuator["ctrllimited"] is False
    assert actuator["gainprm0"] == 1.0
    assert actuator["biasprm"] == pytest.approx([0.0, -joint_entry["kp"], -joint_entry["kd"]])
    assert effective["contact"] == {
        name: {"values": [value], "geoms": 48} for name, value in OFFICIAL_CONTACT.items()
    }


def test_loaded_indices_match_model(mjcf, hand):
    loaded = platform.load(mjcf)
    model = loaded.model
    assert loaded.joint_names == tuple(PREFIX[hand] + suffix for suffix in conventions.JOINT_SUFFIXES)
    for i, name in enumerate(loaded.joint_names):
        jid = model.joint(name).id
        assert loaded.qpos_adr[i] == model.jnt_qposadr[jid]
        assert loaded.dof_adr[i] == model.jnt_dofadr[jid]
        assert model.actuator_trnid[loaded.actuator_ids[i], 0] == jid
        assert loaded.tau_limit[i] == model.jnt_actfrcrange[jid, 1] == -model.jnt_actfrcrange[jid, 0]
    for values in (loaded.qpos_adr, loaded.dof_adr, loaded.actuator_ids):
        assert {type(v) for v in values} == {int}
    assert {type(v) for v in loaded.tau_limit} == {float}
    assert loaded.data.xpos[model.body(f"{PREFIX[hand]}middle_finger_proximal").id].any()


def test_code_override_sets_joint_friction(mjcf, hand):
    name = f"{PREFIX[hand]}index_finger_pip"
    overrides = {"joint_coulomb_friction": {name: {"value": 0.02, "reason": "test friction sweep"}}}
    loaded = platform.load(mjcf, overrides=overrides)
    model = loaded.model
    dof = model.jnt_dofadr[model.joint(name).id]
    assert model.dof_frictionloss[dof] == 0.02
    assert [model.dof_frictionloss[d] for d in loaded.dof_adr if d != dof] == [0.0] * 19
    assert loaded.report["labels"][0] == _mjcf.REPLACEMENT_LABEL
    assert f"with override: joint_coulomb_friction.{name}" in loaded.report["labels"]
    profile_entry = next(
        entry for entry in loaded.report["overrides"] if entry.get("parameter") == f"joint_coulomb_friction.{name}"
    )
    assert profile_entry == {
        "kind": "profile_override",
        "parameter": f"joint_coulomb_friction.{name}",
        "baseline": 0.0,
        "value": 0.02,
        "effective": 0.02,
        "unit": "N*m",
        "reason": "test friction sweep",
        "origin": "code",
    }
    assert loaded.report["effective_values"]["joints"][name]["frictionloss"] == 0.02


def test_override_for_joint_of_other_hand_is_rejected(right_mjcf):
    overrides = {"joint_coulomb_friction": {"l_index_finger_pip": {"value": 0.02, "reason": "test"}}}
    with pytest.raises(ProfileError, match="l_index_finger_pip"):
        platform.load(right_mjcf, overrides=overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"joint_coulomb_friction": {"r_index_finger_pip": {"value": 0.02}}}, "reason"),
        ({"joint_coulomb_friction": {"r_index_finger_pip": {"value": 0.02, "reason": ""}}}, "reason"),
        ({"joint_coulomb_friction": {"r_index_finger_pip": {"value": -0.02, "reason": "x"}}}, "r_index_finger_pip"),
        ({"option": {"bogus": {"value": 1, "reason": "x"}}}, "option.bogus"),
        ({"capacity": {"nconmax": {"value": 128, "reason": "x"}}}, "capacity"),
    ],
)
def test_code_override_is_validated(right_mjcf, overrides, message):
    with pytest.raises(ProfileError, match=re.escape(message)):
        platform.load(right_mjcf, overrides=overrides)


def test_profile_file_override_is_applied(tmp_path, right_mjcf):
    text = official_text() + (
        '\n[overrides.joint_coulomb_friction.r_index_finger_pip]\nvalue = 0.05\nreason = "test profile-file override"\n'
    )
    loaded = platform.load(right_mjcf, profile_path=write_profile(tmp_path, text))
    model = loaded.model
    assert model.dof_frictionloss[model.jnt_dofadr[model.joint("r_index_finger_pip").id]] == 0.05
    assert "with override: joint_coulomb_friction.r_index_finger_pip" in loaded.report["labels"]
    entry = loaded.report["overrides"][1]
    assert entry["origin"] == "profile-file"
    assert (entry["baseline"], entry["value"], entry["effective"]) == (0.0, 0.05, 0.05)


def test_same_override_in_profile_file_and_code_is_rejected(tmp_path, right_mjcf):
    text = official_text() + (
        '\n[overrides.joint_coulomb_friction.r_index_finger_pip]\nvalue = 0.05\nreason = "profile-file"\n'
    )
    overrides = {"joint_coulomb_friction": {"r_index_finger_pip": {"value": 0.02, "reason": "code"}}}
    with pytest.raises(ProfileError, match="r_index_finger_pip"):
        platform.load(right_mjcf, profile_path=write_profile(tmp_path, text), overrides=overrides)


def test_profile_without_a_model_joint_is_rejected(tmp_path, right_mjcf):
    path = write_profile(tmp_path, drop_tables(official_text(), "joint_coulomb_friction.r_pinky_dip"))
    platform.load_profile(path)
    with pytest.raises(ProfileError, match="r_pinky_dip"):
        platform.load(right_mjcf, profile_path=path)


def test_profile_file_values_drive_option(tmp_path, right_mjcf):
    text = edit_line(official_text(), "option.timestep", "value", "value = 0.001")
    text = edit_line(text, "option.integrator", "value", 'value = "Euler"')
    path = write_profile(tmp_path, text)
    loaded = platform.load(right_mjcf, profile_path=path)
    assert loaded.model.opt.timestep == 0.001
    assert loaded.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert loaded.report["option"]["integrator"]["effective"] == "Euler"
    assert loaded.report["profile"]["file"] == str(path.resolve())
    assert loaded.report["profile"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_profile_contact_values_reach_floor_and_every_hand_collision_geom(tmp_path, mjcf, hand):
    text = official_text()
    for name, value in PROFILE_CONTACT.items():
        text = edit_line(text, f"contact.{name}", "value", f"value = {value}")
    path = write_profile(tmp_path, text)
    view = framing.view(mjcf, hand, platform.PLATFORM, "tracking")
    loaded = platform.load(mjcf, scene=lambda spec: scene.add_scene(spec, hand, view), profile_path=path)
    model = loaded.model
    fields = {
        "geom_friction": model.geom_friction,
        "geom_solref": model.geom_solref,
        "geom_solimp": model.geom_solimp,
        "geom_condim": model.geom_condim,
    }
    floor = model.geom("scene_floor").id
    collision = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    assert floor in collision
    assert len([g for g in collision if g != floor]) == 48
    for g in collision:
        for name, array in fields.items():
            assert array[g].tolist() == PROFILE_CONTACT[name], (model.geom(g).name, name)
    assert loaded.report["effective_values"]["contact"] == {
        name: {"values": [value], "geoms": 49} for name, value in PROFILE_CONTACT.items()
    }
    assert [entry for entry in loaded.report["overrides"] if entry["kind"] not in {"actuator_replacement", "model_override"}] == []


def test_contact_override_reaches_collision_geoms_only(right_mjcf):
    requested = {
        "geom_friction": [0.7, 0.01, 0.0002],
        "geom_solref": [0.01, 1.5],
        "geom_solimp": [0.8, 0.9, 0.002, 0.4, 3.0],
        "geom_condim": 6,
    }
    overrides = {"contact": {name: {"value": value, "reason": "test"} for name, value in requested.items()}}
    view = framing.view(right_mjcf, "right", platform.PLATFORM, "tracking")
    loaded = platform.load(right_mjcf, scene=lambda spec: scene.add_scene(spec, "right", view), overrides=overrides)
    model = loaded.model
    fields = {
        "geom_friction": model.geom_friction,
        "geom_solref": model.geom_solref,
        "geom_solimp": model.geom_solimp,
        "geom_condim": model.geom_condim,
    }
    for g in range(model.ngeom):
        is_collision = bool(model.geom_contype[g] or model.geom_conaffinity[g])
        for name, array in fields.items():
            expected = requested[name] if is_collision else MUJOCO_DEFAULT_CONTACT[name]
            assert array[g].tolist() == expected, (model.geom(g).name, name)
    assert loaded.report["effective_values"]["contact"]["geom_solref"] == {"values": [[0.01, 1.5]], "geoms": 49}
    profile_entries = [entry for entry in loaded.report["overrides"] if entry["kind"] == "profile_override"]
    assert [entry["parameter"] for entry in profile_entries] == [
        "contact.geom_friction",
        "contact.geom_solref",
        "contact.geom_solimp",
        "contact.geom_condim",
    ]
    assert profile_entries[0]["effective"] == [[0.7, 0.01, 0.0002]]


def test_joint_inertia_is_the_mj_fullM_diagonal_at_qpos0(mjcf, hand):
    view = framing.view(mjcf, hand, platform.PLATFORM, "tracking")
    loaded = platform.load(mjcf, scene=lambda spec: scene.add_scene(spec, hand, view))
    data = mujoco.MjData(loaded.model)
    mujoco.mj_forward(loaded.model, data)
    full = np.zeros((loaded.model.nv, loaded.model.nv))
    mujoco.mj_fullM(loaded.model, data, full)
    assert list(loaded.joint_inertia) == np.diag(full)[list(loaded.dof_adr)].tolist()
    assert min(loaded.joint_inertia) > 0.0


def test_mit_stability_records_every_joint_below_the_limit(right_mjcf):
    loaded = platform.load(right_mjcf)
    kp = np.full(len(loaded.joint_names), scene.DEMO_KP)
    kd = np.full(len(loaded.joint_names), scene.DEMO_KD)
    report = platform.mit_stability(loaded, kp, kd)
    assert report["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert report["dt"] == 0.005
    assert report["limit"] == mit.MIT_STABILITY_LIMIT == 4.0
    assert report["criterion"] == _mjcf.MIT_STABILITY_CRITERION
    assert list(report["joints"]) == list(loaded.joint_names)
    for name, inertia in zip(loaded.joint_names, loaded.joint_inertia):
        entry = report["joints"][name]
        assert entry == {
            "kp": scene.DEMO_KP,
            "kd": scene.DEMO_KD,
            "inertia": inertia,
            "ratio": scene.DEMO_KP * 0.005**2 / inertia,
            "kp_dt2_over_I": scene.DEMO_KP * 0.005**2 / inertia,
            "kd_dt_over_I": scene.DEMO_KD * 0.005 / inertia,
        }
    assert report["max_joint"] == max(report["joints"], key=lambda name: report["joints"][name]["ratio"])
    assert report["max_ratio"] == max(entry["ratio"] for entry in report["joints"].values())
    assert report["max_ratio"] > 0.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("ratio", [3.99, 4.01])
def test_mit_stability_limit_on_one_joint(right_mjcf, ratio):
    loaded = platform.load(right_mjcf)
    index = loaded.joint_names.index("r_index_finger_pip")
    kp = np.full(len(loaded.joint_names), scene.DEMO_KP)
    kd = np.full(len(loaded.joint_names), scene.DEMO_KD)
    kp[index] = ratio * loaded.joint_inertia[index] / loaded.model.opt.timestep**2
    if ratio < mit.MIT_STABILITY_LIMIT:
        report = platform.mit_stability(loaded, kp, kd)
        assert report["max_joint"] == "r_index_finger_pip"
        assert report["max_ratio"] == pytest.approx(ratio)
        return
    with pytest.raises(mit.MitStabilityError) as excinfo:
        platform.mit_stability(loaded, kp, kd)
    message = str(excinfo.value)
    assert "integrated actuator" in message
    assert "r_index_finger_pip" in message and "r_middle_finger_dip" not in message
    for part in (
        f"kp {kp[index]:g}",
        "dt 0.005 s",
        f"I {loaded.joint_inertia[index]:.4e}",
        "kp*dt^2/I 4.010",
        "lower kp or the timestep",
    ):
        assert part in message


def test_mit_stability_rejects_gains_of_wrong_length(right_mjcf):
    loaded = platform.load(right_mjcf)
    with pytest.raises(ValueError, match="one per joint"):
        platform.mit_stability(loaded, [scene.DEMO_KP] * 19, [scene.DEMO_KD] * 20)


def test_tau_limit_file_is_read_with_source_and_digest(tmp_path):
    path = write_tau_limits(tmp_path, tau_limit_text(right_tau_limits()))
    assert profiles.load_tau_limits(path) == {
        "file": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": TAU_LIMIT_SOURCE,
        "unit": "N*m",
        "values": right_tau_limits(),
    }


@pytest.mark.parametrize(
    "text, message",
    [
        ('unit = "N*m"\n[tau_limit]\nr_pinky_dip = 0.2\n', "missing ['source']"),
        (TAU_LIMIT_HEAD + 'comment = "x"\n[tau_limit]\nr_pinky_dip = 0.2\n', "unknown ['comment']"),
        ('source = " "\nunit = "N*m"\n[tau_limit]\nr_pinky_dip = 0.2\n', "'source' must be a non-empty string"),
        ('source = 1\nunit = "N*m"\n[tau_limit]\nr_pinky_dip = 0.2\n', "'source' must be a non-empty string"),
        ('source = "x"\nunit = "N*mm"\n[tau_limit]\nr_pinky_dip = 0.2\n', "'unit' must be 'N*m', got 'N*mm'"),
        (TAU_LIMIT_HEAD + "[tau_limit]\n", "'tau_limit' must be a non-empty table"),
        (TAU_LIMIT_HEAD + "tau_limit = 0.2\n", "'tau_limit' must be a non-empty table"),
        (TAU_LIMIT_HEAD + "[tau_limit\n", "cannot load tau limit file"),
    ],
)
def test_tau_limit_file_layout_is_validated(tmp_path, text, message):
    path = write_tau_limits(tmp_path, text)
    with pytest.raises(ProfileError, match=re.escape(message)) as excinfo:
        profiles.load_tau_limits(path)
    assert str(path.resolve()) in str(excinfo.value)


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "true", '"0.2"', "[0.2]"])
def test_tau_limit_file_rejects_a_non_positive_or_non_numeric_value(tmp_path, value):
    path = write_tau_limits(tmp_path, tau_limit_text({**right_tau_limits(), "r_pinky_dip": value}))
    with pytest.raises(ProfileError, match=re.escape("tau_limit.r_pinky_dip: value")) as excinfo:
        profiles.load_tau_limits(path)
    assert str(path.resolve()) in str(excinfo.value)


def test_missing_tau_limit_file_is_reported(tmp_path):
    path = tmp_path / "absent.toml"
    with pytest.raises(ProfileError, match=re.escape(f"cannot load tau limit file {path.resolve()}")):
        profiles.load_tau_limits(path)


def test_tau_limit_file_replaces_both_model_clamps(tmp_path, right_mjcf):
    path = write_tau_limits(tmp_path, tau_limit_text(right_tau_limits()))
    overrides = {"joint_coulomb_friction": {"r_index_finger_pip": {"value": 0.02, "reason": "test"}}}
    loaded = platform.load(right_mjcf, overrides=overrides, tau_limits=profiles.load_tau_limits(path))
    model = loaded.model
    assert loaded.tau_limit == SYNTHETIC_TAU_LIMIT
    for name, value in zip(loaded.joint_names, SYNTHETIC_TAU_LIMIT):
        assert model.jnt_actfrcrange[model.joint(name).id].tolist() == [-value, value]
        assert model.actuator_forcerange[model.actuator(f"{name}_actuator").id].tolist() == [-value, value]
    label = "with override: tau_limit (synthetic test limits)"
    assert loaded.report["labels"][-1] == label
    assert f"with override: joint_coulomb_friction.r_index_finger_pip" in loaded.report["labels"]
    non_armature = [entry for entry in loaded.report["overrides"] if entry["kind"] != "model_override"]
    assert [entry["kind"] for entry in non_armature] == [
        "actuator_replacement",
        "profile_override",
        "tau_limit_override",
    ]
    tau_entry = next(entry for entry in non_armature if entry["kind"] == "tau_limit_override")
    assert tau_entry == {
        "kind": "tau_limit_override",
        "parameter": "tau_limit",
        "label": label,
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "file": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": TAU_LIMIT_SOURCE,
        "unit": "N*m",
        "joints": {
            f"r_{suffix}": {
                "baseline": [-official_model_tau_limits()[f"r_{suffix}"], official_model_tau_limits()[f"r_{suffix}"]],
                "value": [-value, value],
                "effective": [-value, value],
            }
            for suffix, value in zip(conventions.JOINT_SUFFIXES, SYNTHETIC_TAU_LIMIT)
        },
    }
    json.dumps(loaded.report, allow_nan=False)


def test_tau_limit_file_bounds_the_applied_torque(tmp_path, right_mjcf):
    path = write_tau_limits(tmp_path, tau_limit_text(right_tau_limits()))
    loaded = platform.load(right_mjcf, tau_limits=profiles.load_tau_limits(path))
    model, data = loaded.model, loaded.data
    data.ctrl[list(loaded.actuator_ids)] = 10.0
    mujoco.mj_forward(model, data)
    assert data.qfrc_actuator[list(loaded.dof_adr)].tolist() == list(SYNTHETIC_TAU_LIMIT)


def test_tau_limit_file_without_a_model_joint_is_rejected(tmp_path, right_mjcf):
    values = right_tau_limits()
    del values["r_pinky_dip"]
    path = write_tau_limits(tmp_path, tau_limit_text(values))
    tau_limits = profiles.load_tau_limits(path)
    with pytest.raises(ProfileError, match=re.escape("missing ['r_pinky_dip']")) as excinfo:
        platform.load(right_mjcf, tau_limits=tau_limits)
    assert str(path.resolve()) in str(excinfo.value)


def test_tau_limit_file_with_a_joint_of_the_other_hand_is_rejected(tmp_path, right_mjcf):
    path = write_tau_limits(tmp_path, tau_limit_text({**right_tau_limits(), "l_index_finger_pip": 0.2}))
    tau_limits = profiles.load_tau_limits(path)
    with pytest.raises(ProfileError, match=re.escape("unknown ['l_index_finger_pip']")) as excinfo:
        platform.load(right_mjcf, tau_limits=tau_limits)
    assert str(path.resolve()) in str(excinfo.value)


def test_tau_limits_need_the_integrated_actuator_for_every_joint(tmp_path):
    path = write_tau_limits(tmp_path, tau_limit_text({"j1": 0.4, "j2": 0.15}))
    with pytest.raises(ProfileError, match="j1_actuator") as excinfo:
        _mjcf.apply_tau_limits(two_joint_spec(), profiles.load_tau_limits(path))
    assert str(path.resolve()) in str(excinfo.value)


def write_model_overrides(tmp_path, *, version="v2026.11.18", body=""):
    path = tmp_path / "model_overrides.toml"
    path.write_text(
        'product = "hand2_2"\n'
        f'description_version = "{version}"\n\n'
        '[joint_armature]\nsource = "test source"\n'
        f"{body}\n"
        '[joint_torque_limit]\nsource = "test source"\n',
        encoding="utf-8",
    )
    return path


def test_model_override_version_mismatch_fails_loudly(tmp_path, right_mjcf):
    with pytest.raises(ProfileError, match="description_version"):
        platform.load(right_mjcf, model_overrides_path=write_model_overrides(tmp_path, version="wrong"))


def test_model_override_unknown_joint_fails_loudly(tmp_path, right_mjcf):
    path = write_model_overrides(tmp_path, body="[joint_armature.r_no_such_joint]\nvalue = 0.001\nunit = \"kg*m^2\"\n")
    with pytest.raises(ProfileError, match="r_no_such_joint"):
        platform.load(right_mjcf, model_overrides_path=path)


def test_model_override_wrong_unit_fails_loudly(tmp_path, right_mjcf):
    path = write_model_overrides(tmp_path, body="[joint_armature.r_index_finger_pip]\nvalue = 0.001\nunit = \"N*m\"\n")
    with pytest.raises(ProfileError, match="unit"):
        platform.load(right_mjcf, model_overrides_path=path)


def test_model_armature_override_covers_every_joint_with_estimate_source(right_mjcf):
    loaded = platform.load(right_mjcf)
    joints = loaded.report["effective_values"]["joints"]
    entries = [
        entry for entry in loaded.report["overrides"]
        if entry["kind"] == "model_override" and entry["parameter"].startswith("joint_armature.")
    ]
    assert len(entries) == len(joints) == 20
    assert all(entry["source"] == "estimate" for entry in entries)
    assert all(joints[name]["armature"] != joints[name]["model_armature"] for name in joints)
    assert all(entry["frictionloss"] == 0.0 and entry["damping"] == 0.0 for entry in joints.values())

def test_model_torque_override_covers_every_joint_with_estimate_source(mjcf, hand):
    file_values = official_model_tau_limits()
    assert len(file_values) == 40
    expected = hand_model_tau_limits(hand)
    loaded = platform.load(mjcf)
    joints = loaded.report["effective_values"]["joints"]
    actuators = loaded.report["effective_values"]["actuators"]
    entries = {
        entry["parameter"].split(".", 1)[1]: entry
        for entry in loaded.report["overrides"]
        if entry["kind"] == "model_override" and entry["parameter"].startswith("joint_torque_limit.")
    }
    assert set(entries) == set(expected) == set(joints)
    assert all(entry["source"] == "estimate: 1.5 A current limit × torque constant" for entry in entries.values())
    for name, value in expected.items():
        joint = joints[name]
        assert loaded.tau_limit[loaded.joint_names.index(name)] == pytest.approx(value, rel=1e-6)
        assert joint["actfrcrange"] == pytest.approx([-value, value], rel=1e-6)
        assert actuators[f"{name}_actuator"]["forcerange"] == pytest.approx([-value, value], rel=1e-6)
        entry = entries[name]
        assert entry["effective"] == pytest.approx(value, rel=1e-6)
        assert entry["unit"] == "N*m"
        assert entry["model_value"] == entry["baseline"] == pytest.approx(joint["model_actfrcrange"][1])


def test_model_override_applies_and_records_model_value(tmp_path, right_mjcf):
    name = "r_index_finger_pip"
    path = write_model_overrides(
        tmp_path,
        body=f"[joint_armature.{name}]\nvalue = 0.001\nunit = \"kg*m^2\"\n",
    )
    loaded = platform.load(right_mjcf, model_overrides_path=path)
    joint = loaded.report["effective_values"]["joints"][name]
    assert joint["armature"] == pytest.approx(0.001)
    entry = next(item for item in loaded.report["overrides"] if item.get("parameter") == f"joint_armature.{name}")
    assert entry["model_value"] == entry["baseline"] == pytest.approx(joint["model_armature"])
    assert entry["effective"] == pytest.approx(0.001)
    assert entry["source"] == "test source"

def test_model_override_shared_file_applies_only_loaded_hand(tmp_path, description_root):
    csv_thumb = 0.0001457162066834878
    body = "".join(
        f"[joint_armature.{prefix}{suffix}]\n"
        f"value = {csv_thumb if suffix == 'thumb_cmc_flex' else 0.001}\n"
        'unit = "kg*m^2"\n'
        for prefix in ("l_", "r_")
        for suffix in conventions.JOINT_SUFFIXES
    )
    path = write_model_overrides(tmp_path, body=body)
    loaded = platform.load(description_root / "mjcf" / "left_with_glove.xml", model_overrides_path=path)
    applied = [entry for entry in loaded.report["overrides"] if entry["kind"] == "model_override"]
    assert len(applied) == 20
    assert all(entry["parameter"].startswith("joint_armature.l_") for entry in applied)
    assert loaded.report["effective_values"]["joints"]["l_thumb_cmc_flex"]["armature"] == pytest.approx(csv_thumb)


def test_tau_limit_file_supersedes_matching_model_override(tmp_path, right_mjcf):
    name = "r_index_finger_pip"
    model_overrides_path = write_model_overrides(
        tmp_path,
        body=f"[joint_torque_limit.{name}]\nvalue = 0.071\nunit = \"N*m\"\n",
    )
    tau_values = right_tau_limits()
    tau_values[name] = 0.173
    tau_path = write_tau_limits(tmp_path, tau_limit_text(tau_values))
    tau_limits = profiles.load_tau_limits(tau_path)
    loaded = platform.load(right_mjcf, model_overrides_path=model_overrides_path, tau_limits=tau_limits)

    joint_id = loaded.model.joint(name).id
    assert loaded.model.jnt_actfrcrange[joint_id].tolist() == [-tau_values[name], tau_values[name]]
    override_entry = next(item for item in loaded.report["overrides"] if item.get("parameter") == f"joint_torque_limit.{name}")
    tau_entry = next(item for item in loaded.report["overrides"] if item["kind"] == "tau_limit_override")
    assert override_entry["superseded"] is True
    assert "effective" not in override_entry
    assert tau_entry["source"] == TAU_LIMIT_SOURCE
    assert tau_entry["joints"][name]["effective"] == [-tau_values[name], tau_values[name]]
    assert f"with override: joint_torque_limit.{name}" not in loaded.report["labels"]
    assert f"with override: tau_limit ({TAU_LIMIT_SOURCE})" in loaded.report["labels"]
