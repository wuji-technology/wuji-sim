# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco

from wuji_sim import mit, profiles

REPLACEMENT_LABEL = "with override: position actuators replaced by integrated affine actuators"
REPLACEMENT_REASON = (
    "The hand model v2026.11.18 ships <position> actuators with built-in kp/kv; wuji-sim replaces each with a general actuator "
    "whose affine bias integrates the same kp/kd and whose forcerange is the joint actuatorfrcrange"
)
OPTION_FIELDS = ("timestep", "integrator", "solver", "iterations", "tolerance", "ls_iterations")
CONTACT_FIELDS = ("geom_friction", "geom_solref", "geom_solimp", "geom_condim")
CONDIMS = (1, 3, 4, 6)
TAU_LIMIT_LABEL = "with override: tau_limit"

INTEGRATORS = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
IMPLICIT_INTEGRATORS = ("implicit", "implicitfast")
SOLVERS = {"PGS": mujoco.mjtSolver.mjSOL_PGS, "CG": mujoco.mjtSolver.mjSOL_CG, "Newton": mujoco.mjtSolver.mjSOL_NEWTON}
JACOBIANS = {
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    "auto": mujoco.mjtJacobian.mjJAC_AUTO,
}
EFFECTIVE_UNITS = {
    "frictionloss": "N*m",
    "damping": "N*m*s/rad",
    "armature": "kg*m^2",
    "actfrcrange": "N*m",
    "range": "rad",
    "ctrlrange": "N*m",
    "gainprm0": "1",
    "biasprm": "[N*m, N*m/rad, N*m*s/rad]",
}
MIT_STABILITY_CRITERION = (
    "kp_j*dt^2/I_j < 4 for every integrated actuator joint; I_j is the diagonal of the compiled joint-space "
    "inertia matrix (mj_fullM, armature included) at the initial pose qpos0"
)
EXPLICIT_MIT_STABILITY_CRITERION = (
    "kd_j*dt/I_j <= 2 for every explicit torque joint; I_j is the diagonal of the compiled joint-space inertia "
    "matrix (mj_fullM, armature included) at the initial pose qpos0"
)
# None: the parameter names are model joint names, checked against the model in apply_profile.
PROFILE_GROUPS = {
    "option": OPTION_FIELDS,
    "contact": CONTACT_FIELDS,
    "joint_coulomb_friction": None,
    "joint_damping": None,
}
VALUE_RULES = {
    ("option", "timestep"): ("a positive number", lambda v: profiles.is_number(v) and v > 0),
    ("option", "integrator"): (f"one of {list(INTEGRATORS)}", lambda v: isinstance(v, str) and v in INTEGRATORS),
    ("option", "solver"): (f"one of {list(SOLVERS)}", lambda v: isinstance(v, str) and v in SOLVERS),
    ("option", "iterations"): ("a positive integer", profiles.is_count),
    ("option", "tolerance"): ("a non-negative number", lambda v: profiles.is_number(v) and v >= 0),
    ("option", "ls_iterations"): ("a positive integer", profiles.is_count),
    ("contact", "geom_friction"): ("a list of 3 numbers", profiles.is_numbers(3)),
    ("contact", "geom_solref"): ("a list of 2 numbers", profiles.is_numbers(2)),
    ("contact", "geom_solimp"): ("a list of 5 numbers", profiles.is_numbers(5)),
    ("contact", "geom_condim"): (f"one of {list(CONDIMS)}", lambda v: profiles.is_count(v) and v in CONDIMS),
    ("joint_coulomb_friction", None): ("a non-negative number", lambda v: profiles.is_number(v) and v >= 0),
    ("joint_damping", None): ("a non-negative number", lambda v: profiles.is_number(v) and v >= 0),
}
# Model arrays the load report reads; MuJoCo reads them from the host model, MuJoCo Warp from the device model.
REPORT_FIELDS = (
    "dof_armature", "dof_frictionloss", "dof_damping", "jnt_actfrcrange", "jnt_range",
    "actuator_ctrlrange", "actuator_forcerange", "actuator_forcelimited", "actuator_ctrllimited",
    "actuator_gainprm", "actuator_biasprm", "geom_friction", "geom_solref", "geom_solimp", "geom_condim",
)


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _is_position_actuator(act) -> bool:
    return (
        act.trntype == mujoco.mjtTrn.mjTRN_JOINT
        and act.gaintype == mujoco.mjtGain.mjGAIN_FIXED
        and act.biastype == mujoco.mjtBias.mjBIAS_AFFINE
        and act.biasprm[1] == -act.gainprm[0]
    )


# The hand model currently provides <position> actuators; replace them with general actuators for MIT control.
def replace_position_actuators(spec: mujoco.MjSpec) -> dict:
    hinges = [joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE]
    actuators = list(spec.actuators)
    by_joint = {}
    for act in actuators:
        if not _is_position_actuator(act):
            raise profiles.ProfileError(
                f"actuator {act.name!r} is not a hand-model <position> actuator; the position-to-general "
                "replacement is outdated for this MJCF and must be removed"
            )
        if act.target in by_joint:
            raise profiles.ProfileError(f"joint {act.target!r} is driven by more than one <position> actuator")
        by_joint[act.target] = act
    hinge_names = [joint.name for joint in hinges]
    if sorted(by_joint) != sorted(hinge_names):
        raise profiles.ProfileError(
            f"<position> actuators drive {sorted(by_joint)}; the replacement expects one per hinge joint {hinge_names}"
        )
    joints = []
    for joint in hinges:
        lo, hi = _floats(joint.actfrcrange)
        if not (hi > 0 and lo == -hi):
            raise profiles.ProfileError(f"joint {joint.name!r}: actuatorfrcrange {[lo, hi]} must be symmetric and non-zero")
        act = by_joint[joint.name]
        kp = float(act.gainprm[0])
        kd = -float(act.biasprm[2])
        joints.append(
            {
                "joint": joint.name,
                "model_actuator": {
                    "name": act.name,
                    "type": "position",
                    "kp": kp,
                    "kv": kd,
                    "ctrlrange": _floats(act.ctrlrange),
                    "forcerange": _floats(act.forcerange),
                },
                "model_actfrcrange": [lo, hi],
                "kp": kp,
                "kd": kd,
                "tau_limit": hi,
                "control_path": mit.INTEGRATED_CONTROL_PATH,
                "actuator": {
                    "name": f"{joint.name}_actuator",
                    "type": "general",
                    "gaintype": "fixed",
                    "biastype": "affine",
                    "gainprm": [1.0, 0.0, 0.0],
                    "biasprm": [0.0, -kp, -kd],
                    "forcerange": [lo, hi],
                    "forcelimited": True,
                    "ctrllimited": False,
                },
            }
        )
    for act in actuators:
        spec.delete(act)
    for entry in joints:
        actuator = spec.add_actuator(
            name=entry["actuator"]["name"], target=entry["joint"], trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        actuator.set_to_motor()
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        for index, value in enumerate(entry["actuator"]["gainprm"]):
            actuator.gainprm[index] = value
        for index, value in enumerate(entry["actuator"]["biasprm"]):
            actuator.biasprm[index] = value
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.forcerange = entry["actuator"]["forcerange"]
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return {
        "kind": "actuator_replacement",
        "label": REPLACEMENT_LABEL,
        "temporary": True,
        "reason": REPLACEMENT_REASON,
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "joints": joints,
    }


def apply_model_overrides(spec: mujoco.MjSpec, document: dict) -> list[dict]:
    hinges = {joint.name: joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE}
    actuators = {act.name: act for act in spec.actuators}
    prefixes = {name[:2] for name in hinges if name.startswith(("l_", "r_"))}
    model_prefix = next(iter(prefixes)) if len(prefixes) == 1 else None
    other_prefix = {"l_": "r_", "r_": "l_"}.get(model_prefix)
    applied = []
    for group, table in document["groups"].items():
        for name, entry in table["values"].items():
            if name not in hinges:
                # The shared file carries both hand variants; a single-hand MJCF uses only its side.
                if (
                    other_prefix is not None
                    and name.startswith(other_prefix)
                    and f"{model_prefix}{name[2:]}" in hinges
                ):
                    continue
                raise profiles.ProfileError(f"{group}.{name}: {name!r} is not a hinge joint of this model")
            joint = hinges[name]
            value = entry["value"]
            if group == "joint_armature":
                baseline = float(joint.armature)
                joint.armature = value
            else:
                baseline = float(joint.actfrcrange[1])
                limit = [-value, value]
                joint.actfrcrange = limit
                actuator_name = f"{name}_actuator"
                if actuator_name not in actuators:
                    raise profiles.ProfileError(f"joint {name!r} has no actuator {actuator_name!r} for model torque override")
                actuators[actuator_name].forcerange = limit
            applied.append({
                "kind": "model_override",
                "parameter": f"{group}.{name}",
                "baseline": baseline,
                "model_value": baseline,
                "value": value,
                "unit": entry["unit"],
                "source": table["source"],
                "origin": "model_overrides.toml",
            })
    return applied


def apply_profile(
    spec: mujoco.MjSpec, profile: dict, schema: profiles.Schema, overrides: Mapping | None = None
) -> list[dict]:
    if profile["platform"] != schema.platform:
        raise profiles.ProfileError(f"profile {profile['path']} is for platform {profile['platform']!r}, not {schema.platform!r}")
    params = profile["params"]
    sources = {
        "profile-file": profiles.check_overrides(schema, profile["overrides"], params, f"profile {profile['path']}"),
        "code": profiles.check_overrides(schema, overrides or {}, params, "code overrides"),
    }
    both = sorted({(g, n) for g, n, _ in sources["profile-file"]} & {(g, n) for g, n, _ in sources["code"]})
    if both:
        targets = ", ".join(f"{g}.{n}" for g, n in both)
        raise profiles.ProfileError(f"{targets}: overridden both in profile {profile['path']} and in code overrides")

    hinges = [joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_names = {joint.name for joint in hinges}
    joint_groups = [group for group, fields in schema.groups.items() if fields is None]
    requested = {group: {name: entry["value"] for name, entry in table.items()} for group, table in params.items()}
    applied = []
    for origin, entries in sources.items():
        for group, name, entry in entries:
            if group in joint_groups and name not in hinge_names:
                raise profiles.ProfileError(
                    f"overrides.{group}.{name}: {name!r} is not a hinge joint of this model ({origin} override)"
                )
            requested[group][name] = entry["value"]
            baseline = params[group][name]
            applied.append(
                {
                    "kind": "profile_override",
                    "parameter": f"{group}.{name}",
                    "baseline": baseline["value"],
                    "value": entry["value"],
                    "unit": baseline["unit"],
                    "reason": entry["reason"],
                    "origin": origin,
                }
            )
    for group in joint_groups:
        missing = [name for name in sorted(hinge_names) if name not in requested[group]]
        if missing:
            raise profiles.ProfileError(f"{group}: no entry for model joint(s) {missing} (profile {profile['path']})")

    option = requested["option"]
    spec.option.timestep = option["timestep"]
    spec.option.integrator = INTEGRATORS[option["integrator"]]
    spec.option.solver = SOLVERS[option["solver"]]
    spec.option.iterations = option["iterations"]
    spec.option.tolerance = option["tolerance"]
    spec.option.ls_iterations = option["ls_iterations"]
    contact = requested["contact"]
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.friction = contact["geom_friction"]
            geom.solref = contact["geom_solref"]
            geom.solimp = contact["geom_solimp"]
            geom.condim = contact["geom_condim"]
    for joint in hinges:
        joint.frictionloss = requested["joint_coulomb_friction"][joint.name]
        joint.damping = [requested["joint_damping"][joint.name], 0.0, 0.0]
    return applied


def apply_tau_limits(spec: mujoco.MjSpec, tau_limits: Mapping) -> dict:
    where = f"tau limit file {tau_limits['file']}"
    hinges = [joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE]
    values = tau_limits["values"]
    names = [joint.name for joint in hinges]
    missing, unknown = sorted(set(names) - set(values)), sorted(set(values) - set(names))
    if missing or unknown:
        raise profiles.ProfileError(
            f"tau_limit must name exactly the model's hinge joints: missing {missing}, unknown {unknown} ({where})"
        )
    actuators = {act.name: act for act in spec.actuators}
    actuator_list = []
    for joint in hinges:
        name = f"{joint.name}_actuator"
        if name not in actuators:
            raise profiles.ProfileError(f"joint {joint.name!r} has no actuator {name!r} to take its tau limit ({where})")
        actuator_list.append(actuators[name])
    joints = {}
    for joint, actuator in zip(hinges, actuator_list):
        limit = [-values[joint.name], values[joint.name]]
        joints[joint.name] = {"baseline": _floats(joint.actfrcrange), "value": limit}
        joint.actfrcrange = limit
        actuator.forcerange = limit
    return {
        "kind": "tau_limit_override",
        "parameter": "tau_limit",
        "label": f"{TAU_LIMIT_LABEL} ({tau_limits['source']})",
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "file": tau_limits["file"],
        "sha256": tau_limits["sha256"],
        "source": tau_limits["source"],
        "unit": tau_limits["unit"],
        "joints": joints,
    }


def enum_name(names: dict, value) -> str:
    return {int(enum): name for name, enum in names.items()}[int(value)]


def read_option(opt) -> dict:
    return {
        "timestep": float(opt.timestep),
        "integrator": enum_name(INTEGRATORS, opt.integrator),
        "solver": enum_name(SOLVERS, opt.solver),
        "iterations": int(opt.iterations),
        "tolerance": float(opt.tolerance),
        "ls_iterations": int(opt.ls_iterations),
        "jacobian": enum_name(JACOBIANS, opt.jacobian),
    }


def _unique(rows: list, ids: list[int]) -> list:
    values = []
    for i in ids:
        if rows[i] not in values:
            values.append(rows[i])
    return values


def _joint_inertia(model: mujoco.MjModel, data: mujoco.MjData, hinge_ids: list[int]) -> tuple[float, ...]:
    # Platform files may not import numpy (mj_fullM needs it); each CSR row of data.M ends at its diagonal entry.
    dofs = [int(model.jnt_dofadr[j]) for j in hinge_ids]
    return tuple(float(data.M[model.M_rowadr[i] + model.M_rownnz[i] - 1]) for i in dofs)


@dataclass
class Loaded:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_names: tuple[str, ...]
    qpos_adr: tuple[int, ...]
    dof_adr: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    tau_limit: tuple[float, ...]
    joint_inertia: tuple[float, ...]
    report: dict
    control_path: str = mit.INTEGRATED_CONTROL_PATH


@dataclass
class Build:
    mjcf: Path
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    data: mujoco.MjData
    profile: dict
    model_option: dict
    model_joint_values: dict
    replacement: dict
    applied: list[dict]
    overrides_applied: list[dict]
    tau_overrides: list[dict]
    hinge_ids: list[int]
    joint_names: tuple[str, ...]
    qpos_adr: tuple[int, ...]
    dof_adr: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    tau_limit: tuple[float, ...]
    joint_inertia: tuple[float, ...]

    def loaded_fields(self) -> dict:
        return dict(
            spec=self.spec,
            model=self.model,
            data=self.data,
            joint_names=self.joint_names,
            qpos_adr=self.qpos_adr,
            dof_adr=self.dof_adr,
            actuator_ids=self.actuator_ids,
            tau_limit=self.tau_limit,
            joint_inertia=self.joint_inertia,
        )


def build(
    mjcf: Path,
    schema: profiles.Schema,
    *,
    scene: Callable[[mujoco.MjSpec], None] | None,
    overrides: Mapping | None,
    profile_path: Path,
    tau_limits: Mapping | None,
    model_overrides_path: Path,
    description_version: str | None,
) -> Build:
    mjcf = Path(mjcf).resolve()
    spec = mujoco.MjSpec.from_file(str(mjcf))
    model_option = read_option(spec.option)
    replacement = replace_position_actuators(spec)
    if scene is not None:
        scene(spec)
    model_joint_values = {
        joint.name: {"armature": float(joint.armature), "actfrcrange": _floats(joint.actfrcrange)}
        for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE
    }
    profile = profiles.load_profile(profile_path, schema)
    applied = apply_profile(spec, profile, schema, overrides)
    overrides_document = profiles.load_model_overrides(model_overrides_path, description_version)
    overrides_applied = apply_model_overrides(spec, overrides_document)
    tau_overrides = [] if tau_limits is None else [apply_tau_limits(spec, tau_limits)]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hinge_ids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = tuple(model.joint(j).name for j in hinge_ids)
    tau_limit = []
    for j in hinge_ids:
        lo, hi = _floats(model.jnt_actfrcrange[j])
        if not (hi > 0 and lo == -hi):
            raise profiles.ProfileError(
                f"joint {model.joint(j).name!r}: compiled actfrcrange {[lo, hi]} is not symmetric"
            )
        tau_limit.append(hi)
    for entry in tau_overrides:
        for name, joint in entry["joints"].items():
            joint["effective"] = _floats(model.jnt_actfrcrange[model.joint(name).id])
    return Build(
        mjcf=mjcf,
        spec=spec,
        model=model,
        data=data,
        profile=profile,
        model_option=model_option,
        model_joint_values=model_joint_values,
        replacement=replacement,
        applied=applied,
        overrides_applied=overrides_applied,
        tau_overrides=tau_overrides,
        hinge_ids=hinge_ids,
        joint_names=joint_names,
        qpos_adr=tuple(int(model.jnt_qposadr[j]) for j in hinge_ids),
        dof_adr=tuple(int(model.jnt_dofadr[j]) for j in hinge_ids),
        actuator_ids=tuple(model.actuator(f"{name}_actuator").id for name in joint_names),
        tau_limit=tuple(tau_limit),
        joint_inertia=_joint_inertia(model, data, hinge_ids),
    )


def host_arrays(model: mujoco.MjModel) -> dict[str, list]:
    return {field: getattr(model, field).tolist() for field in REPORT_FIELDS}


def report(
    build: Build,
    platform: str,
    arrays: Mapping[str, list],
    readouts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict:
    """The load report. `arrays` holds REPORT_FIELDS of the model the simulation steps (one world); `readouts`
    gives the effective values of profile groups this module does not read from the host model."""
    model = build.model
    params = build.profile["params"]
    for entry in build.replacement["joints"]:
        aid = model.actuator(entry["actuator"]["name"]).id
        entry["actuator"]["forcerange_effective"] = _floats(arrays["actuator_forcerange"][aid])
        entry["actuator"]["forcelimited_effective"] = bool(arrays["actuator_forcelimited"][aid])
        entry["actuator"]["ctrllimited_effective"] = bool(arrays["actuator_ctrllimited"][aid])

    effective_option = read_option(model.opt)
    requested_option = {name: params["option"][name]["value"] for name in OPTION_FIELDS}
    collision = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    compiled_contact = {name: _unique(getattr(model, name).tolist(), collision) for name in CONTACT_FIELDS}
    contact = {
        name: {"values": _unique(arrays[name], collision), "geoms": len(collision)} for name in CONTACT_FIELDS
    }
    for entry in build.applied:
        group, name = entry["parameter"].split(".", 1)
        if group == "option":
            requested_option[name] = entry["value"]
            entry["effective"] = effective_option[name]
        elif group == "contact":
            entry["effective"] = compiled_contact[name]
        elif group == "joint_coulomb_friction":
            entry["effective"] = float(model.dof_frictionloss[model.jnt_dofadr[model.joint(name).id]])
        elif group == "joint_damping":
            entry["effective"] = float(model.dof_damping[model.jnt_dofadr[model.joint(name).id]])
        elif readouts is not None and group in readouts:
            entry["effective"] = readouts[group][name]
        else:
            raise AssertionError(f"no effective-value readout for profile group {group!r}")
    tau_override_joints = {name for override in build.tau_overrides for name in override["joints"]}
    for entry in build.overrides_applied:
        group, name = entry["parameter"].split(".", 1)
        if group == "joint_torque_limit" and name in tau_override_joints:
            entry["superseded"] = True
            continue
        joint_id = model.joint(name).id
        dof = model.jnt_dofadr[joint_id]
        entry["effective"] = (
            float(arrays["dof_armature"][dof]) if group == "joint_armature"
            else float(arrays["jnt_actfrcrange"][joint_id][1])
        )
    option = {
        name: {
            "unit": params["option"][name]["unit"],
            "model_source": build.model_option[name],
            "profile": requested_option[name],
            "effective": effective_option[name],
        }
        for name in OPTION_FIELDS
    }
    option["jacobian"] = {
        "unit": "enum",
        "model_source": build.model_option["jacobian"],
        "profile": None,
        "effective": effective_option["jacobian"],
    }
    replacement_by_joint = {entry["joint"]: entry for entry in build.replacement["joints"]}
    joints = {
        model.joint(j).name: {
            "model_armature": build.model_joint_values[model.joint(j).name]["armature"],
            "armature": arrays["dof_armature"][model.jnt_dofadr[j]],
            "frictionloss": arrays["dof_frictionloss"][model.jnt_dofadr[j]],
            "damping": arrays["dof_damping"][model.jnt_dofadr[j]],
            "model_actfrcrange": build.model_joint_values[model.joint(j).name]["actfrcrange"],
            "actfrcrange": arrays["jnt_actfrcrange"][j],
            "range": arrays["jnt_range"][j],
            "kp": replacement_by_joint[model.joint(j).name]["kp"],
            "kd": replacement_by_joint[model.joint(j).name]["kd"],
            "tau_limit": arrays["jnt_actfrcrange"][j][1],
            "control_path": mit.INTEGRATED_CONTROL_PATH,
        }
        for j in build.hinge_ids
    }
    actuators = {
        model.actuator(a).name: {
            "joint": model.joint(int(model.actuator_trnid[a, 0])).name,
            "ctrlrange": arrays["actuator_ctrlrange"][a],
            "forcerange": arrays["actuator_forcerange"][a],
            "forcelimited": bool(arrays["actuator_forcelimited"][a]),
            "ctrllimited": bool(arrays["actuator_ctrllimited"][a]),
            "gainprm0": arrays["actuator_gainprm"][a][0],
            "biasprm": arrays["actuator_biasprm"][a][:3],
        }
        for a in range(model.nu)
    }
    return {
        "platform": platform,
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "mjcf": str(build.mjcf),
        "profile": {
            "platform": build.profile["platform"],
            "file": build.profile["path"],
            "sha256": build.profile["sha256"],
        },
        "labels": [REPLACEMENT_LABEL, mit.CONTROL_PATH_LABEL]
        + [f"with override: {entry['parameter']}" for entry in (*build.applied, *build.overrides_applied)
           if not entry.get("superseded")]
        + [entry["label"] for entry in build.tau_overrides],
        "overrides": [build.replacement, *build.applied, *build.overrides_applied, *build.tau_overrides],
        "option": option,
        "effective_values": {
            "units": {
                **EFFECTIVE_UNITS,
                **{name: params["contact"][name]["unit"] for name in CONTACT_FIELDS},
            },
            "joints": joints,
            "actuators": actuators,
            "contact": contact,
        },
        "backend_imposed": [],
        "backend_checked": [],
        "import_options": {},
    }


def mit_stability(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> dict:
    control_path = getattr(loaded, "control_path", None) or loaded.report.get(
        "control_path", mit.INTEGRATED_CONTROL_PATH
    )
    integrated = control_path == mit.INTEGRATED_CONTROL_PATH
    if integrated:
        integrator = enum_name(INTEGRATORS, loaded.model.opt.integrator)
        if integrator not in IMPLICIT_INTEGRATORS:
            raise mit.MitStabilityError(
                f"MIT integrated actuator stability check covers only the {' and '.join(IMPLICIT_INTEGRATORS)} "
                f"integrators, got {integrator}: {integrator} integrates the actuator damping explicitly, which the "
                "kp*dt^2/I bound does not cover; set option.integrator to implicitfast or implicit"
            )
    return mit.stability(
        loaded.joint_names,
        loaded.joint_inertia,
        float(loaded.model.opt.timestep),
        kp,
        kd,
        control_path=control_path,
        criterion=MIT_STABILITY_CRITERION if integrated else EXPLICIT_MIT_STABILITY_CRITERION,
    )


def set_host_gains(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> tuple[list[float], list[float]]:
    kp, kd = mit.validate_gains(loaded.joint_names, kp, kd)
    expected = len(loaded.joint_names)
    if len(loaded.actuator_ids) != expected:
        raise ValueError(f"loaded actuator count {len(loaded.actuator_ids)} does not match joints {expected}")
    for actuator_id, gain_kp, gain_kd in zip(loaded.actuator_ids, kp, kd):
        loaded.model.actuator_biasprm[actuator_id, 1] = -gain_kp
        loaded.model.actuator_biasprm[actuator_id, 2] = -gain_kd
    return kp, kd
