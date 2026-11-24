# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from wuji_sim import conventions, mit, profiles


PLATFORM = "isaaclab"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "profiles" / "isaaclab.toml"
REPLACEMENT_LABEL = "with override: USD joint drives replaced by runtime MIT gains"
CONTACT_ATTRIBUTES = {
    "default_material_static_friction": "physics:staticFriction",
    "default_material_dynamic_friction": "physics:dynamicFriction",
}
SCHEMA = profiles.Schema(
    platform=PLATFORM,
    groups={
        "option": ("timestep",),
        "contact": tuple(CONTACT_ATTRIBUTES),
        "joint_friction": None,
        "joint_velocity_limit": None,
    },
    rules={
        ("option", "timestep"): ("a positive number", lambda v: profiles.is_number(v) and v > 0),
        **{
            ("contact", name): ("a non-negative number", lambda v: profiles.is_number(v) and v >= 0)
            for name in CONTACT_ATTRIBUTES
        },
        ("joint_friction", None): ("a non-negative number", lambda v: profiles.is_number(v) and v >= 0),
        ("joint_velocity_limit", None): ("a positive number", lambda v: profiles.is_number(v) and v > 0),
    },
)
MIT_STABILITY_CRITERION = (
    "kp_j*dt^2/I_j < 4 for every integrated actuator joint; I_j is the diagonal of the generalized mass "
    "matrix (armature included) at the initial pose qpos0"
)
HAND_ROOT_QUAT = [1.0, 0.0, 0.0, 0.0]


def load_profile(path: Path = PROFILE_PATH) -> dict:
    return profiles.load_profile(path, SCHEMA)


@contextlib.contextmanager
def launch(*, headless: bool = True, device: str = "cuda:0", enable_cameras: bool = False):
    from isaaclab.app import launch_simulation

    # cfg=None gives launch_simulation nothing to detect Kit from, so require_kit=True is what starts Kit.
    args = {"headless": headless, "device": device, "enable_cameras": enable_cameras, "require_kit": True}
    if not headless:
        args["visualizer"] = ["kit"]
    with contextlib.ExitStack() as stack:
        # Only the window open is wrapped: display failures keep the one-line error, headless ones raise as-is.
        if not headless:
            try:
                physics_cfg = stack.enter_context(launch_simulation(None, args))
            except Exception as exc:
                raise conventions.ViewerError(f"could not open the Isaac Sim window: {exc}; {conventions.headless_rerun_hint()}") from exc
        else:
            physics_cfg = stack.enter_context(launch_simulation(None, args))
        yield physics_cfg


def _joint_names(hand: str) -> tuple[str, ...]:
    if hand not in conventions.HAND_PREFIXES:
        raise ValueError(f"hand must be one of {tuple(conventions.HAND_PREFIXES)}, got {hand!r}")
    return tuple(conventions.HAND_PREFIXES[hand] + suffix for suffix in conventions.JOINT_SUFFIXES)


def _usd_value(attribute):
    if attribute is None or not attribute.IsValid():
        return None
    return attribute.Get()


def _usd_per_radian(value):
    return None if value is None else float(value) * 180.0 / math.pi


def _read_usd_joints(path: Path) -> dict[str, dict]:
    from pxr import Usd, UsdPhysics

    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        raise profiles.ProfileError(f"cannot open USD asset {path}: {exc}") from exc
    if stage is None:
        raise profiles.ProfileError(f"cannot open USD asset {path}")
    found = {}
    for prim in stage.Traverse():
        if "Joint" not in prim.GetTypeName():
            continue
        name = str(prim.GetName())
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive_type = _usd_value(drive.GetTypeAttr()) if drive else None
        found[name] = {
            "type": None if drive_type is None else str(drive_type),
            "stiffness": _usd_per_radian(_usd_value(drive.GetStiffnessAttr())) if drive else None,
            "damping": _usd_per_radian(_usd_value(drive.GetDampingAttr())) if drive else None,
            "max_force": _usd_value(drive.GetMaxForceAttr()) if drive else None,
            "target_position": (
                None
                if not drive
                else (
                    None
                    if _usd_value(drive.GetTargetPositionAttr()) is None
                    else float(_usd_value(drive.GetTargetPositionAttr())) * math.pi / 180.0
                )
            ),
            "armature": _usd_value(prim.GetAttribute("physxJoint:armature")),
            "friction": _usd_value(prim.GetAttribute("physxJoint:jointFriction")),
            "max_velocity": _usd_value(prim.GetAttribute("physxJoint:maxJointVelocity")),
        }
    return found



def _read_runtime_self_collision(configured: bool) -> dict:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import PhysxSchema

    stage = get_current_stage()
    candidates = []
    for prim in stage.Traverse():
        if not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            continue
        attribute = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if not attribute.IsValid() or not attribute.HasAuthoredValue():
            continue
        value = attribute.Get()
        candidates.append({"prim_path": str(prim.GetPath()), "value": bool(value)})
    enabled = [candidate for candidate in candidates if candidate["value"]]
    if not enabled:
        raise profiles.ProfileError(
            "spawned articulation has no enabled physxArticulation:enabledSelfCollisions attribute; "
            f"candidates={candidates!r}"
        )
    selected = enabled[0]
    return {
        "configured": bool(configured),
        "attribute": "physxArticulation:enabledSelfCollisions",
        "prim_path": selected["prim_path"],
        "value": selected["value"],
        "candidates": candidates,
    }


def _read_default_material(physics_prim_path: str) -> dict:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdShade

    stage = get_current_stage()
    path = f"{physics_prim_path}/defaultMaterial"
    scene_prim = stage.GetPrimAtPath(physics_prim_path)
    bound = (
        [str(target) for target in UsdShade.MaterialBindingAPI(scene_prim).GetDirectBindingRel("physics").GetTargets()]
        if scene_prim.IsValid() else []
    )
    if bound != [path]:
        raise profiles.ProfileError(f"physics scene {physics_prim_path} binds physics material {bound!r}, not {path}")
    prim = stage.GetPrimAtPath(path)
    values = {name: _usd_value(prim.GetAttribute(attribute)) for name, attribute in CONTACT_ATTRIBUTES.items()}
    unset = sorted(name for name, value in values.items() if value is None)
    if unset:
        raise profiles.ProfileError(f"default physics material {path} has no value for {unset}")
    return {"default_material_prim": path, **{name: float(value) for name, value in values.items()}}


def _tensor_values(tensor, indices: Sequence[int]) -> tuple[float, ...]:
    values = tensor[0, list(indices)]
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return tuple(float(value) for value in values)


def _tensor_matrix_values(tensor, indices: Sequence[int]) -> list[list[float]]:
    values = tensor[0, list(indices)]
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [[float(value) for value in row] for row in values]


def _read_option(sim) -> dict:
    cfg = sim.cfg
    solver = {0: "PGS", 1: "TGS"}.get(int(cfg.physics.solver_type), str(cfg.physics.solver_type))
    return {
        "timestep": float(cfg.dt),
        "solver_type": solver,
        "position_iterations": {
            "min": int(cfg.physics.min_position_iteration_count),
            "max": int(cfg.physics.max_position_iteration_count),
        },
        "velocity_iterations": {
            "min": int(cfg.physics.min_velocity_iteration_count),
            "max": int(cfg.physics.max_velocity_iteration_count),
        },
        "gravity": [float(value) for value in cfg.gravity],
        "device": str(cfg.device),
        "physics_backend": sim.physics_manager.__name__,
    }


@dataclass
class Loaded:
    sim: object
    articulation: object
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    tau_limit: tuple[float, ...]
    joint_inertia: tuple[float, ...]
    dt: float
    device: str
    report: dict
    control_path: str = mit.INTEGRATED_CONTROL_PATH


def load(
    usd: Path,
    *,
    hand: str,
    floor_z: float,
    scene,
    overrides: Mapping | None = None,
    profile_path: Path = PROFILE_PATH,
    tau_limits: Mapping | None = None,
    model_overrides_path: Path = profiles.MODEL_OVERRIDES_PATH,
    description_version: str | None = None,
    device: str = "cuda:0",
    num_envs: int = 1,
    env_spacing: float = 2.0,
) -> Loaded:
    usd = Path(usd).resolve()
    if not usd.is_file():
        raise profiles.ProfileError(f"USD asset {usd} does not exist")
    if not profiles.is_count(num_envs):
        raise profiles.ProfileError(f"num_envs must be a positive int, got {num_envs!r}")
    if not (profiles.is_number(env_spacing) and env_spacing > 0):
        raise profiles.ProfileError(f"env_spacing must be a finite positive number of m, got {env_spacing!r}")
    if not profiles.is_number(floor_z):
        raise profiles.ProfileError(f"floor_z must be a finite number of m, got {floor_z!r}")
    if scene is not None and not (callable(getattr(scene, "assets", None)) and callable(getattr(scene, "finish", None))):
        raise profiles.ProfileError(f"scene must provide assets() and finish(num_envs), got {scene!r}")
    expected_joints = _joint_names(hand)
    profile = load_profile(profile_path)
    params = profile["params"]
    profile_entries = profiles.check_overrides(SCHEMA, profile["overrides"], params, f"profile {profile['path']}")
    code_entries = profiles.check_overrides(SCHEMA, overrides or {}, params, "code overrides")
    requested = {
        group: {name: float(entry["value"]) if isinstance(entry["value"], (int, float)) else entry["value"]
                for name, entry in table.items()}
        for group, table in params.items()
    }
    applied = []
    for origin, entries in (("profile-file", profile_entries), ("code", code_entries)):
        for group, name, entry in entries:
            if SCHEMA.groups[group] is None and name not in expected_joints:
                raise profiles.ProfileError(f"overrides.{group}.{name}: {name!r} is not a joint of {hand!r} hand ({origin} override)")
            requested[group][name] = entry["value"]
            baseline = params[group][name]
            applied.append({
                "kind": "profile_override",
                "parameter": f"{group}.{name}",
                "baseline": baseline["value"],
                "value": entry["value"],
                "unit": baseline["unit"],
                "reason": entry["reason"],
                "origin": origin,
            })
    both = sorted({(g, n) for g, n, _ in profile_entries} & {(g, n) for g, n, _ in code_entries})
    if both:
        targets = ", ".join(f"{group}.{name}" for group, name in both)
        raise profiles.ProfileError(f"{targets}: overridden both in profile {profile['path']} and in code overrides")
    for group in ("joint_friction", "joint_velocity_limit"):
        missing = [name for name in expected_joints if name not in requested[group]]
        if missing:
            raise profiles.ProfileError(f"{group}: no entry for model joint(s) {missing} (profile {profile['path']})")

    overrides_document = profiles.load_model_overrides(model_overrides_path, description_version)
    overrides_applied = []
    override_values = {"joint_armature": {}, "joint_torque_limit": {}}
    for group, table in overrides_document["groups"].items():
        for name, entry in table["values"].items():
            if name not in expected_joints:
                if name[:2] == ("r_" if hand == "left" else "l_"):
                    continue
                raise profiles.ProfileError(f"{group}.{name}: {name!r} is not a joint of {hand!r} hand")
            override_values[group][name] = float(entry["value"])
        missing = [name for name in expected_joints if name not in override_values[group]]
        if missing:
            raise profiles.ProfileError(
                f"{group}: no entry for model joint(s) {missing} (model override file {Path(model_overrides_path).resolve()})"
            )

    if tau_limits is None:
        tau_values = dict(override_values["joint_torque_limit"])
        tau_override = None
    else:
        if not isinstance(tau_limits, Mapping) or "values" not in tau_limits:
            raise profiles.ProfileError("tau_limits must be the mapping returned by load_tau_limits")
        values = tau_limits["values"]
        missing, unknown = sorted(set(expected_joints) - set(values)), sorted(set(values) - set(expected_joints))
        if missing or unknown:
            where = tau_limits.get("file", "tau limit file")
            raise profiles.ProfileError(
                f"tau_limit must name exactly the hand's joints: missing {missing}, unknown {unknown} ({where})"
            )
        tau_values = {name: float(values[name]) for name in expected_joints}
        tau_override = tau_limits

    usd_joints = _read_usd_joints(usd)
    for group, values in override_values.items():
        for name, value in values.items():
            original = usd_joints.get(name, {})
            baseline = original.get("armature" if group == "joint_armature" else "max_force")
            overrides_applied.append({
                "kind": "model_override",
                "parameter": f"{group}.{name}",
                "baseline": 0.0 if baseline is None else float(baseline),
                "model_value": value,
                "value": value,
                "unit": profiles.MODEL_OVERRIDE_UNITS[group],
                "source": overrides_document["groups"][group]["source"],
                "origin": "model_overrides.toml",
            })

    from isaaclab import sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg
    sim_utils.create_new_stage()
    sim_cfg = sim_utils.SimulationCfg(
        dt=float(requested["option"]["timestep"]),
        gravity=tuple(conventions.GRAVITY),
        device=device,
        physics=PhysxCfg(),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=float(requested["contact"]["default_material_static_friction"]),
            dynamic_friction=float(requested["contact"]["default_material_dynamic_friction"]),
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    # One prim is both the PhysX ground and the visible checker floor.
    ground_cfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0), visible=True),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, float(floor_z))),
    )
    look = dict(scene.assets()) if scene is not None else {}
    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=list(expected_joints),
        stiffness=0.0,
        damping=0.0,
        joint_effort_limit={name: tau_values[name] for name in expected_joints},
        joint_velocity_limit={name: float(requested["joint_velocity_limit"][name]) for name in expected_joints},
        armature={name: override_values["joint_armature"][name] for name in expected_joints},
        friction={name: float(requested["joint_friction"][name]) for name in expected_joints},
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    self_collision_enabled = True

    articulation_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            fix_root_link=True,
            # "/.*" matches descendants: the USD puts the joints and articulation root below the spawn prim.
            joint_drive_props={"/.*": [sim_utils.UsdPhysicsDriveCfg(drive_type="force")]},
            articulation_props={"/.*": [PhysxArticulationCfg(enabled_self_collisions=self_collision_enabled)]},
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=tuple(HAND_ROOT_QUAT)),
        actuators={"mit": actuator_cfg},
    )

    scene_cfg = configclass(
        type("SceneCfg", (InteractiveSceneCfg,), {"ground": ground_cfg, **look, "hand": articulation_cfg})
    )
    interactive_scene = InteractiveScene(
        scene_cfg(num_envs=num_envs, env_spacing=float(env_spacing), replicate_physics=False)
    )
    if scene is not None:
        scene.finish(num_envs)
    sim.reset()
    interactive_scene.reset()
    articulation = interactive_scene["hand"]
    self_collision_readback = _read_runtime_self_collision(self_collision_enabled)
    contact_readback = _read_default_material(sim.cfg.physics_prim_path)

    joint_ids, resolved_names = articulation.find_joints(list(expected_joints), preserve_order=True)
    if tuple(resolved_names) != expected_joints:
        raise profiles.ProfileError(f"USD joint names do not match expected order: {resolved_names!r} != {expected_joints!r}")
    joint_ids = tuple(int(index) for index in joint_ids)
    data = articulation.data
    tau_readback = _tensor_values(data.joint_effort_limits.torch, joint_ids)
    armature_readback = _tensor_values(data.joint_armature.torch, joint_ids)
    friction_readback = _tensor_values(data.joint_friction_coeff.torch, joint_ids)
    dynamic_friction_readback = _tensor_values(data.joint_dynamic_friction_coeff.torch, joint_ids)
    viscous_friction_readback = _tensor_values(data.joint_viscous_friction_coeff.torch, joint_ids)
    velocity_readback = _tensor_values(data.joint_vel_limits.torch, joint_ids)
    stiffness_readback = _tensor_values(data.joint_stiffness.torch, joint_ids)
    damping_readback = _tensor_values(data.joint_damping.torch, joint_ids)
    position_limits = _tensor_matrix_values(data.joint_pos_limits.torch, joint_ids)
    mass_matrix = data.mass_matrix.torch
    mass_diagonal = tuple(float(mass_matrix[0, index, index].detach().cpu().item()) for index in joint_ids)
    joint_inertia = tuple(
        diagonal + armature for diagonal, armature in zip(mass_diagonal, armature_readback)
    )
    effective_joints = {}
    for index, name in enumerate(expected_joints):
        original = usd_joints.get(name, {})
        effective_joints[name] = {
            "range": position_limits[index],
            "armature": armature_readback[index],
            "inertia_matrix_diagonal": mass_diagonal[index],
            "joint_inertia": joint_inertia[index],
            "friction": friction_readback[index],
            "dynamic_friction": dynamic_friction_readback[index],
            "viscous_friction": viscous_friction_readback[index],
            "stiffness": stiffness_readback[index],
            "damping": damping_readback[index],
            "max_force": tau_readback[index],
            "max_velocity": velocity_readback[index],
            "tau_limit": tau_readback[index],
            "usd_drive": original,
            "control_path": mit.INTEGRATED_CONTROL_PATH,
        }
    replacement = {
        "kind": "actuator_replacement",
        "label": REPLACEMENT_LABEL,
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "reason": "ImplicitActuator gains are set at runtime so USD-authored drives do not control the hand.",
        "joints": [
            {
                "joint": name,
                "usd_drive": usd_joints.get(name, {}),
                "replacement": {
                    "drive_type": "force",
                    "kp": stiffness_readback[index],
                    "kd": damping_readback[index],
                    "tau_limit": tau_readback[index],
                },
            }
            for index, name in enumerate(expected_joints)
        ],
    }
    if tau_override is not None:
        tau_override_entry = {
            "kind": "tau_limit_override",
            "parameter": "tau_limit",
            "label": f"with override: tau_limit ({tau_override['source']})",
            "control_path": mit.INTEGRATED_CONTROL_PATH,
            "file": tau_override["file"],
            "sha256": tau_override["sha256"],
            "source": tau_override["source"],
            "unit": tau_override["unit"],
            "joints": {
                name: {
                    "baseline": [-override_values["joint_torque_limit"][name], override_values["joint_torque_limit"][name]],
                    "value": [-tau_values[name], tau_values[name]],
                    "effective": [-tau_readback[index], tau_readback[index]],
                }
                for index, name in enumerate(expected_joints)
            },
        }
    else:
        tau_override_entry = None
    option_readback = _read_option(sim)
    option = {
        "timestep": {
            "unit": params["option"]["timestep"]["unit"],
            "profile": float(requested["option"]["timestep"]),
            "effective": option_readback["timestep"],
        },
        "solver_type": {"unit": "enum", "profile": None, "effective": option_readback["solver_type"]},
        "position_iterations": {"unit": "count", "profile": None, "effective": option_readback["position_iterations"]},
        "velocity_iterations": {"unit": "count", "profile": None, "effective": option_readback["velocity_iterations"]},
        "gravity": {"unit": "m/s^2", "profile": list(conventions.GRAVITY), "effective": option_readback["gravity"]},
        "device": {"unit": "device", "profile": device, "effective": option_readback["device"]},
        "physics_backend": {"unit": "class", "profile": "PhysxCfg", "effective": option_readback["physics_backend"]},
    }
    backend_checked = [
        "SimulationCfg.dt read back from the running simulation",
        "ImplicitActuator joint stiffness, damping, armature, friction, effort and velocity limits read back",
        "generalized mass matrix diagonal read at qpos0",
        "PhysX generalized mass matrix excludes joint armature; joint_inertia adds the articulation data.joint_armature readback",
        "articulation root physxArticulation:enabledSelfCollisions read back after spawn",
        "scene default physics material static and dynamic friction read back from the stage after spawn",
    ]

    report = {
        "platform": PLATFORM,
        "control_path": mit.INTEGRATED_CONTROL_PATH,
        "usd": str(usd),
        "profile": {"platform": profile["platform"], "file": profile["path"], "sha256": profile["sha256"]},
        "labels": [REPLACEMENT_LABEL, mit.CONTROL_PATH_LABEL]
        + [f"with override: {entry['parameter']}" for entry in (*applied, *overrides_applied) if not entry.get("superseded")]
        + ([] if tau_override_entry is None else [tau_override_entry["label"]]),
        "overrides": [replacement, *applied, *overrides_applied]
        + ([] if tau_override_entry is None else [tau_override_entry]),
        "option": option,
        "self_collision_readback": self_collision_readback,
        "effective_values": {
            "units": {
                "range": "rad",
                "stiffness": "N*m/rad",
                "damping": "N*m*s/rad",
                "armature": "kg*m^2",
                "friction": "N*m",
                "dynamic_friction": "N*m",
                "viscous_friction": "N*m*s/rad",
                "max_force": "N*m",
                "max_velocity": "rad/s",
                **{name: params["contact"][name]["unit"] for name in CONTACT_ATTRIBUTES},
            },
            "joints": effective_joints,
            "contact": contact_readback,
        },
        "backend_imposed": [
            "PhysX maxForce only clamps the implicit PD force; tau_ff must be zero",
            "PhysX implicit joint-drive kp is solved implicitly; MuJoCo implicitfast only treats the velocity term implicitly",
        ],
        "backend_checked": backend_checked,
        "import_options": {
            "usd": str(usd),
            "fix_root_link": True,
            "self_collision": True,
            "drive_type": "force",
        },
    }
    return Loaded(
        sim=sim,
        articulation=articulation,
        joint_names=expected_joints,
        joint_ids=joint_ids,
        tau_limit=tau_readback,
        joint_inertia=joint_inertia,
        dt=float(sim.cfg.dt),
        device=str(device),
        report=report,
    )


def mit_stability(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> dict:
    return mit.stability(
        loaded.joint_names,
        loaded.joint_inertia,
        float(loaded.dt),
        kp,
        kd,
        control_path=mit.INTEGRATED_CONTROL_PATH,
        criterion=MIT_STABILITY_CRITERION,
    )


def set_mit_gains(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> None:
    kp, kd = mit.validate_gains(loaded.joint_names, kp, kd)
    import torch

    num_envs = loaded.articulation.num_instances
    kp_tensor = torch.as_tensor(kp, dtype=torch.float32, device=loaded.device).reshape(1, -1).repeat(num_envs, 1)
    kd_tensor = torch.as_tensor(kd, dtype=torch.float32, device=loaded.device).reshape(1, -1).repeat(num_envs, 1)
    loaded.articulation.write_joint_stiffness_to_sim_index(stiffness=kp_tensor, joint_ids=list(loaded.joint_ids))
    loaded.articulation.write_joint_damping_to_sim_index(damping=kd_tensor, joint_ids=list(loaded.joint_ids))
    for index, name in enumerate(loaded.joint_names):
        loaded.report["effective_values"]["joints"][name]["stiffness"] = kp[index]
        loaded.report["effective_values"]["joints"][name]["damping"] = kd[index]


def _as_tensor(values, *, device):
    import torch

    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor([float(value) for value in values], dtype=torch.float32, device=device)


def set_command(loaded: Loaded, q_des, qd_des, tau_ff) -> None:
    expected = len(loaded.joint_names)
    q_des = _as_tensor(q_des, device=loaded.device)
    qd_des = _as_tensor(qd_des, device=loaded.device)
    tau_values = [float(value) for value in tau_ff]
    if q_des.numel() != expected or qd_des.numel() != expected or len(tau_values) != expected:
        raise ValueError(f"command values must have one value per joint {loaded.joint_names}")
    if any(value != 0.0 for value in tau_values):
        raise ValueError("tau_ff must be zero for the Isaac Lab implicit actuator path")
    num_envs = loaded.articulation.num_instances
    joint_ids = list(loaded.joint_ids)
    target_command = loaded.articulation.actuators.target_command
    target_command.set_position_index(value=q_des.reshape(1, expected).repeat(num_envs, 1), joint_ids=joint_ids)
    target_command.set_velocity_index(value=qd_des.reshape(1, expected).repeat(num_envs, 1), joint_ids=joint_ids)
    target_command.set_effort_index(
        value=_as_tensor([0.0] * expected, device=loaded.device).reshape(1, expected).repeat(num_envs, 1),
        joint_ids=joint_ids,
    )


def step(loaded: Loaded) -> None:
    loaded.articulation.write_data_to_sim()
    loaded.sim.step(render=False)
    loaded.articulation.update(loaded.dt)


def joint_state(loaded: Loaded) -> tuple[list[float], list[float]]:
    return (
        list(_tensor_values(loaded.articulation.data.joint_pos.torch, loaded.joint_ids)),
        list(_tensor_values(loaded.articulation.data.joint_vel.torch, loaded.joint_ids)),
    )



def set_joint_state(loaded: Loaded, q, qd) -> None:
    expected = len(loaded.joint_names)
    q_tensor = _as_tensor(q, device=loaded.device)
    qd_tensor = _as_tensor(qd, device=loaded.device)
    if q_tensor.numel() != expected or qd_tensor.numel() != expected:
        raise ValueError(f"joint state values must have one value per joint {loaded.joint_names}")
    num_envs = loaded.articulation.num_instances
    loaded.articulation.write_joint_state_to_sim_index(
        position=q_tensor.reshape(1, expected).repeat(num_envs, 1),
        velocity=qd_tensor.reshape(1, expected).repeat(num_envs, 1),
        joint_ids=list(loaded.joint_ids),
    )
    loaded.sim.forward()


def applied_effort(loaded: Loaded) -> list[float]:
    return list(_tensor_values(loaded.articulation.actuators.applied_effort.torch, loaded.joint_ids))
