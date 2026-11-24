# SPDX-License-Identifier: Apache-2.0
"""MotrixSim platform adapter.

The hand MJCF is built and patched on a `mujoco.MjSpec` exactly like the MuJoCo platform (same replacement of the
model's <position> actuators with integrated affine general actuators, same profile machinery), then the patched
spec is serialized to MJCF and imported into MotrixSim through `motrixsim.msd.from_str`; `msd.World.build()` yields
the `SceneModel` the backend steps. The compiled MuJoCo model of the same spec stays around as the reporting twin:
`_mjcf.report` reads effective values from it, and `audit_backend` cross-checks the quantities MotrixSim exposes
(runtime options, joint armature/frictionloss/range, actuator control limits) against that twin, recording any
backend-imposed deviation.

MIT control follows the integrated actuator path, in the same parameterisation as the MuJoCo platform: each
general actuator keeps gainprm[0] = 1 and biasprm = [0, -kp, -kd] with a forcerange at the joint tau limit, and
`ctrl = kp*q_des + kd*qd_des + tau_ff` plus the actuator's own bias and forcerange realises the MIT law — the
examples' control loops are line-identical to the MuJoCo ones. MotrixSim integrates the actuator damping
implicitly (verified empirically: kp = 60, kd = 1.5 at dt = 0.005 s settles stably, and an isolated kd with
kd*dt/I ~ 250 decays monotonically, where an explicit integrator would diverge), so the `kp*dt^2/I < 4`
criterion applies. `set_mit_gains` rewrites the bias through the per-actuator stiffness/damping overrides, which
write biasprm[1] and biasprm[2] only and leave the unit gain untouched. A native alternative — gainprm[0] = kp
so `ctrl` is the target position, with `qd_des`/`tau_ff` in the bias override — was verified equivalent to
float32 epsilon (1.2e-7 rad over 2 s) and rejected only to keep the MuJoCo-family parameterisation.
"""
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import motrixsim
import motrixsim.msd as msd

from wuji_sim import mit, profiles
from wuji_sim.platforms import _mjcf

PLATFORM = "motrixsim"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "profiles" / "motrixsim.toml"
# MotrixSim has no MuJoCo integrator or line-search option; timestep/solver/iterations map onto
# msd.World.simulate_option. `tolerance` is deliberately NOT configured: MotrixSim's criterion (velocity-space
# residuals, unnormalized) is not equivalent to MuJoCo's (inertia-normalized cost improvement / gradient), no
# honest conversion exists, so the backend default (1e-4) applies — see _sanitize_xml.
OPTION_FIELDS = ("timestep", "solver", "iterations")
# MotrixSim's own default when the MJCF carries no tolerance attribute (measured on a bare world).
BACKEND_DEFAULT_TOLERANCE = 1e-4
SOLVERS = {"PGS": msd.SolverType.PGS, "Newton": msd.SolverType.Newton}
SCHEMA = profiles.Schema(
    platform=PLATFORM,
    groups={
        "option": OPTION_FIELDS,
        "contact": _mjcf.CONTACT_FIELDS,
        "joint_coulomb_friction": None,
        "joint_damping": None,
    },
    rules={
        key: rule
        for key, rule in _mjcf.VALUE_RULES.items()
        if key[0] != "option" or key[1] in OPTION_FIELDS
    },
)
# SceneModel quantities cross-checked against the compiled MuJoCo twin; a relative difference below this is
# float32 representation, not a backend change.
AUDIT_REL_TOL = 1e-5


def load_profile(path: Path = PROFILE_PATH) -> dict:
    return profiles.load_profile(path, SCHEMA)


@dataclass(kw_only=True)
class Loaded(_mjcf.Loaded):
    scene_model: motrixsim.SceneModel
    scene_data: motrixsim.SceneData
    actuators: tuple  # The general actuators driving the hinge joints, in joint_names order.
    gains: tuple[tuple[float, float], ...]  # (kp, kd) per joint, as last installed by set_mit_gains / load.
    _build: object = None  # The _mjcf.Build this Loaded came from; kept for post-load audits in tests.


def _requested_option(build: _mjcf.Build) -> dict:
    params = build.profile["params"]
    option = {name: params["option"][name]["value"] for name in OPTION_FIELDS}
    for entry in build.applied:
        group, name = entry["parameter"].split(".", 1)
        if group == "option":
            option[name] = entry["value"]
    return option


def _audit_differs(requested, effective) -> bool:
    if isinstance(requested, (list, tuple)):
        return len(requested) != len(effective) or any(map(_audit_differs, requested, effective))
    if isinstance(requested, str):
        return requested != effective
    return not math.isclose(float(requested), float(effective), rel_tol=AUDIT_REL_TOL)


def audit_backend(build: _mjcf.Build, scene_model: motrixsim.SceneModel, option: Mapping) -> tuple[list[dict], list[str]]:
    model = build.model
    imposed, checked = [], []

    def note(parameter: str, requested, effective) -> None:
        checked.append(parameter)
        if _audit_differs(requested, effective):
            imposed.append({"parameter": parameter, "requested": requested, "effective": effective,
                            "note": "backend-imposed"})

    runtime_option = {
        "timestep": float(scene_model.options.timestep),
        "iterations": int(scene_model.options.max_iterations),
        "tolerance": float(scene_model.options.solver_tolerance),
    }
    for name in ("timestep", "iterations"):
        note(f"option.{name}", option[name], runtime_option[name])
    # The runtime Options object exposes no solver type; selection is recorded from the import path instead.
    checked.append("option.solver")
    # tolerance is backend-managed (not configured through the profile); record what actually applies.
    note("option.tolerance", BACKEND_DEFAULT_TOLERANCE, runtime_option["tolerance"])

    backend_joints = {name: joint for name, joint in zip(scene_model.joint_names, scene_model.joints)}
    for name in build.joint_names:
        joint = backend_joints[name]
        jid = model.joint(name).id
        dof = int(model.jnt_dofadr[jid])
        note(f"joint.{name}.armature", float(model.dof_armature[dof]), float(joint.armature))
        note(f"joint.{name}.frictionloss", float(model.dof_frictionloss[dof]), float(joint.frictionloss))
        # A hinge joint's runtime range is a (1, 2) array; compare against its single row.
        note(f"joint.{name}.range", model.jnt_range[jid].tolist(), [float(value) for value in joint.range.flat])

    backend_actuators = {name: index for index, name in enumerate(scene_model.actuator_names)}
    for name in build.joint_names:
        actuator_name = f"{name}_actuator"
        aid = model.actuator(actuator_name).id
        # The replaced actuators are ctrllimited=False: MuJoCo then stores [0, 0] while the backend reports
        # (-inf, inf); both mean "unlimited", so normalize the host side before comparing.
        host_range = ([-float("inf"), float("inf")] if not model.actuator_ctrllimited[aid]
                      else model.actuator_ctrlrange[aid].tolist())
        index = backend_actuators[actuator_name]
        note(f"actuator.{actuator_name}.ctrlrange", host_range,
             scene_model.actuator_ctrl_limits[:, index].tolist())
    return imposed, checked


def _sanitize_xml(xml: str) -> str:
    """Drop MuJoCo-only attribute values MotrixSim's MJCF parser rejects outright (it only warns on unknown
    attributes, but a known attribute with an unsupported variant is a hard error). The XML document is
    otherwise passed through unchanged."""
    return xml.replace(' colorspace="auto"', "")


def load(
    mjcf: Path,
    *,
    scene: Callable | None = None,
    overrides: Mapping | None = None,
    profile_path: Path = PROFILE_PATH,
    tau_limits: Mapping | None = None,
    model_overrides_path: Path = profiles.MODEL_OVERRIDES_PATH,
    description_version: str | None = None,
    render_resolution: Sequence[int] | None = None,
    render_fovy: float | None = None,
    world_scene: Callable[[object], None] | None = None,
) -> Loaded:
    """`render_resolution` configures the world's system camera for offscreen image capture before the model is
    built (the msd layer is immutable after build); pass e.g. (1920, 1080) when a RenderApp will capture images.
    `render_fovy` sets the system camera's vertical field of view in degrees before build; only the renderer's
    system camera is captured offscreen, so every view shares this one fovy.
    `world_scene` receives the msd.World after the MJCF import and option setup, before build — the hook for
    authoring additional bodies natively through the msd API (the isaaclab platform's USD authoring is the
    precedent); geoms added here set their own PhysicsMaterial, the profile's contact parameters do not reach
    them through the XML route."""
    build = _mjcf.build(
        mjcf,
        SCHEMA,
        scene=scene,
        overrides=overrides,
        profile_path=profile_path,
        tau_limits=tau_limits,
        model_overrides_path=model_overrides_path,
        description_version=description_version,
    )
    option = _requested_option(build)
    xml = _sanitize_xml(build.spec.to_xml())
    world = msd.from_str(xml, format="mjcf", file_path=str(build.mjcf))
    world.simulate_option.timestep = option["timestep"]
    world.simulate_option.solver_type = SOLVERS[option["solver"]]
    world.simulate_option.constraint_solver_iterations = option["iterations"]
    # The XML keeps the hand model's MuJoCo-tuned tolerance untouched (the bridge does not rewrite the model
    # document), but it is overridden here: the parser would clamp it (1e-10 parses as 1e-5) and the value has
    # no MotrixSim-equivalent meaning, so the backend default is the deliberate choice.
    world.simulate_option.constraint_solver_tolerance = BACKEND_DEFAULT_TOLERANCE
    if render_resolution is not None:
        if len(render_resolution) != 2:
            raise ValueError(f"render_resolution must be (width, height), got {render_resolution!r}")
        world.system_camera.render_target = msd.RenderTarget.Image
        world.system_camera.resolution = [int(render_resolution[0]), int(render_resolution[1])]
    if render_fovy is not None:
        world.system_camera.fovy = float(render_fovy)
    if world_scene is not None:
        world_scene(world)
    scene_model = world.build()
    scene_data = motrixsim.SceneData(scene_model)

    expected_joints = tuple(build.joint_names)
    if tuple(scene_model.joint_names) != expected_joints:
        raise profiles.ProfileError(
            f"MotrixSim model joints {tuple(scene_model.joint_names)} do not match the patched spec "
            f"joints {expected_joints}"
        )
    expected_actuators = tuple(f"{name}_actuator" for name in expected_joints)
    if tuple(scene_model.actuator_names) != expected_actuators:
        raise profiles.ProfileError(
            f"MotrixSim model actuators {tuple(scene_model.actuator_names)} do not match the patched spec "
            f"actuators {expected_actuators}"
        )

    imposed, checked = audit_backend(build, scene_model, option)
    report = _mjcf.report(build, PLATFORM, _mjcf.host_arrays(build.model))
    report["backend_imposed"] = imposed
    report["backend_checked"] = checked
    report["import_options"] = {
        "route": "mujoco.MjSpec -> MJCF string -> motrixsim.msd.from_str -> msd.World.build",
        "solver": option["solver"],
        "motrixsim": metadata.version("motrixsim"),
    }

    actuators = tuple(scene_model.actuators[scene_model.actuator_names.index(name)] for name in expected_actuators)
    gains = tuple(
        (entry["kp"], entry["kd"]) for entry in build.replacement["joints"]
    )
    return Loaded(
        **build.loaded_fields(),
        report=report,
        scene_model=scene_model,
        scene_data=scene_data,
        actuators=actuators,
        gains=gains,
        _build=build,
    )


def mit_stability(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> dict:
    # MotrixSim's integrator is not the MuJoCo implicitfast path the shared check knows, but it integrates the
    # actuator damping implicitly (see the module docstring), so the integrated-actuator bound applies unchanged.
    return mit.stability(
        loaded.joint_names,
        loaded.joint_inertia,
        float(loaded.model.opt.timestep),
        kp,
        kd,
        control_path=mit.INTEGRATED_CONTROL_PATH,
        criterion=_mjcf.MIT_STABILITY_CRITERION,
    )


def set_mit_gains(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> tuple[list[float], list[float]]:
    # Keep the compiled MuJoCo twin in sync, then push the bias into the MotrixSim actuators through the
    # stiffness/damping overrides, which write biasprm[1] / biasprm[2] and leave gainprm[0] = 1 untouched.
    kp, kd = _mjcf.set_host_gains(loaded, kp, kd)
    for actuator, gain_kp, gain_kd in zip(loaded.actuators, kp, kd):
        actuator.set_stiffness_override(loaded.scene_data, gain_kp)
        actuator.set_damping_override(loaded.scene_data, gain_kd)
    loaded.gains = tuple(zip(kp, kd))
    return kp, kd


def apply_mit(loaded: Loaded, q_des: Sequence[float], qd_des: Sequence[float],
              tau_ff: Sequence[float] | None = None) -> None:
    """Write the MIT command for every joint: ctrl = kp*q_des + kd*qd_des + tau_ff, as one batched
    scene_data.actuator_ctrls property assignment (in-place writes into a returned actuator_ctrls array are
    not consumed by step; the property setter is). The actuator's affine bias adds -kp*q - kd*qd and its
    forcerange clips the resulting torque to the joint tau limit — the same parameterisation as the MuJoCo
    platform, so the examples' control loops stay line-identical. Platform modules stay numpy-free, so the
    command is assembled element-wise."""
    gains = loaded.gains
    if len(q_des) != len(gains) or len(qd_des) != len(gains):
        raise ValueError(f"MIT command shapes {len(q_des)}/{len(qd_des)} do not match {len(gains)} joints")
    feedforward = [0.0] * len(gains) if tau_ff is None else list(tau_ff)
    if len(feedforward) != len(gains):
        raise ValueError(f"tau_ff shape {len(feedforward)} does not match {len(gains)} joints")
    # One batched property assignment; in-place writes into a returned scene_data.actuator_ctrls array are NOT
    # consumed by step (verified: the model ignores them while get_ctrl reads them back).
    loaded.scene_data.actuator_ctrls = [
        kp * q_target + kd * qd_target + feed
        for (kp, kd), q_target, qd_target, feed in zip(gains, q_des, qd_des, feedforward)
    ]


def step(loaded: Loaded) -> None:
    motrixsim.step(loaded.scene_model, loaded.scene_data)


def forward(loaded: Loaded) -> None:
    motrixsim.forward_kinematic(loaded.scene_model, loaded.scene_data)


def set_state(loaded: Loaded, qpos: Sequence[float], qvel: Sequence[float] | None = None) -> None:
    """Install a full DoF state; SceneData.reset is the backend's supported entry point for rewriting state.
    Note the backend side effects documented for reset: actuator ctrls return to their model defaults, and
    Data-side override buffers reset (inertias, physics materials, armature/frictionloss/shape overrides) —
    the actuator stiffness/damping overrides that carry the MIT gains are not among them, so
    set_mit_gains-then-set_state keeps the gains."""
    n_pos, n_vel = int(loaded.scene_model.num_dof_pos), int(loaded.scene_model.num_dof_vel)
    position = [float(value) for value in qpos]
    velocity = [0.0] * n_vel if qvel is None else [float(value) for value in qvel]
    if len(position) != n_pos or len(velocity) != n_vel:
        raise ValueError(f"state shapes {len(position)}/{len(velocity)} do not match DoFs {(n_pos, n_vel)}")
    loaded.scene_data.reset(
        loaded.scene_model, dof_pos=position, dof_vel=velocity, forward_kinematic=True
    )


def link_pose(loaded: Loaded, name: str) -> list[float]:
    """World pose of link `name` as [x, y, z, qw, qx, qy, qz], the wxyz order wuji_sim uses throughout. Platform
    modules stay numpy-free, so the pose is returned as a plain list."""
    index = loaded.scene_model.get_link_index(name)
    if index is None:
        raise ValueError(f"the MotrixSim model has no link {name!r}")
    x, y, z, qi, qj, qk, qw = loaded.scene_model.get_link_poses(loaded.scene_data, [index])[0, :7]
    return [float(x), float(y), float(z), float(qw), float(qi), float(qj), float(qk)]
