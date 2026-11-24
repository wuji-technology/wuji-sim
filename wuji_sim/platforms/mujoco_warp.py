# SPDX-License-Identifier: Apache-2.0
import functools
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import warp as wp

from wuji_sim import profiles
from wuji_sim.platforms import _mjcf


PLATFORM = "mujoco_warp"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "profiles" / "mujoco_warp.toml"
CAPACITY_FIELDS = ("nconmax", "njmax")
AUDITED_MODEL_FIELDS = (
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
)
# The device model stores float32; a relative difference below this is representation, not a backend change.
AUDIT_REL_TOL = 1e-6
MIT_FIELDS = ("q_des", "qd_des", "tau_ff")
SCHEMA = profiles.Schema(
    platform=PLATFORM,
    groups={**_mjcf.PROFILE_GROUPS, "capacity": CAPACITY_FIELDS},
    rules={
        **_mjcf.VALUE_RULES,
        ("capacity", "nconmax"): ("a positive integer", profiles.is_count),
        ("capacity", "njmax"): ("a positive integer", profiles.is_count),
    },
)


def load_profile(path: Path = PROFILE_PATH) -> dict:
    return profiles.load_profile(path, SCHEMA)


def _world0(array: wp.array) -> list:
    values = array.numpy()
    return values[0].tolist() if values.ndim > 1 else values.tolist()


def _backend_option(m: mjw.Model) -> dict:
    return {
        "timestep": float(m.opt.timestep.numpy()[0]),
        "integrator": _mjcf.enum_name(_mjcf.INTEGRATORS, m.opt.integrator),
        "solver": _mjcf.enum_name(_mjcf.SOLVERS, m.opt.solver),
        "iterations": int(m.opt.iterations),
        "tolerance": float(m.opt.tolerance.numpy()[0]),
        "ls_iterations": int(m.opt.ls_iterations),
        "jacobian": "sparse" if m.is_sparse else "dense",
    }


def _differs(requested, effective) -> bool:
    if isinstance(requested, list):
        return len(requested) != len(effective) or any(map(_differs, requested, effective))
    if isinstance(requested, str):
        return requested != effective
    return not math.isclose(requested, effective, rel_tol=AUDIT_REL_TOL)


def audit_backend(mjm: mujoco.MjModel, m: mjw.Model) -> tuple[list[dict], list[str]]:
    requested = _mjcf.read_option(mjm.opt)
    # MJWarp resolves jacobian "auto" with its own size rule (nv > 32; C MuJoCo nv >= 60): compare resolved layouts.
    requested["jacobian"] = "sparse" if mujoco.mj_isSparse(mjm) else "dense"
    effective = _backend_option(m)
    imposed, checked = [], []
    for name in (*_mjcf.OPTION_FIELDS, "jacobian"):
        checked.append(f"option.{name}")
        if _differs(requested[name], effective[name]):
            imposed.append(
                {
                    "parameter": f"option.{name}",
                    "requested": requested[name],
                    "effective": effective[name],
                    "note": "backend-imposed",
                }
            )
    for field in AUDITED_MODEL_FIELDS:
        checked.append(field)
        values, backend = getattr(mjm, field).tolist(), _world0(getattr(m, field))
        if len(values) != len(backend):
            imposed.append({"parameter": field, "requested": values, "effective": backend, "note": "backend-imposed"})
            continue
        for index, (value, got) in enumerate(zip(values, backend)):
            if _differs(value, got):
                imposed.append(
                    {"parameter": f"{field}[{index}]", "requested": value, "effective": got, "note": "backend-imposed"}
                )
    return imposed, checked


@dataclass
class MitIndex:
    qpos_adr: wp.array
    dof_adr: wp.array
    act_id: wp.array
    tau_limit: wp.array


@dataclass(kw_only=True)
class Loaded(_mjcf.Loaded):
    m: mjw.Model
    d: mjw.Data
    nworld: int
    mit_index: MitIndex


def load(
    mjcf: Path,
    *,
    nworld: int,
    scene: Callable[[mujoco.MjSpec], None] | None = None,
    overrides: Mapping | None = None,
    profile_path: Path = PROFILE_PATH,
    model_overrides_path: Path = profiles.MODEL_OVERRIDES_PATH,
    description_version: str | None = None,
    tau_limits: Mapping | None = None,
) -> Loaded:
    if isinstance(nworld, bool) or not isinstance(nworld, int) or nworld < 1:
        raise ValueError(f"nworld must be a positive integer, got {nworld!r}")
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
    model = build.model
    params = build.profile["params"]
    capacity = {name: params["capacity"][name]["value"] for name in CAPACITY_FIELDS}
    for entry in build.applied:
        group, name = entry["parameter"].split(".", 1)
        if group == "capacity":
            capacity[name] = entry["value"]
    m = mjw.put_model(model)
    d = mjw.put_data(model, build.data, nworld=nworld, nconmax=capacity["nconmax"], njmax=capacity["njmax"])
    imposed, checked = audit_backend(model, m)

    arrays = {field: _world0(getattr(m, field)) for field in _mjcf.REPORT_FIELDS}
    report = _mjcf.report(
        build, PLATFORM, arrays, readouts={"capacity": {"nconmax": d.naconmax // nworld, "njmax": d.njmax}}
    )
    backend_option = _backend_option(m)
    for entry in build.applied:
        group, name = entry["parameter"].split(".", 1)
        if group == "option":
            entry["backend"] = backend_option[name]
        elif group == "contact":
            entry["backend"] = [list(value) for value in report["effective_values"]["contact"][name]["values"]]
        elif group == "joint_coulomb_friction":
            entry["backend"] = arrays["dof_frictionloss"][model.jnt_dofadr[model.joint(name).id]]
        elif group == "joint_damping":
            entry["backend"] = arrays["dof_damping"][model.jnt_dofadr[model.joint(name).id]]
    for name, entry in report["option"].items():
        entry["backend"] = backend_option[name]
    report["backend_imposed"] = imposed
    report["backend_checked"] = checked
    report["import_options"] = {
        "nworld": d.nworld,
        "nconmax": capacity["nconmax"],
        "njmax": d.njmax,
        "naconmax": d.naconmax,
        "broadphase": mjw.BroadphaseType(m.opt.broadphase).name,
        "is_sparse": bool(m.is_sparse),
        "precision": d.qpos.dtype.__name__,
        "device": str(d.qpos.device),
    }
    device = d.ctrl.device
    return Loaded(
        **build.loaded_fields(),
        report=report,
        m=m,
        d=d,
        nworld=nworld,
        mit_index=MitIndex(
            qpos_adr=wp.array(list(build.qpos_adr), dtype=int, device=device),
            dof_adr=wp.array(list(build.dof_adr), dtype=int, device=device),
            act_id=wp.array(list(build.actuator_ids), dtype=int, device=device),
            tau_limit=wp.array(list(build.tau_limit), dtype=float, device=device),
        ),
    )


def mit_stability(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> dict:
    return _mjcf.mit_stability(loaded, kp, kd)


def set_mit_gains(loaded: Loaded, kp: Sequence[float], kd: Sequence[float]) -> None:
    # The host model gets the same gains, so CPU MuJoCo replays on loaded.model use the Warp worlds' controller.
    kp, kd = _mjcf.set_host_gains(loaded, kp, kd)
    biasprm = loaded.m.actuator_biasprm.numpy()
    for world in range(len(biasprm)):
        for actuator_id, gain_kp, gain_kd in zip(loaded.actuator_ids, kp, kd):
            biasprm[world, actuator_id, 1] = -gain_kp
            biasprm[world, actuator_id, 2] = -gain_kd
    wp.copy(
        loaded.m.actuator_biasprm,
        wp.array(biasprm, dtype=loaded.m.actuator_biasprm.dtype, device=loaded.m.actuator_biasprm.device),
    )


def _counters(d: mjw.Data) -> tuple[dict, list[int], list[int]]:
    nefc = d.nefc.numpy().tolist()
    overflow = d.overflow.numpy().tolist()
    usage = {
        "ncollision": int(d.ncollision.numpy()[0]),
        "nacon": int(d.nacon.numpy()[0]),
        "naconmax": d.naconmax,
        "nefc_max": max(nefc),
        "njmax": d.njmax,
        "overflow": functools.reduce(operator.or_, overflow),
    }
    return usage, nefc, overflow


def capacity_usage(loaded: Loaded) -> dict:
    return _counters(loaded.d)[0]


def check_capacity(loaded: Loaded, peak: dict) -> None:
    usage, nefc, overflow = _counters(loaded.d)
    for key, value in usage.items():
        peak[key] = peak.get(key, 0) | value if key == "overflow" else max(peak.get(key, value), value)
    # MJWarp only warns on overflow and drops contacts (or crashes later), so stop at the capacity instead.
    full = [
        f"{key}={usage[key]} >= naconmax={usage['naconmax']} (all worlds)"
        for key in ("ncollision", "nacon")
        if usage[key] >= usage["naconmax"]
    ]
    full += [f"nefc={n} >= njmax={usage['njmax']} in world {w}" for w, n in enumerate(nefc) if n >= usage["njmax"]]
    # d.overflow bits are sticky per world; some, like EPA_HORIZON, drop contacts without moving any counter above.
    full += [f"overflow bitmap {bits} in world {w}" for w, bits in enumerate(overflow) if bits]
    if full:
        raise RuntimeError(
            f"MuJoCo Warp capacity reached, contacts or constraint rows would be dropped: {'; '.join(full)}; "
            f"raise capacity.nconmax / capacity.njmax in {loaded.report['profile']['file']}"
        )


def host_data(loaded: Loaded, world_id: int = 0) -> mujoco.MjData:
    if not 0 <= world_id < loaded.nworld:
        raise ValueError(f"world_id {world_id} is outside the batch of {loaded.nworld} world(s)")
    mjw.get_data_into(loaded.data, loaded.model, loaded.d, world_id)
    # get_data_into leaves poses one step behind qpos; mj_forward would also overwrite qacc and other Warp results.
    mujoco.mj_kinematics(loaded.model, loaded.data)
    mujoco.mj_comPos(loaded.model, loaded.data)
    mujoco.mj_camlight(loaded.model, loaded.data)
    return loaded.data


_MJV_GEOM_FIELDS = tuple(name for name in dir(mujoco.MjvGeom) if not name.startswith("_"))


def _copy_mjv_geom(source: mujoco.MjvGeom, target: mujoco.MjvGeom) -> None:
    for name in _MJV_GEOM_FIELDS:
        value = getattr(source, name)
        if isinstance(value, (int, float, str)):
            setattr(target, name, value)
        else:
            getattr(target, name)[...] = value


def add_world_geoms(loaded: Loaded, world_id: int, offset: Sequence[float], option: mujoco.MjvOption,
                    pert: mujoco.MjvPerturb, scn: mujoco.MjvScene) -> None:
    """Render `world_id`'s hand into `scn` as extra geoms shifted by `offset`, for showing more than one Warp world
    in a single MuJoCo viewer scene whose own model/data already show world 0 and the shared floor.

    Geoms are selected by body ownership, not by category: the wrist and mount are welded to the world, so they are
    mjCAT_STATIC just like the floor. Every geom owned by a hand body is kept; worldbody geoms (the floor) are
    dropped, so the scene keeps exactly one floor and one floor reflection."""
    data = host_data(loaded, world_id)
    i = scn.ngeom
    mujoco.mjv_addGeoms(loaded.model, data, option, pert, mujoco.mjtCatBit.mjCAT_ALL, scn)
    while i < scn.ngeom:
        geom = scn.geoms[i]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and loaded.model.geom_bodyid[geom.objid] == 0:
            scn.ngeom -= 1
            if i < scn.ngeom:
                _copy_mjv_geom(scn.geoms[scn.ngeom], geom)
            continue
        for axis in range(3):
            geom.pos[axis] += offset[axis]
        i += 1


@wp.kernel
def mit_ctrl_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qpos_adr: wp.array(dtype=int),
    dof_adr: wp.array(dtype=int),
    act_id: wp.array(dtype=int),
    tau_limit: wp.array(dtype=float),
    actuator_biasprm: wp.array2d(dtype=wp.types.vector(length=mujoco.mjNBIAS, dtype=float)),
    q_des: wp.array2d(dtype=float),
    qd_des: wp.array2d(dtype=float),
    tau_ff: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    w, j = wp.tid()
    # Gains come from the actuator bias so the command and the bias term -kp*q - kd*qd always share them; the row
    # index is MuJoCo Warp's rule for a model field shared by all worlds or batched per world.
    bias = actuator_biasprm[w % actuator_biasprm.shape[0], act_id[j]]
    kp = -bias[1]
    kd = -bias[2]
    command = kp * q_des[w, j] + kd * qd_des[w, j] + tau_ff[w, j]
    ctrl[w, act_id[j]] = command


# Slots make assigning a non-field such as `buffers.kp` fail: the MIT gains come only from set_mit_gains.
@dataclass(slots=True)
class MitBuffers:
    q_des: wp.array
    qd_des: wp.array
    tau_ff: wp.array


def mit_buffers(loaded: Loaded) -> MitBuffers:
    shape = (loaded.nworld, len(loaded.joint_names))
    return MitBuffers(**{name: wp.zeros(shape, dtype=float, device=loaded.d.ctrl.device) for name in MIT_FIELDS})


def apply_mit(loaded: Loaded, buffers: MitBuffers) -> None:
    shape = (loaded.nworld, len(loaded.joint_names))
    for name in MIT_FIELDS:
        got = tuple(getattr(buffers, name).shape)
        if got != shape:
            raise ValueError(f"MitBuffers.{name} has shape {got}; expected (nworld, joints) = {shape}")
    index = loaded.mit_index
    wp.launch(
        mit_ctrl_kernel,
        dim=shape,
        inputs=[
            loaded.d.qpos,
            loaded.d.qvel,
            index.qpos_adr,
            index.dof_adr,
            index.act_id,
            index.tau_limit,
            loaded.m.actuator_biasprm,
            *(getattr(buffers, name) for name in MIT_FIELDS),
        ],
        outputs=[loaded.d.ctrl],
        device=loaded.d.ctrl.device,
    )
