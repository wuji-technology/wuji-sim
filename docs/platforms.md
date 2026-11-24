# Platform Profiles

Simulator settings live in `wuji_sim/profiles/mujoco.toml`, `mujoco_warp.toml` and `isaaclab.toml`; hand model
values (geometry, inertia, joint ranges) are not stored there. Every parameter has a `value` and `unit`. A
`source` says where a value comes from, in one of four forms: `default: <project> <version>` (an upstream
default), `choice` (a value wuji-sim selected), `estimate` or `estimate: <basis>`, and `measured: <reference>`. A
group-level `source` applies to the parameters that set none of their own.

| Group | Parameter | MuJoCo | MuJoCo Warp | MotrixSim |
| :--- | :--- | :--- | :--- | :--- |
| `option` | timestep | 0.005 s | 0.005 s | 0.005 s |
| `option` | integrator | implicitfast | implicitfast | n/a (backend-chosen) |
| `option` | solver | Newton | Newton | Newton |
| `option` | iterations | 100 | 100 | 100 |
| `option` | tolerance | 1e-8 | 1e-6 | n/a (backend default 1e-4) |
| `option` | ls_iterations | 50 | 50 | n/a |
| `contact` | geom_friction | [0.7, 0.005 m, 0.0001 m] | [0.7, 0.005 m, 0.0001 m] | [0.7, 0.005 m, 0.0001 m] |
| `contact` | geom_condim | 4 | 4 | 4 |
| `joint_coulomb_friction` | every hinge joint | 0.0 N·m | 0.0 N·m | 0.0 N·m |
| `joint_damping` | every hinge joint | 0.0 N·m·s/rad | 0.0 N·m·s/rad | 0.0 N·m·s/rad |
| `capacity` | nconmax | n/a | 256 contacts/world | n/a |
| `capacity` | njmax | n/a | 1024 constraint rows/world | n/a |

`wuji_sim/model_overrides.toml` holds the hand's armature and joint torque limits; both are estimates not
confirmed against hardware. A `tau_limits` mapping passed to the adapter's `load()` takes precedence over
`joint_torque_limit`.

`[overrides.<group>.<name>]` with a `value` and a non-empty `reason` overrides a parameter for a single run;
applied overrides are reported as `with override: ...`.

## MIT Control

```
tau = clip(kp * (q_des - q) + kd * (qd_des - qd) + tau_ff, -tau_limit, tau_limit)
```

MuJoCo and MuJoCo Warp evaluate this law with an integrated actuator. `mit_stability()` rejects integrators other
than `implicit` and `implicitfast`, and gains with `kp*dt^2/I >= 4`. `set_mit_gains()` does not run this check, so
call `mit_stability()` before installing gains.

## MotrixSim

The adapter builds and patches the hand MJCF on a `mujoco.MjSpec` exactly like the MuJoCo platform, serializes the
patched spec back to MJCF and imports it into MotrixSim (`motrixsim.msd.from_str` → `msd.World.build()`). The
compiled MuJoCo model of the same spec serves as the reporting twin; the backend quantities MotrixSim exposes
(runtime options, joint armature/frictionloss/range, actuator control limits) are cross-checked against it and any
deviation is recorded in the load report as backend-imposed.

MotrixSim has no MuJoCo integrator or line-search option, so the profile carries no `option.integrator` or
`option.ls_iterations`; the backend picks its own integration scheme. Its integrator treats the actuator damping
implicitly (verified empirically: kp = 60, kd = 1.5 at dt = 0.005 s settles stably where an explicit scheme would
diverge), so the MIT law uses the same integrated actuator as MuJoCo and the `kp*dt^2/I < 4` bound applies.
Solver choice maps onto MotrixSim's `Newton`/`PGS`; `CG` is not available. `option.tolerance` reaches the backend
(`constraint_solver_tolerance`) but is **not equivalent** to MuJoCo's tolerance: MotrixSim stops on the solver
residual, MuJoCo on the cost improvement (PGS) or gradient norm (Newton/CG), so the same number is not the same
convergence tightness; the value is carried for configuration comparability only. Actuator commands must go
through `Actuator.set_ctrl` — assignment into `scene_data.actuator_ctrls` is not consumed by `step`.

## Isaac Lab

| Group | Parameter | Isaac Lab |
| :--- | :--- | :--- |
| `option` | timestep | 0.005 s |
| `contact` | default_material_static_friction | 0.7 |
| `contact` | default_material_dynamic_friction | 0.7 |
| `joint_velocity_limit` | every hinge joint | 100.0 rad/s |
| `joint_friction` | every hinge joint | 0.0 N·m (static friction torque) |

Isaac Lab drives the MIT law with PhysX implicit joint drives, so `tau_ff` must be zero. Load one scene per
Isaac Lab process, and call `step()` once after `set_joint_state()` before rendering the new pose.

## Scope

- Deformable bodies, cloth, SDF collision and muscle models are out of scope.
- Validation levels are those listed in the README Support Matrix.
- Custom parameter overrides and locally modified assets are not validated.
