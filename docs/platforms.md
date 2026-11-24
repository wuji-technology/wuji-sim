# Platform Profiles

Simulator settings live in `wuji_sim/profiles/mujoco.toml`, `mujoco_warp.toml` and `isaaclab.toml`; hand model
values (geometry, inertia, joint ranges) are not stored there. Every parameter has a `value` and `unit`. A
`source` says where a value comes from, in one of four forms: `default: <project> <version>` (an upstream
default), `choice` (a value wuji-sim selected), `estimate` or `estimate: <basis>`, and `measured: <reference>`. A
group-level `source` applies to the parameters that set none of their own.

| Group | Parameter | MuJoCo | MuJoCo Warp |
| :--- | :--- | :--- | :--- |
| `option` | timestep | 0.005 s | 0.005 s |
| `option` | integrator | implicitfast | implicitfast |
| `option` | solver | Newton | Newton |
| `option` | iterations | 100 | 100 |
| `option` | tolerance | 1e-8 | 1e-6 |
| `option` | ls_iterations | 50 | 50 |
| `contact` | geom_friction | [0.7, 0.005 m, 0.0001 m] | [0.7, 0.005 m, 0.0001 m] |
| `contact` | geom_condim | 4 | 4 |
| `joint_coulomb_friction` | every hinge joint | 0.0 N·m | 0.0 N·m |
| `joint_damping` | every hinge joint | 0.0 N·m·s/rad | 0.0 N·m·s/rad |
| `capacity` | nconmax | n/a | 256 contacts/world |
| `capacity` | njmax | n/a | 1024 constraint rows/world |

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
