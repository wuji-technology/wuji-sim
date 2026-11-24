# SPDX-License-Identifier: Apache-2.0
"""The pen-spin pen built for MotrixSim through the native msd authoring API.

The hand, floor, cameras and the actuator replacement are authored on a `mujoco.MjSpec` and imported as MJCF
(see wuji_sim/platforms/motrixsim.py), but the pen is authored directly on the imported `msd.World` through
`platform.load(world_scene=...)` — the same per-backend-native precedent the isaaclab platform sets with its
USD (pxr) authoring. Mass, inertia and collision geometry stay field-identical to
examples/_shared/pen_spin.add_pen because every number comes from the same shared `pen_spin` constants; the end
blocks get plain colors instead of MuJoCo cube textures (MotrixSim's MJCF import takes no cube textures), so the
marker labels of the real pen are two shades of grey.

msd quats are xyzw (the floating-base DoF layout and World.attach's rotation convention agree); `pen_pose_w`
arrives wxyz from the MJCF world, so it is converted here. Geoms authored natively set their own
PhysicsMaterial — the profile's contact parameters travel the XML route and do not reach this side, so the
caller passes them in `contact`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))

import pen_spin  # noqa: E402
from motrixsim import msd  # noqa: E402

END_COLOR = {"pen_end_a": [0.75, 0.75, 0.78, 1.0], "pen_end_b": [0.35, 0.35, 0.40, 1.0]}
# The visible shaft is a cylinder along the msd Z axis; the pen's long axis is X, so it is rotated 90 deg
# about Y (xyzw quaternion with unit real part last).
CYLINDER_TO_X = [0.0, 2.0 ** -0.5, 0.0, 2.0 ** -0.5]


def _xyzw(wxyz) -> np.ndarray:
    w, x, y, z = (float(value) for value in wxyz)
    return np.array([x, y, z, w], dtype=np.float32)


def _physics_material(contact: dict) -> msd.PhysicsMaterial:
    material = msd.PhysicsMaterial()
    material.friction = [float(value) for value in contact["geom_friction"]]
    material.condim = int(contact["geom_condim"])
    return material


def add_pen_world(world, pen_pose_w, contact: dict) -> None:
    """Author the pen natively on the imported world: explicit training-setup inertia (the collision mesh
    cannot change it), a visible cylinder shaft, the prism collision mesh MuJoCo's demos use (Warp 3.11 gives
    cylinder-mesh pairs one contact point and the pen slips out), and the two end blocks."""
    shape = pen_spin.pen_shape()
    radius, shaft_half, end_center = shape["radius"], shape["shaft_half"], shape["end_center"]

    body = msd.Body()
    body.link.name = pen_spin.PEN_BODY
    body.link.local_translation = np.asarray(pen_pose_w[:3], dtype=np.float32)
    body.link.local_rotation = _xyzw(pen_pose_w[3:])
    body.free = msd.FreeJoint()
    inertia = msd.Inertia()
    inertia.mass = pen_spin.PEN_MASS_KG
    inertia.diag_inertia = np.asarray(shape["inertia"], dtype=np.float32)
    inertia.pos = np.zeros(3, dtype=np.float32)
    inertia.orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    body.link.inertial = inertia

    material = _physics_material(contact)

    shaft = msd.Geometry()
    shaft.name = "pen_shaft"
    shaft.shape = msd.ShapeType.Cylinder
    # Cylinder sizes are [radius, radius, half-length] along the shape's Z axis.
    shaft.size = [radius, radius, shaft_half]
    shaft.orientation = np.asarray(CYLINDER_TO_X, dtype=np.float32)
    shaft.collision_mask.collide_group = 0
    shaft.collision_mask.collide_with = 0
    shaft.visual.color = [1.0, 1.0, 1.0, 1.0]
    body.link.geoms.append(shaft)

    data = msd.MeshData()
    data.vertices = np.asarray(shape["shaft_vertices"], dtype=np.float32)
    asset = msd.MeshAsset()
    asset.source = msd.MeshSource.buffer(data)
    world.assets.meshes["pen_shaft_collision"] = asset
    collision = msd.Geometry()
    collision.name = "pen_shaft_collision"
    collision.shape = msd.ShapeType.Mesh
    collision.mesh = "pen_shaft_collision"
    collision.visual.group = 2  # hidden in the demos, like the MjSpec-authored collision mesh
    collision.physics_material = material
    body.link.geoms.append(collision)

    for name, x in (("pen_end_a", -end_center), ("pen_end_b", end_center)):
        block = msd.Geometry()
        block.name = name
        block.shape = msd.ShapeType.Box
        block.size = [float(value) for value in pen_spin.PEN_END_HALF_M]
        block.position = np.array([x, 0.0, 0.0], dtype=np.float32)
        block.orientation = _xyzw(pen_spin.PEN_END_QUAT)
        block.visual.color = [float(value) for value in END_COLOR[name]]
        block.physics_material = material
        body.link.geoms.append(block)

    world.hierarchy.bodies.append(body)


def move_wrist_mount(model_source, hand: str, wrist_pose_w: np.ndarray) -> None:
    """Call after scene.add_scene: moves the wrist mount so the wrist body sits at `wrist_pose_w`
    (pen_spin.add_pen_scene's mount move; the pen itself is authored natively through add_pen_world)."""
    from wuji_sim import conventions

    prefix = conventions.HAND_PREFIXES[hand]
    mount = model_source.body(prefix + "mount")
    wrists = [body for body in mount.bodies if body.name == prefix + "wrist"]
    if len(wrists) != 1:
        raise pen_spin.PenSpinError(f"the hand model must have {prefix}wrist as a direct child of {prefix}mount")
    quat = pen_spin._unit(pen_spin._multiply(wrist_pose_w[3:],
                                             pen_spin._conjugate(pen_spin._unit(np.array(wrists[0].quat,
                                                                                          dtype=np.float64)))))
    mount.quat = quat.tolist()
    mount.pos = (wrist_pose_w[:3] - pen_spin._rotate(quat, np.array(wrists[0].pos, dtype=np.float64))).tolist()


def pen_record() -> dict:
    record = pen_spin.pen_record()
    record["geometry"] = (
        "a round white shaft with a block on each end, authored natively through motrixsim.msd; MotrixSim takes "
        "no cube textures, so the marker labels of the real pen are replaced by two shades of grey; the shaft "
        f"collides as a {pen_spin.PEN_SHAFT_SIDES}-sided prism mesh; mass and inertia are the training setup's "
        "values with the mass spread by volume"
    )
    return record
