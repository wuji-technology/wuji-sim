# SPDX-License-Identifier: Apache-2.0
import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from wuji_sim import conventions, scene

KINDS = ("tracking", "sweep", "grid", "pen_spin")
FOVY_DEG = 27.0
FRAME_FRACTION = 0.90
PREVIEW_HEIGHT_PX = 480
SUPERSAMPLE = 2
# Single-hand views look down by more than half of FOVY_DEG, so the horizon never enters the frame.
PITCH_DEG = {"tracking": 20.0, "sweep": 20.0, "grid": scene.PARALLEL_PITCH_DEG}
# Extra distance beyond FRAME_FRACTION of the envelope, leaving margin around the swept fingers.
SWEEP_DISTANCE_SCALE = 1.08
# Preview look per simulator: MuJoCo raises a single hand above a fixed floor; Isaac Lab drops the floor below the hand.
MUJOCO_FLOOR_Z = 0.0
FLOOR_CLEARANCE_M = 0.08
HAND_RAISE_M = {"mujoco": 0.08, "mujoco_warp": 0.08, "isaaclab": 0.0}
# The camera faces the palm, which points to -y on the right hand and +y on the left (scene.HAND_ROOT_QUAT).
CAMERA_SIDE = {"right": -1.0, "left": 1.0}
# World-frame view that keeps both pen ends in frame over every clip of the wuji-mjlab v2026.10.22 release.
PEN_SPIN_LOOKAT = (-0.11, 0.0, 0.53)
PEN_SPIN_DISTANCE_M = 0.70
PEN_SPIN_AZIMUTH_DEG = -140.0
PEN_SPIN_ELEVATION_DEG = -45.0
PEN_SPIN_FOVY_DEG = 27.0
PEN_SPIN_SIZE_PX = (576, 480)

_TRACKING_SAMPLES = 21
_SWEEP_SAMPLES = 61
_GRID_SEARCH_FAR_M = 50.0
_GRID_SEARCH_STEPS = 60


@dataclass(frozen=True)
class View:
    platform: str
    hand: str
    kind: str
    camera_position: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    fovy_deg: float
    render_size: tuple[int, int]
    floor_z: float
    hand_roots: tuple[tuple[float, float, float], ...]

    @property
    def distance_m(self) -> float:
        return math.dist(self.camera_position, self.camera_target)

    # mjvCamera convention: these angles, camera_target as lookat and distance_m place the camera at camera_position.
    @property
    def azimuth_deg(self) -> float:
        return math.degrees(math.atan2(self.camera_target[1] - self.camera_position[1],
                                       self.camera_target[0] - self.camera_position[0]))

    @property
    def elevation_deg(self) -> float:
        return math.degrees(math.asin((self.camera_target[2] - self.camera_position[2]) / self.distance_m))

    def record(self) -> dict:
        return {
            "platform": self.platform,
            "hand": self.hand,
            "kind": self.kind,
            "camera_position": list(self.camera_position),
            "camera_target": list(self.camera_target),
            "distance_m": self.distance_m,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "fovy_deg": self.fovy_deg,
            "render_size": list(self.render_size),
            "floor_z": self.floor_z,
            "hand_roots": [list(root) for root in self.hand_roots],
        }


def view(mjcf: Path, hand: str, platform: str, kind: str, *, worlds: int = 1) -> View:
    if hand not in CAMERA_SIDE:
        raise ValueError(f"hand must be one of {tuple(CAMERA_SIDE)}, got {hand!r}")
    if platform not in HAND_RAISE_M:
        raise ValueError(f"platform must be one of {tuple(HAND_RAISE_M)}, got {platform!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if worlds < 1 or (kind != "grid" and worlds != 1):
        raise ValueError(f"a {kind} view needs {'worlds >= 1' if kind == 'grid' else 'worlds == 1'}, got {worlds}")
    if kind == "pen_spin":
        return _pen_spin_view(platform, hand)
    low, high = (np.array(bound) for bound in _hand_envelope(Path(mjcf), hand, kind))
    if kind == "grid":
        return _grid_view(platform, hand, low, high, scene.env_grid_offsets(worlds, scene.PARALLEL_SPACING_M))
    return _single_view(platform, hand, kind, low, high)


# Posing the hand through every sampled qpos takes seconds, and the views of one hand on every platform share it.
@functools.cache
def _hand_envelope(mjcf: Path, hand: str, kind: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    model = _hand_model(mjcf, hand)
    names = tuple(model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    if kind == "grid":
        qposes = [scene.pose_vector(names, pose) for pose in scene.POSES]
    else:
        qposes = _tracking_qposes(names)
        if kind == "sweep":
            qposes += _sweep_qposes(names, mjcf)
    low, high = _envelope(model, names, qposes)
    return tuple(low.tolist()), tuple(high.tolist())


def _hand_model(mjcf: Path, hand: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(mjcf))
    for body in spec.worldbody.bodies:
        body.quat = scene.HAND_ROOT_QUAT
    model = spec.compile()
    prefix = conventions.HAND_PREFIXES[hand]
    for j in range(model.njnt):
        if not model.joint(j).name.startswith(prefix):
            raise ValueError(f"{mjcf} is not a {hand} hand model: joint {model.joint(j).name!r} lacks {prefix!r}")
    return model


def _tracking_qposes(names: Sequence[str]) -> list[np.ndarray]:
    period = scene.MOVE_S + scene.HOLD_S
    return [
        scene.tracking_reference(names, segment * period + fraction * scene.MOVE_S)[0]
        for segment in range(len(scene.TRACKING_SEQUENCE) - 1)
        for fraction in np.linspace(0.0, 1.0, _TRACKING_SAMPLES)
    ]


def _sweep_qposes(names: Sequence[str], mjcf: Path) -> list[np.ndarray]:
    description = scene.description_joint_ranges(mjcf)
    ranges = np.array([description[name] for name in names])
    window = scene.sweep_window_duration()
    return [
        scene.sweep_reference(names, ranges, index * window + t)[0]
        for index in range(len(scene.sweep_groups(names)))
        for t in np.linspace(0.0, window, _SWEEP_SAMPLES)
    ]


def _envelope(model: mujoco.MjModel, names: Sequence[str], qposes: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    meshes = []
    for g in range(model.ngeom):
        if (model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] >= 2
                or model.body(int(model.geom_bodyid[g])).name.endswith(scene.MOUNT_SUFFIX)):
            continue
        mesh = int(model.geom_dataid[g])
        start = int(model.mesh_vertadr[mesh])
        meshes.append((g, model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])].astype(np.float64)))
    if not meshes:
        raise ValueError("the hand model has no visible mesh to frame")
    data = mujoco.MjData(model)
    qpos_adr = [model.jnt_qposadr[model.joint(name).id] for name in names]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for qpos in qposes:
        data.qpos[qpos_adr] = qpos
        mujoco.mj_kinematics(model, data)
        for g, vertices in meshes:
            points = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            low = np.minimum(low, points.min(axis=0))
            high = np.maximum(high, points.max(axis=0))
    return low, high


def _away(hand: str, kind: str) -> np.ndarray:
    pitch = math.radians(PITCH_DEG[kind])
    return np.array([0.0, CAMERA_SIDE[hand] * math.cos(pitch), math.sin(pitch)])


def _vector(values) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)


def _floor_z(platform: str, low: np.ndarray) -> float:
    return float(low[2] - FLOOR_CLEARANCE_M) if platform == "isaaclab" else MUJOCO_FLOOR_Z


def _single_view(platform: str, hand: str, kind: str, low: np.ndarray, high: np.ndarray) -> View:
    span = high - low
    raise_m = HAND_RAISE_M[platform]
    target = (low + high) * 0.5 + np.array([0.0, 0.0, raise_m])
    distance = span[2] / (2.0 * math.tan(math.radians(FOVY_DEG) * 0.5) * FRAME_FRACTION)
    if kind == "sweep":
        distance *= SWEEP_DISTANCE_SCALE
    width = round(PREVIEW_HEIGHT_PX * span[0] / span[2])
    return View(
        platform=platform,
        hand=hand,
        kind=kind,
        camera_position=_vector(target + distance * _away(hand, kind)),
        camera_target=_vector(target),
        fovy_deg=FOVY_DEG,
        render_size=(SUPERSAMPLE * width, SUPERSAMPLE * PREVIEW_HEIGHT_PX),
        floor_z=_floor_z(platform, low),
        hand_roots=((0.0, 0.0, raise_m),),
    )


def _grid_view(platform: str, hand: str, low: np.ndarray, high: np.ndarray, roots: np.ndarray) -> View:
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    points = (roots[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    target = (points.min(axis=0) + points.max(axis=0)) * 0.5
    away = _away(hand, "grid")
    forward = -away
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    vertical_limit = math.tan(math.radians(FOVY_DEG) * 0.5) * FRAME_FRACTION

    def extent(distance: float) -> tuple[float, float]:
        relative = points - (target + away * distance)
        depth = relative @ forward
        if not np.all(depth > 0.0):
            return math.inf, math.inf
        return float((np.abs(relative @ right) / depth).max()), float((np.abs(relative @ up) / depth).max())

    near, far = 0.0, _GRID_SEARCH_FAR_M
    if not extent(far)[1] <= vertical_limit:
        raise ValueError(f"the {len(roots)}-world layout does not fit the view even {far} m away")
    for _ in range(_GRID_SEARCH_STEPS):
        middle = 0.5 * (near + far)
        near, far = (near, middle) if extent(middle)[1] <= vertical_limit else (middle, far)
    width = round(PREVIEW_HEIGHT_PX * extent(far)[0] / vertical_limit)
    return View(
        platform=platform,
        hand=hand,
        kind="grid",
        camera_position=_vector(target + away * far),
        camera_target=_vector(target),
        fovy_deg=FOVY_DEG,
        render_size=(SUPERSAMPLE * width, SUPERSAMPLE * PREVIEW_HEIGHT_PX),
        floor_z=_floor_z(platform, low),
        hand_roots=tuple(_vector(root) for root in roots),
    )


def _pen_spin_view(platform: str, hand: str) -> View:
    if hand != "right":
        raise ValueError(f"the pen_spin view frames the right hand at its training wrist pose, got {hand!r}")
    azimuth, elevation = math.radians(PEN_SPIN_AZIMUTH_DEG), math.radians(PEN_SPIN_ELEVATION_DEG)
    forward = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation)])
    target = np.array(PEN_SPIN_LOOKAT)
    return View(
        platform=platform,
        hand=hand,
        kind="pen_spin",
        camera_position=_vector(target - PEN_SPIN_DISTANCE_M * forward),
        camera_target=_vector(target),
        fovy_deg=PEN_SPIN_FOVY_DEG,
        render_size=(SUPERSAMPLE * PEN_SPIN_SIZE_PX[0], SUPERSAMPLE * PEN_SPIN_SIZE_PX[1]),
        floor_z=MUJOCO_FLOOR_Z,
        hand_roots=((0.0, 0.0, 0.0),),
    )
