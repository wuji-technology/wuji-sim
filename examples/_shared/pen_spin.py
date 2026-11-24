# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import http.client
import io
import json
import math
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from wuji_sim import conventions, framing, scene

# From the wuji-mjlab release below (https://github.com/wuji-technology/wuji-mjlab, Apache-2.0); paths are in that repo.
HAND = "right"
RELEASE_TAG = "v2026.10.22"
RELEASE_NAME = f"wuji-mjlab-{RELEASE_TAG}-assets"
RELEASE_URL = f"https://github.com/wuji-technology/wuji-mjlab/releases/download/{RELEASE_TAG}/{RELEASE_NAME}.zip"
RELEASE_BYTES = 246518648
RELEASE_SHA256 = "3a2cc3362a2aec1df6db8ec1b4007174d2ccbf7595c968e33e2b542548d53654"
GROUPS = {
    "01": "01_single_axis",
    "02": "02_continuous_single_axis",
    "03": "03_continuous_multi_axis",
    "04": "04_stable_grasp_rotation",
    "05": "05_multi_axis_two_rotations",
}
DEFAULT_GROUP = "03"
# The default replay is a clip in which the pen turns more than twice and all three simulators hold it to the end.
DEFAULT_CLIP = "clip_005"
DOWNLOAD_TIMEOUT_S = 60.0

JOINTS = 20
LINKS = 21
COMMAND_DIM = JOINTS + 6 + 3 * LINKS
OBS_DIM = COMMAND_DIM + JOINTS + JOINTS + 6 + JOINTS
ACTION_DIM = JOINTS
CONTROL_DT = 0.02
JOINT_VEL_SCALE = 0.25
AXIS_POINT_RADIUS_M = 0.10
AXIS_POINT_TANH_SCALE_M = 0.15
EXPECTED_CONFIG = {"control_mode": "reference_residual", "ctrl_dt": CONTROL_DT, "history_len": 0}

# src/wuji_mjlab/tasks/pen_spin/config/wuji_hand2/hand2_constants.py: training wrist pose (m, wxyz) and MIT gains.
WRIST_POSE_W = (-0.028110361960901435, 0.00300004, 0.4952974018391883,
                0.5416752204197018, 0.45451947767204376, 0.45451947767204376, 0.5416752204197018)
MIT_KP = 3.0
MIT_KD = (0.075, 0.07, 0.05, 0.05)
AMPS_PER_MIT_UNIT = 1.9114709702965849
TORQUE_CONSTANT = (0.4136, 0.3545, 0.2216, 0.1329, 0.4136, 0.1673, 0.1673, 0.1673, 0.4136, 0.1673, 0.1673, 0.1673,
                   0.4136, 0.1673, 0.1673, 0.1673, 0.4461, 0.1115, 0.1115, 0.1115)

# src/wuji_mjlab/assets/objects/aruco_pen250/xmls/pen_tracking.xml: a shaft along +x with a block on each end.
PEN_BODY = "pen"
PEN_MASS_KG = 0.030
PEN_LENGTH_M = 0.250
PEN_DIAMETER_M = 0.0149
PEN_SHAFT_SIDES = 24
PEN_END_HALF_M = (0.009, 0.009, 0.018)
# Turns each end block's box so its long z axis runs along the shaft.
PEN_END_QUAT = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
# From src/wuji_mjlab/assets/objects/aruco_pen250/textures/, as cube faces right, left, up, down, front, back.
PEN_TEXTURE_DIR = Path(__file__).resolve().parent / "textures" / "pen_spin"
PEN_END_TEXTURES = {
    "pen_end_a": ("bottom_right.png", "bottom_left.png", "bottom_up.png", "bottom_down.png", "white.png",
                  "bottom_back.png"),
    "pen_end_b": ("top_right.png", "top_left.png", "top_up.png", "top_down.png", "top_front.png", "white.png"),
}

_AXIS_POINTS = np.array([[AXIS_POINT_RADIUS_M, 0.0, 0.0], [-AXIS_POINT_RADIUS_M, 0.0, 0.0]])


class PenSpinError(RuntimeError):
    pass


def _unit(quat: np.ndarray) -> np.ndarray:
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def _conjugate(quat: np.ndarray) -> np.ndarray:
    return quat * np.array([1.0, -1.0, -1.0, -1.0])


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[..., 1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector + quat[..., :1] * t + np.cross(xyz, t)


def rotation_deg(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    delta = _multiply(quat_b, _conjugate(quat_a))
    return np.degrees(2.0 * np.arctan2(np.linalg.norm(delta[..., 1:], axis=-1), np.abs(delta[..., 0])))


def _rotation_vector(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    """Rotation from `quat_a` to `quat_b` as axis times angle in deg, the angle taken the short way round."""
    delta = _multiply(quat_b, _conjugate(quat_a))
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    sine = np.linalg.norm(delta[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sine, delta[..., :1])
    return np.degrees(np.divide(angle, sine, out=np.zeros_like(sine), where=sine > 0.0) * delta[..., 1:])


def wrist_pose(root) -> np.ndarray:
    pose = np.array(WRIST_POSE_W, dtype=np.float64)
    pose[:3] += root
    pose[3:] = _unit(pose[3:])
    return pose


def to_wrist_local(pose_w, wrist_pose_w: np.ndarray) -> np.ndarray:
    pose_w = np.asarray(pose_w, dtype=np.float64)
    inverse = _conjugate(wrist_pose_w[3:])
    return np.concatenate([_rotate(inverse, pose_w[..., :3] - wrist_pose_w[:3]),
                           _unit(_multiply(inverse, _unit(pose_w[..., 3:])))], axis=-1)


def to_world(pose_local, wrist_pose_w: np.ndarray) -> np.ndarray:
    pose_local = np.asarray(pose_local, dtype=np.float64)
    return np.concatenate([_rotate(wrist_pose_w[3:], pose_local[..., :3]) + wrist_pose_w[:3],
                           _unit(_multiply(wrist_pose_w[3:], pose_local[..., 3:]))], axis=-1)


def axis_points(pen_pose_local) -> np.ndarray:
    pose = np.asarray(pen_pose_local, dtype=np.float64)
    points = pose[..., None, :3] + _rotate(pose[..., None, 3:], _AXIS_POINTS)
    return (AXIS_POINT_TANH_SCALE_M * np.tanh(points / AXIS_POINT_TANH_SCALE_M)).reshape(*pose.shape[:-1], 6)


class Reference:
    """The reference the policy tracks, here one motion clip replayed from its first frame.

    The examples only use num_steps, restart(), command(), joint_pos() and pen_pose(), each indexed by the control
    step, so another source with these members can replace a single clip."""

    def __init__(self, reference_qpos, wrist_pose_w, object_pose_w, link_pos_w):
        qpos = np.array(reference_qpos, dtype=np.float32)
        wrist = np.asarray(wrist_pose_w, dtype=np.float64)
        pen = np.asarray(object_pose_w, dtype=np.float64)
        links = np.asarray(link_pos_w, dtype=np.float64)
        frames = len(qpos)
        shapes = {"reference_qpos": (qpos, (frames, JOINTS)), "wrist_pose_w": (wrist, (frames, 7)),
                  "object_pose_w": (pen, (frames, 7)), "link_pos_w": (links, (frames, LINKS, 3))}
        for name, (value, expected) in shapes.items():
            if frames < 1 or value.shape != expected or not np.isfinite(value).all():
                raise PenSpinError(f"clip array {name} must be finite with shape {expected}, got {value.shape}")
        # Only the quaternions are normalized; positions stay as recorded.
        inverse = _conjugate(_unit(wrist[:, 3:]))
        self.num_steps = frames
        self._joint_pos = qpos
        self._pen_pose = np.concatenate([_rotate(inverse, pen[:, :3] - wrist[:, :3]),
                                         _unit(_multiply(inverse, _unit(pen[:, 3:])))], axis=1)
        links_local = _rotate(inverse[:, None, :], links - wrist[:, None, :3])
        self._command = np.concatenate([qpos, axis_points(self._pen_pose), links_local.reshape(frames, -1)],
                                       axis=1).astype(np.float32)
        for table in (self._joint_pos, self._pen_pose, self._command):
            table.flags.writeable = False

    def restart(self, step: int) -> bool:
        """Whether hand and pen must be put back on the reference before control step `step`."""
        return step % self.num_steps == 0

    def command(self, step: int) -> np.ndarray:
        return self._command[step % self.num_steps]

    def joint_pos(self, step: int) -> np.ndarray:
        return self._joint_pos[step % self.num_steps]

    def pen_pose(self, step: int) -> np.ndarray:
        """Wrist-frame pen pose (position, wxyz quaternion)."""
        return self._pen_pose[step % self.num_steps]


def reference_spin(reference: Reference) -> tuple[np.ndarray, np.ndarray]:
    """The reference pen's spin axis, the wrist-frame direction of the sum of its per-step rotations (zero when that
    sum vanishes), and its signed rotation about that axis summed up to each step, in deg."""
    quats = np.array([reference.pen_pose(step)[3:] for step in range(reference.num_steps)])
    steps = _rotation_vector(quats[:-1], quats[1:])
    total = steps.sum(axis=0)
    norm = np.linalg.norm(total)
    axis = total / norm if norm > 1e-9 else np.zeros(3)
    return axis, np.concatenate([[0.0], np.cumsum(steps @ axis)])


def observation(command, joint_pos, joint_vel, pen_pose_local, last_action) -> np.ndarray:
    """(155,) policy input: command, joint positions, scaled joint velocities, the pen's bounded axis points in the
    wrist frame, and the previous raw action."""
    obs = np.concatenate([command, joint_pos, JOINT_VEL_SCALE * np.asarray(joint_vel), axis_points(pen_pose_local),
                          last_action]).astype(np.float32)
    if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
        raise PenSpinError(f"the observation must be finite with shape ({OBS_DIM},), got shape {obs.shape}: the "
                           "simulation state is no longer finite or an input has the wrong size")
    return obs


class ResidualAction:
    def __init__(self, config: dict):
        self._scale = float(config["action_scale"])
        self._low, self._high = (float(value) for value in config["action_residual_clip"])
        self._alpha = float(config["action_ema_alpha"])
        self._processed = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self) -> None:
        self._processed[:] = 0.0

    def step(self, raw: np.ndarray, reference_joint_pos: np.ndarray) -> np.ndarray:
        residual = np.clip(raw * self._scale, self._low, self._high)
        self._processed = (self._alpha * residual + (1.0 - self._alpha) * self._processed).astype(np.float32)
        return reference_joint_pos + self._processed


class Policy:
    """The published policy with ONNX Runtime on the CPU, one observation per run."""

    def __init__(self, model_bytes: bytes):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(model_bytes, sess_options=options,
                                                     providers=["CPUExecutionProvider"])
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        shapes = ([value.shape for value in inputs], [value.shape for value in outputs])
        if shapes != ([[1, OBS_DIM]], [[1, ACTION_DIM]]):
            raise PenSpinError(f"the policy must have one input of shape [1, {OBS_DIM}] and one output of shape "
                               f"[1, {ACTION_DIM}], got inputs {shapes[0]} and outputs {shapes[1]}")
        self._input = inputs[0].name
        self._output = outputs[0].name
        self._runtime = f"onnxruntime {onnxruntime.__version__}"
        self.session_runs = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        feed = np.ascontiguousarray(obs, dtype=np.float32)[None]
        action = self._session.run([self._output], {self._input: feed})[0][0]
        self.session_runs += 1
        return action

    def record(self) -> dict:
        return {"runtime": self._runtime, "provider": "CPUExecutionProvider", "session_runs": self.session_runs,
                "observation_shape": [1, OBS_DIM], "action_shape": [1, ACTION_DIM]}


def default_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "wuji-sim"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(path: Path, opener) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RELEASE_URL} ({RELEASE_BYTES / 1e6:.0f} MB) to {path}", flush=True)
    # A unique temporary file: a failed download never looks like the archive and concurrent runs do not collide.
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    try:
        digest = hashlib.sha256()
        received, reported = 0, 0
        with os.fdopen(handle, "wb") as out, opener(RELEASE_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:
            while chunk := response.read(1 << 20):
                if received + len(chunk) > RELEASE_BYTES:
                    raise PenSpinError(f"the download of {RELEASE_URL} exceeds the expected {RELEASE_BYTES} bytes; "
                                       "nothing was cached")
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received - reported >= RELEASE_BYTES // 10:
                    reported = received
                    print(f"  {received / 1e6:.0f} MB", flush=True)
        if digest.hexdigest() != RELEASE_SHA256:
            raise PenSpinError(f"the download of {RELEASE_URL} has SHA-256 {digest.hexdigest()}, expected "
                               f"{RELEASE_SHA256}; nothing was cached")
        os.replace(temporary, path)
    # A connection that drops mid-body raises http.client.IncompleteRead, which is not an OSError.
    except (OSError, http.client.HTTPException) as exc:
        raise PenSpinError(f"could not download {RELEASE_URL}: {exc}; to run without a download, place the archive "
                           f"at {path}") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


def release_zip(cache_dir: Path | None = None, *, opener=None) -> Path:
    """The release archive in `cache_dir`, downloaded on first use; its SHA-256 is checked on every call."""
    path = (Path(cache_dir) if cache_dir is not None else default_cache_dir()).absolute() / f"{RELEASE_NAME}.zip"
    if not path.is_file():
        # Looked up here, not bound as a default argument, so a patched urlopen is honoured.
        _download(path, opener or urllib.request.urlopen)
    digest = _sha256(path)
    if digest != RELEASE_SHA256:
        raise PenSpinError(f"{path} has SHA-256 {digest}, expected {RELEASE_SHA256} ({RELEASE_URL}); delete the file "
                           "to download it again")
    return path


@dataclass(frozen=True)
class Task:
    group: str
    clip: str
    reference: Reference
    model: bytes = field(repr=False)
    config: dict
    source: dict

    def describe(self) -> str:
        return (f"pen-spinning policy {self.group} from wuji-mjlab {RELEASE_TAG}, clip {self.clip} "
                f"({self.reference.num_steps} frames, {self.reference.num_steps * CONTROL_DT:.2f} s)")

    def record(self) -> dict:
        return {**self.source, "config": self.config}


def load(group: str, clip: str | None = None, cache_dir: Path | None = None, *, opener=None) -> Task:
    if group not in GROUPS:
        raise PenSpinError(f"group must be one of {sorted(GROUPS)}, got {group!r}")
    path = release_zip(cache_dir, opener=opener)
    name = GROUPS[group]
    checkpoint = f"{RELEASE_NAME}/pen_spin/checkpoints/{name}"
    clips = f"{RELEASE_NAME}/pen_spin/data/rawdata/clips/{name}"
    try:
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read(f"{checkpoint}/config.json"))
            index = json.loads(archive.read(f"{clips}/index.json"))
            entries = {entry["clip_id"]: entry for entry in index["clips"]}
            if clip is None:
                clip = DEFAULT_CLIP if group == DEFAULT_GROUP else next(iter(entries))
            if clip not in entries:
                raise PenSpinError(f"group {name} has no clip {clip!r}; its clips are {list(entries)}")
            model = archive.read(f"{checkpoint}/policy.onnx")
            with np.load(io.BytesIO(archive.read(f"{clips}/{entries[clip]['file']}")), allow_pickle=False) as data:
                reference = Reference(data["reference_qpos"], data["wrist_pose_w"], data["object_pose_w"],
                                      data["link_pos_w"])
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise PenSpinError(f"{path} does not hold the expected files of group {name}: {exc!r}") from exc
    wrong = {key: config.get(key) for key, value in EXPECTED_CONFIG.items() if config.get(key) != value}
    if wrong or index.get("frame_dt") != CONTROL_DT:
        raise PenSpinError(f"group {name} is not a {CONTROL_DT} s reference-residual policy: config {wrong}, clip "
                           f"frame_dt {index.get('frame_dt')}")
    source = {
        "url": RELEASE_URL,
        "release": RELEASE_TAG,
        "archive": str(path),
        "archive_sha256": RELEASE_SHA256,
        "group": name,
        "clip": clip,
        "clip_sha256": entries[clip]["sha256"],
        "policy_sha256": hashlib.sha256(model).hexdigest(),
        "frames": reference.num_steps,
        "frame_dt": CONTROL_DT,
    }
    return Task(group=name, clip=clip, reference=reference, model=model, config=config, source=source)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--group", choices=sorted(GROUPS), default=DEFAULT_GROUP,
                        help="motion group of the published policy: " + ", ".join(GROUPS.values())
                             + f" (default: {DEFAULT_GROUP})")
    parser.add_argument("--clip", default=None,
                        help=f"clip of the group to replay, for example clip_003 (default: {DEFAULT_CLIP} for group "
                             f"{DEFAULT_GROUP}, the group's first clip for the others)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help=f"directory that holds {RELEASE_NAME}.zip; the archive is downloaded there on the first "
                             "run (default: $XDG_CACHE_HOME/wuji-sim, else ~/.cache/wuji-sim)")


def gains() -> tuple[np.ndarray, np.ndarray]:
    """Per-joint kp (N*m/rad) and kd (N*m*s/rad) the policy was trained with."""
    scale = np.array(TORQUE_CONSTANT) * AMPS_PER_MIT_UNIT
    return scale * MIT_KP, scale * np.tile(MIT_KD, JOINTS // len(MIT_KD))


def check_joint_order(joint_names, hand: str) -> None:
    expected = tuple(conventions.HAND_PREFIXES[hand] + suffix for suffix in conventions.JOINT_SUFFIXES)
    if tuple(joint_names) != expected:
        raise PenSpinError(f"the policy needs the joints in the order {expected}, the model has {tuple(joint_names)}")


def pen_shape() -> dict:
    radius = PEN_DIAMETER_M / 2
    shaft_half = PEN_LENGTH_M / 2 - 2 * PEN_END_HALF_M[2]
    end_center = PEN_LENGTH_M / 2 - PEN_END_HALF_M[2]
    side, block_length = 2 * PEN_END_HALF_M[0], 2 * PEN_END_HALF_M[2]
    shaft_volume = math.pi * radius**2 * 2 * shaft_half
    block_volume = side * side * block_length
    density = PEN_MASS_KG / (shaft_volume + 2 * block_volume)
    shaft_mass, block_mass = density * shaft_volume, density * block_volume
    axial = shaft_mass * radius**2 / 2 + 2 * block_mass * (side**2 + side**2) / 12
    transverse = (shaft_mass * (3 * radius**2 + (2 * shaft_half) ** 2) / 12
                  + 2 * block_mass * ((side**2 + block_length**2) / 12 + end_center**2))
    angles = [2.0 * math.pi * k / PEN_SHAFT_SIDES for k in range(PEN_SHAFT_SIDES)]
    vertices = [[x, radius * math.cos(a), radius * math.sin(a)] for x in (-shaft_half, shaft_half) for a in angles]
    return {"radius": radius, "shaft_half": shaft_half, "end_center": end_center,
            "inertia": [axial, transverse, transverse], "shaft_vertices": vertices}


def add_pen(model_source: mujoco.MjSpec, pen_pose_w) -> None:
    # Training-setup inertia (round shaft + both end blocks), set on the body so the collision mesh cannot change it.
    shape = pen_shape()
    radius, shaft_half, end_center = shape["radius"], shape["shaft_half"], shape["end_center"]
    body = model_source.worldbody.add_body(name=PEN_BODY, pos=[float(v) for v in pen_pose_w[:3]],
                                           quat=[float(v) for v in pen_pose_w[3:]])
    body.add_freejoint()
    body.explicitinertial = 1
    body.mass = PEN_MASS_KG
    body.ipos = [0.0, 0.0, 0.0]
    body.iquat = [1.0, 0.0, 0.0, 0.0]
    body.inertia = shape["inertia"]
    model_source.add_material(name="pen_shaft", rgba=[1.0, 1.0, 1.0, 1.0], specular=0.0)
    body.add_geom(name="pen_shaft", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  fromto=[-shaft_half, 0.0, 0.0, shaft_half, 0.0, 0.0], size=[radius, 0.0, 0.0], contype=0,
                  conaffinity=0, material="pen_shaft")
    # Collides as a prism mesh: MuJoCo Warp 3.11 gives cylinder-mesh pairs one contact point and the pen slips out.
    model_source.add_mesh(name="pen_shaft_collision",
                          uservert=[c for vertex in shape["shaft_vertices"] for c in vertex])
    body.add_geom(name="pen_shaft_collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pen_shaft_collision",
                  group=2, material="pen_shaft")
    for name, x in (("pen_end_a", -end_center), ("pen_end_b", end_center)):
        # Absolute paths: the compiler looks for a relative texture file next to the hand model, not here.
        model_source.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                                 cubefiles=[str(PEN_TEXTURE_DIR / face) for face in PEN_END_TEXTURES[name]])
        material = model_source.add_material(name=name, specular=0.0)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, 0.0, 0.0],
                      quat=list(PEN_END_QUAT), size=list(PEN_END_HALF_M), material=name)


def add_pen_scene(model_source: mujoco.MjSpec, hand: str, wrist_pose_w: np.ndarray, pen_pose_w) -> None:
    """Call after scene.add_scene: moves the wrist mount so the wrist body sits at `wrist_pose_w`, then adds the pen."""
    prefix = conventions.HAND_PREFIXES[hand]
    mount = model_source.body(prefix + "mount")
    wrists = [body for body in mount.bodies if body.name == prefix + "wrist"]
    if len(wrists) != 1:
        raise PenSpinError(f"the hand model must have {prefix}wrist as a direct child of {prefix}mount")
    quat = _unit(_multiply(wrist_pose_w[3:], _conjugate(_unit(np.array(wrists[0].quat, dtype=np.float64)))))
    mount.quat = quat.tolist()
    mount.pos = (wrist_pose_w[:3] - _rotate(quat, np.array(wrists[0].pos, dtype=np.float64))).tolist()
    add_pen(model_source, pen_pose_w)


def pen_qpos_adr(model: mujoco.MjModel) -> int:
    return int(model.jnt_qposadr[model.body(PEN_BODY).jntadr[0]])


def scene_record(hand: str, view: framing.View, wrist_pose_w: np.ndarray) -> dict:
    record = scene.scene_record(hand, view)
    del record["finger_axis"]
    record["wrist_pose_w"] = wrist_pose_w.tolist()
    record["wrist_pose_note"] = ("position (m) and wxyz quaternion of the wrist body, the wrist pose the policy was "
                                 "trained with: palm up")
    return record


def pen_record() -> dict:
    return {"body": PEN_BODY, "mass_kg": PEN_MASS_KG, "length_m": PEN_LENGTH_M, "diameter_m": PEN_DIAMETER_M,
            "geometry": ("a round white shaft with a block on each end that shows the marker labels of the real pen, "
                         f"built in code; the shaft collides as a {PEN_SHAFT_SIDES}-sided prism mesh; mass and "
                         "inertia are the training setup's values with the mass spread by volume")}


class PenLog:
    """What the pen did during one pass; numbers to look at, nothing is scored."""

    def __init__(self, reference: Reference, wrist_pose_w: np.ndarray):
        self._reference = reference
        self._wrist = wrist_pose_w
        self._steps = 0
        self._start_quat = None
        self._quat = None
        self._path_rotation = 0.0
        self._spin_axis, self._reference_spin = reference_spin(reference)
        self._spin_rotation = 0.0
        self._path_rotation_at_s: list[float] = []
        self._distance: list[float] = []
        self._height: list[float] = []

    def start(self, pen_pose_w) -> None:
        pose = np.asarray(pen_pose_w, dtype=np.float64)
        self._start_quat = to_wrist_local(pose, self._wrist)[3:]
        self._quat = self._start_quat
        self._height.append(float(pose[2]))

    def update(self, step: int, pen_pose_w) -> None:
        """Call with the pen pose right after control step `step`."""
        pose = np.asarray(pen_pose_w, dtype=np.float64)
        local = to_wrist_local(pose, self._wrist)
        self._path_rotation += float(rotation_deg(self._quat, local[3:]))
        self._spin_rotation += float(_rotation_vector(self._quat, local[3:]) @ self._spin_axis)
        self._quat = local[3:]
        self._steps += 1
        if self._steps % round(1.0 / CONTROL_DT) == 0:
            self._path_rotation_at_s.append(self._path_rotation)
        self._distance.append(float(np.linalg.norm(local[:3] - self._reference.pen_pose(step)[:3])))
        self._height.append(float(pose[2]))

    def record(self) -> dict:
        frames = np.array([self._reference.pen_pose(k)[3:] for k in range(self._reference.num_steps)])
        reference_path = np.concatenate([[0.0], np.cumsum(rotation_deg(frames[:-1], frames[1:]))])
        per_second = round(1.0 / CONTROL_DT)
        seconds = list(range(1, len(self._path_rotation_at_s) + 1))
        return {
            "spin_rotation_definition": "signed rotation of the pen about the spin axis, summed over the control "
                                        "steps, in deg; 360 deg is one whole turn. The spin axis is the wrist-frame "
                                        "direction of the sum of the reference's per-step rotations, and pen and "
                                        "reference are measured about that same axis",
            "spin_axis": [float(value) for value in self._spin_axis],
            "pen_spin_rotation_deg": self._spin_rotation,
            "reference_spin_rotation_deg": float(self._reference_spin[-1]),
            "net_rotation_definition": "angle of the single rotation between the pen's orientation before the first "
                                       "control step and after the last one (for the reference: between its first "
                                       "and last frame), in deg, 0 to 180",
            "pen_net_rotation_deg": float(rotation_deg(self._start_quat, self._quat)),
            "reference_net_rotation_deg": float(rotation_deg(frames[0], frames[-1])),
            "path_rotation_definition": "sum over the control steps of the angle the pen rotates during the step, in "
                                        "deg; it counts every wobble back and forth, so it is larger than the net "
                                        "rotation",
            "path_rotation_sampled_at_s": seconds,
            "pen_path_rotation_deg": list(self._path_rotation_at_s),
            "pen_path_rotation_total_deg": self._path_rotation,
            "reference_path_rotation_deg": [float(reference_path[min(s * per_second, len(reference_path) - 1)])
                                            for s in seconds],
            "reference_path_rotation_total_deg": float(reference_path[-1]),
            "reference_distance_definition": "distance between the pen centre after a control step and the "
                                             "reference pen centre of that step, in m",
            "reference_distance_max_m": max(self._distance),
            "reference_distance_final_m": self._distance[-1],
            "height_start_m": self._height[0],
            "height_min_m": min(self._height),
            "height_final_m": self._height[-1],
        }


def describe_replay(replay: dict) -> str:
    pen, reference = replay["pen_spin_rotation_deg"], replay["reference_spin_rotation_deg"]
    return (f"{replay['steps']} control steps, pen turned {pen / 360:.1f} turns ({pen:.0f} deg) about its spin axis "
            f"(reference {reference / 360:.1f} turns, {reference:.0f} deg), at most "
            f"{replay['reference_distance_max_m']:.3f} m from the reference, lowest height "
            f"{replay['height_min_m']:.3f} m")
