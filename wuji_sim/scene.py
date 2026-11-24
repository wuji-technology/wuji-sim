# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from wuji_sim import conventions

if TYPE_CHECKING:
    from wuji_sim.framing import View

CAMERAS = ("front", "side", "back")
PREVIEW_CAMERA = "preview"
POSES = ("open", "half_grip", "fist", "pinch")

# Shared by both hands (mirrored joint axes); presets and the straight paths to them from open are free of self-contact.
_OPEN = dict.fromkeys(conventions.JOINT_SUFFIXES, 0.0)
_FIST = {
    "thumb_cmc_flex": 0.625, "thumb_cmc_abd": -1.0, "thumb_mcp": 1.085, "thumb_ip": 1.085,
    "index_finger_mcp_flex": 1.3, "index_finger_mcp_abd": -0.15, "index_finger_pip": 1.9, "index_finger_dip": 1.1,
    "middle_finger_mcp_flex": 1.3, "middle_finger_mcp_abd": 0.0, "middle_finger_pip": 1.9, "middle_finger_dip": 1.1,
    "ring_finger_mcp_flex": 1.3, "ring_finger_mcp_abd": 0.075, "ring_finger_pip": 1.9, "ring_finger_dip": 1.1,
    "pinky_mcp_flex": 1.3, "pinky_mcp_abd": 0.15, "pinky_pip": 1.9, "pinky_dip": 1.1,
}
PRESET_POSES = {
    "open": _OPEN,
    "half_grip": {suffix: (_OPEN[suffix] + _FIST[suffix]) / 2 for suffix in conventions.JOINT_SUFFIXES},
    "fist": _FIST,
    "pinch": {
        "thumb_cmc_flex": 0.96, "thumb_cmc_abd": -0.45, "thumb_mcp": 0.34, "thumb_ip": 0.31,
        "index_finger_mcp_flex": 0.7, "index_finger_mcp_abd": 0.0, "index_finger_pip": 0.87, "index_finger_dip": 0.64,
        "middle_finger_mcp_flex": 0.0, "middle_finger_mcp_abd": 0.0, "middle_finger_pip": 0.0, "middle_finger_dip": 0.0,
        "ring_finger_mcp_flex": 0.0, "ring_finger_mcp_abd": 0.0, "ring_finger_pip": 0.0, "ring_finger_dip": 0.0,
        "pinky_mcp_flex": 0.0, "pinky_mcp_abd": 0.0, "pinky_pip": 0.0, "pinky_dip": 0.0,
    },
}
TRACKING_SEQUENCE = ("open", "half_grip", "fist", "open", "pinch", "open")
MOVE_S = 0.6
HOLD_S = 0.2
DEMO_KP = 1.0  # N*m/rad
DEMO_KD = 0.03  # N*m*s/rad

SWEEP_WAYPOINTS = ("open", "inner_lower", "inner_upper", "open")
SWEEP_MARGIN = 0.05  # fraction of each joint's travel kept clear of both limits; the limits themselves are unchanged
SWEEP_MOVE_S = 0.8
SWEEP_HOLD_S = 0.2
SWEEP_ERROR_FLAG_RAD = 0.05
# The four fingers sweep the same role together, so no finger is swept against neighbours held at open.
SWEEP_FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
SWEEP_THUMB_PREFIX = "thumb_"
SWEEP_SERIES = ("q_des", "q", "qd", "tau_command", "tau_applied")
SWEEP_SERIES_UNITS = {"q_des": "rad", "q": "rad", "qd": "rad/s", "tau_command": "N*m", "tau_applied": "N*m"}
# MuJoCo Warp stores the compiled joint ranges in float32 (1.57 rad reads back as 1.5700000524520874).
JOINT_RANGE_ATOL = 1e-6  # rad

# The hand is fixed at its wrist mount and rotated so every finger extends along +z.
HAND_ROOT_QUAT = [0.0, 1.0, 0.0, 0.0]
MOUNT_SUFFIX = "_mount"
# The right palm faces -y and the left palm +y; the thumb of both hands points to +x.
FRONT_CAMERA_POS = {"right": [0.03, 0.35, 0.20], "left": [0.03, -0.35, 0.20]}
SIDE_CAMERA_POS = [0.36, 0.0, 0.20]
# Front and back cameras remain opposite each other around the proximal-finger target.
BACK_CAMERA_POS = {"right": [-0.017, -0.334, 0.20], "left": [-0.017, 0.334, 0.20]}
PREVIEW_TARGET_BODY = "scene_preview_target"
FLOOR_CHECKER_RGB = {
    "mujoco": ([0.2, 0.3, 0.4], [0.1, 0.2, 0.3]),
    "mujoco_warp": ([0.2, 0.4, 0.3], [0.1, 0.3, 0.2]),
}
# The hand AABB is about 0.24 m, so 0.3 m keeps neighbours on the parallel-demo grid apart within one view.
PARALLEL_SPACING_M = 0.3
# Looking down this far lifts the back row of the grid clear of the front row; at 35 deg or less they overlap.
PARALLEL_PITCH_DEG = 40.0


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headless", action="store_true", help="only write files; do not open the MuJoCo viewer")


def add_scene(spec: mujoco.MjSpec, hand: str, view: "View") -> None:
    # Imported here: framing imports this module while it loads.
    from wuji_sim import framing

    if hand not in conventions.HAND_PREFIXES:
        raise ValueError(f"hand must be one of {tuple(conventions.HAND_PREFIXES)}, got {hand!r}")
    if view.hand != hand or view.platform not in FLOOR_CHECKER_RGB:
        raise ValueError(
            f"add_scene needs a view of the {hand} hand on one of {tuple(FLOOR_CHECKER_RGB)}, "
            f"got the {view.hand} hand on {view.platform!r}"
        )
    side = framing.CAMERA_SIDE[hand]
    root = list(view.hand_roots[0])
    spec.option.gravity = conventions.GRAVITY
    for body in spec.worldbody.bodies:
        body.pos = root
        body.quat = HAND_ROOT_QUAT
        # The wrist mount is a test-bench fixture; group 2 hides it in every view, as in the README media.
        if body.name.endswith(MOUNT_SUFFIX):
            for geom in body.geoms:
                geom.group = 2
    spec.visual.global_.fovy = view.fovy_deg
    spec.visual.global_.offwidth = max(640, view.render_size[0])
    spec.visual.global_.offheight = max(480, view.render_size[1])
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]
    spec.add_texture(
        name="scene_skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    checker1, checker2 = FLOOR_CHECKER_RGB[view.platform]
    spec.add_texture(
        name="scene_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=checker1,
        rgb2=checker2,
        width=300,
        height=300,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.8, 0.8, 0.8],
    )
    material = spec.add_material(name="scene_grid", texrepeat=[5, 5], texuniform=True, reflectance=0.45)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "scene_grid"
    spec.worldbody.add_geom(
        name="scene_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.01],
        pos=[0.0, 0.0, view.floor_z],
        material="scene_grid",
    )
    spec.worldbody.add_light(
        name="scene_fill_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.0, 1.5],
        dir=[0.15, side * 0.1, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.1, 0.1, 0.1],
        castshadow=False,
    )
    spec.worldbody.add_light(
        name="scene_key_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.3, side * 0.6, 1.2],
        dir=[-0.2, -side * 0.5, -1.0],
        diffuse=[0.9, 0.9, 0.9],
        specular=[0.5, 0.5, 0.5],
        castshadow=True,
    )
    target = f"{conventions.HAND_PREFIXES[hand]}middle_finger_proximal"
    for name, pos in (("front", FRONT_CAMERA_POS[hand]), ("side", SIDE_CAMERA_POS), ("back", BACK_CAMERA_POS[hand])):
        spec.worldbody.add_camera(
            name=name, pos=np.add(pos, root).tolist(), mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
            targetbody=target, fovy=45.0,
        )
    spec.worldbody.add_body(name=PREVIEW_TARGET_BODY, pos=list(view.camera_target))
    spec.worldbody.add_camera(
        name=PREVIEW_CAMERA, pos=list(view.camera_position), mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody=PREVIEW_TARGET_BODY, fovy=view.fovy_deg,
    )


def scene_record(hand: str, view: "View") -> dict:
    if hand not in conventions.HAND_PREFIXES:
        raise ValueError(f"hand must be one of {tuple(conventions.HAND_PREFIXES)}, got {hand!r}")
    if view.hand != hand:
        raise ValueError(f"the view frames the {view.hand} hand, not the {hand} hand")
    return {
        "wrist": "fixed",
        "finger_axis": "+z",
        "gravity": list(conventions.GRAVITY),
        "floor_z": view.floor_z,
        "cameras": [*CAMERAS, PREVIEW_CAMERA],
        "preview": view.record(),
    }


def env_grid_offsets(count: int, spacing: float) -> np.ndarray:
    """The (count, 3) env origins Isaac Lab's cloner uses for `count` envs `spacing` m apart (rows along -x, columns
    along +y, centered on the world origin); MuJoCo Warp 05 draws its worlds at the same offsets."""
    if count < 1 or not (math.isfinite(spacing) and spacing > 0.0):
        raise ValueError(f"need count >= 1 and finite spacing > 0, got count={count}, spacing={spacing}")
    rows = math.ceil(count / math.sqrt(count))
    cols = math.ceil(count / rows)
    row, col = np.divmod(np.arange(count), cols)
    return np.stack(
        [-(row - (rows - 1) / 2) * spacing, (col - (cols - 1) / 2) * spacing, np.zeros(count)], axis=1
    )


def pose_vector(joint_names: Sequence[str], pose: str) -> np.ndarray:
    if pose not in PRESET_POSES:
        raise ValueError(f"unknown pose {pose!r}, expected one of {POSES}")
    values = PRESET_POSES[pose]
    return np.array([values[suffix] for suffix in _joint_suffixes(joint_names)], dtype=np.float64)


def _joint_suffixes(joint_names: Sequence[str]) -> list[str]:
    prefixes = sorted({name[:2] for name in joint_names})
    if len(prefixes) != 1 or prefixes[0] not in conventions.HAND_PREFIXES.values():
        raise ValueError(f"joint names must share one hand prefix l_ or r_, got prefixes {prefixes}")
    suffixes = [name[2:] for name in joint_names]
    if sorted(suffixes) != sorted(conventions.JOINT_SUFFIXES):
        missing = sorted(set(conventions.JOINT_SUFFIXES) - set(suffixes))
        unknown = sorted(set(suffixes) - set(conventions.JOINT_SUFFIXES))
        raise ValueError(
            f"joint names must name each of the {len(conventions.JOINT_SUFFIXES)} hand joints once; "
            f"missing {missing}, unknown {unknown}, got {len(suffixes)} names"
        )
    return suffixes


def _min_jerk(q0: np.ndarray, q1: np.ndarray, t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    s = min(max(t / duration, 0.0), 1.0)
    shape = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    rate = 30.0 * s**2 * (1.0 - s) ** 2 / duration
    return q0 + (q1 - q0) * shape, (q1 - q0) * rate


def whole_steps(duration_s: float, timestep: float) -> int:
    ratio = duration_s / timestep
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"duration {duration_s} s is not a whole number of {timestep} s physics steps")
    return steps


def tracking_duration() -> float:
    return (len(TRACKING_SEQUENCE) - 1) * (MOVE_S + HOLD_S)


def tracking_reference(joint_names: Sequence[str], t: float) -> tuple[np.ndarray, np.ndarray]:
    if not t >= 0.0:
        raise ValueError(f"t must be a non-negative time in s, got {t!r}")
    period = MOVE_S + HOLD_S
    segment = min(int(t // period), len(TRACKING_SEQUENCE) - 2)
    start = pose_vector(joint_names, TRACKING_SEQUENCE[segment])
    target = pose_vector(joint_names, TRACKING_SEQUENCE[segment + 1])
    return _min_jerk(start, target, t - segment * period, MOVE_S)


def parallel_duration() -> float:
    return len(TRACKING_SEQUENCE) * (MOVE_S + HOLD_S)


def parallel_reference(joint_names: Sequence[str], world: int, t: float) -> tuple[np.ndarray, np.ndarray]:
    if not t >= 0.0:
        raise ValueError(f"t must be a non-negative time in s, got {t!r}")
    if world < 0:
        raise ValueError(f"world must be a non-negative index, got {world!r}")
    period = MOVE_S + HOLD_S
    count = len(TRACKING_SEQUENCE)
    # The closed sequence never moves fist -> pinch, which would push the thumb into the index finger.
    phase = TRACKING_SEQUENCE.index(POSES[world % len(POSES)]) * period
    shifted = (t + phase) % parallel_duration()
    segment = min(int(shifted // period), count - 1)
    start = pose_vector(joint_names, TRACKING_SEQUENCE[segment])
    target = pose_vector(joint_names, TRACKING_SEQUENCE[(segment + 1) % count])
    return _min_jerk(start, target, shifted - segment * period, MOVE_S)


class JointTableError(ValueError):
    pass


def description_joint_ranges(mjcf: Path) -> dict[str, list[float]]:
    spec = mujoco.MjSpec.from_file(str(mjcf))
    scale = math.pi / 180.0 if spec.compiler.degree else 1.0
    ranges = {}
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if joint.limited == mujoco.mjtLimited.mjLIMITED_FALSE:
            raise JointTableError(f"joint {joint.name!r} in {mjcf} is declared unlimited")
        ranges[joint.name] = [float(value) * scale for value in joint.range]
    return ranges


def check_joint_table(hand: str, description: Mapping[str, Sequence[float]], effective: Mapping[str, Sequence[float]]) -> None:
    if hand not in conventions.HAND_PREFIXES:
        raise ValueError(f"hand must be one of {tuple(conventions.HAND_PREFIXES)}, got {hand!r}")
    expected = [conventions.HAND_PREFIXES[hand] + suffix for suffix in conventions.JOINT_SUFFIXES]
    problems = []
    for label, table in (("description MJCF", description), ("compiled model", effective)):
        if list(table) != expected:
            problems.append(
                f"{label} has {len(table)} hinge joints {list(table)}; expected the {len(expected)} joints {expected}"
            )
    for name in expected:
        if name not in description or name not in effective:
            continue
        lo, hi = (float(value) for value in description[name])
        got = [float(value) for value in effective[name]]
        if not lo < hi:
            problems.append(f"{name}: description range {[lo, hi]} is empty")
        if len(got) != 2 or abs(got[0] - lo) > JOINT_RANGE_ATOL or abs(got[1] - hi) > JOINT_RANGE_ATOL:
            problems.append(f"{name}: compiled range {got} differs from description range {[lo, hi]} by > {JOINT_RANGE_ATOL} rad")
    if problems:
        raise JointTableError("joint table does not match description: " + "; ".join(problems))


def sweep_targets(joint_names: Sequence[str], ranges) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=np.float64)
    if ranges.shape != (len(joint_names), 2) or not np.all(np.isfinite(ranges)) or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError(
            f"ranges must hold one finite [lower, upper] row with lower < upper per joint ({len(joint_names)}), "
            f"got {ranges.tolist()}"
        )
    margin = SWEEP_MARGIN * (ranges[:, 1] - ranges[:, 0])
    inner = np.stack([ranges[:, 0] + margin, ranges[:, 1] - margin], axis=1)
    open_q = pose_vector(joint_names, "open")
    outside = [name for name, q, (lo, hi) in zip(joint_names, open_q, inner) if not lo <= q <= hi]
    if outside:
        raise ValueError(f"the open pose lies outside the inner sweep range of {outside}")
    return inner


def sweep_groups(joint_names: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
    groups: dict[str, list[int]] = {}
    for index, suffix in enumerate(_joint_suffixes(joint_names)):
        if suffix.startswith(SWEEP_THUMB_PREFIX):
            groups[suffix] = [index]
            continue
        finger = next((finger for finger in SWEEP_FINGERS if suffix.startswith(finger + "_")), None)
        if finger is None:
            raise ValueError(f"joint {joint_names[index]!r} is neither a thumb joint nor a joint of {SWEEP_FINGERS}")
        groups.setdefault(f"fingers_{suffix[len(finger) + 1:]}", []).append(index)
    return [(label, tuple(members)) for label, members in groups.items()]


def sweep_window_duration() -> float:
    return (len(SWEEP_WAYPOINTS) - 1) * (SWEEP_MOVE_S + SWEEP_HOLD_S)


def sweep_duration(joint_names: Sequence[str]) -> float:
    return len(sweep_groups(joint_names)) * sweep_window_duration()


def sweep_schedule(joint_names: Sequence[str]) -> list[dict]:
    period = SWEEP_MOVE_S + SWEEP_HOLD_S
    schedule = []
    for window, (label, members) in enumerate(sweep_groups(joint_names)):
        start = window * sweep_window_duration()
        schedule.append(
            {
                "window": window,
                "label": label,
                "joints": [joint_names[i] for i in members],
                "t_s": [start, start + sweep_window_duration()],
                "waypoint_reached_s": [start + k * period + SWEEP_MOVE_S for k in range(len(SWEEP_WAYPOINTS) - 1)],
            }
        )
    return schedule


def sweep_reference(joint_names: Sequence[str], ranges, t: float) -> tuple[np.ndarray, np.ndarray, int]:
    if not t >= 0.0:
        raise ValueError(f"t must be a non-negative time in s, got {t!r}")
    inner = sweep_targets(joint_names, ranges)
    groups = sweep_groups(joint_names)
    q_des = pose_vector(joint_names, "open")
    qd_des = np.zeros_like(q_des)
    moves = len(SWEEP_WAYPOINTS) - 1
    period = SWEEP_MOVE_S + SWEEP_HOLD_S
    segment = min(int(t // period), len(groups) * moves - 1)
    window, move = divmod(segment, moves)
    members = list(groups[window][1])
    points = {"open": q_des[members], "inner_lower": inner[members, 0], "inner_upper": inner[members, 1]}
    q_des[members], qd_des[members] = _min_jerk(
        points[SWEEP_WAYPOINTS[move]], points[SWEEP_WAYPOINTS[move + 1]], t - segment * period, SWEEP_MOVE_S
    )
    return q_des, qd_des, window


def sweep_reference_record(joint_names: Sequence[str]) -> dict:
    return {
        "mode": "one window at a time: each thumb joint alone, then the four fingers' joints of one role together on "
        "one min-jerk time axis, each joint to its own inner limits; joints outside the window hold the open pose "
        "under the same MIT law",
        "grouping": f"thumb joints (suffix {SWEEP_THUMB_PREFIX}*) alone; finger joints <finger>_<role> grouped by "
        f"role over the fingers {list(SWEEP_FINGERS)}; windows ordered by their first joint in model order",
        "waypoints": list(SWEEP_WAYPOINTS),
        "margin_fraction": SWEEP_MARGIN,
        "inner_definition": "[lower + m*(upper - lower), upper - m*(upper - lower)] of the effective range, "
        "m = margin_fraction; the model joint limits are not modified",
        "move_s": SWEEP_MOVE_S,
        "hold_s": SWEEP_HOLD_S,
        "profile": "min-jerk",
        "windows": sweep_schedule(joint_names),
    }


def sweep_joint_table(
    joint_names: Sequence[str],
    description: Mapping[str, Sequence[float]],
    effective: Mapping[str, Sequence[float]],
    active: np.ndarray,
    q: np.ndarray,
    error: np.ndarray,
    saturated: np.ndarray,
) -> list[dict]:
    inner = sweep_targets(joint_names, [effective[name] for name in joint_names])
    schedule = {index: entry for entry, (_, members) in zip(sweep_schedule(joint_names), sweep_groups(joint_names))
                for index in members}
    rows = []
    for j, name in enumerate(joint_names):
        entry = schedule[j]
        window = active == entry["window"]
        if not window.any():
            raise ValueError(f"window {entry['label']} of joint {name} was never active")
        rows.append(
            {
                "joint": name,
                "window": entry["label"],
                "description_range_rad": [float(value) for value in description[name]],
                "effective_range_rad": [float(value) for value in effective[name]],
                "sweep_target_rad": inner[j].tolist(),
                "sweep_window_s": entry["t_s"],
                "waypoint_reached_s": entry["waypoint_reached_s"],
                "reached_min_rad": float(q[window, j].min()),
                "reached_max_rad": float(q[window, j].max()),
                "max_abs_error_active_rad": float(np.abs(error[window, j]).max()),
                "max_abs_error_run_rad": float(np.abs(error[:, j]).max()),
                "saturation_fraction_active": float(saturated[window, j].mean()),
                "saturation_fraction_run": float(saturated[:, j].mean()),
            }
        )
    return rows


def sweep_flags(rows: Sequence[Mapping]) -> list[str]:
    return [
        row["joint"]
        for row in rows
        if row["saturation_fraction_run"] > 0.0 or row["max_abs_error_run_rad"] > SWEEP_ERROR_FLAG_RAD
    ]


def write_sweep_csv(
    path: Path, joint_names: Sequence[str], t: np.ndarray, active: np.ndarray, series: Mapping[str, np.ndarray]
) -> None:
    labels = [label for label, _ in sweep_groups(joint_names)]
    values = np.concatenate([series[field] for field in SWEEP_SERIES], axis=1)
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "active_window", *(f"{field}:{name}" for field in SWEEP_SERIES for name in joint_names)])
        for k in range(len(t)):
            writer.writerow([f"{t[k]:.4f}", labels[active[k]], *(f"{value:.9g}" for value in values[k])])


def sweep_figure(rows: Sequence[Mapping], t: np.ndarray, series: Mapping[str, np.ndarray], title: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    cols = 4
    fig = Figure(figsize=(5.0 * cols, 3.0 * math.ceil(len(rows) / cols)), dpi=80, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(math.ceil(len(rows) / cols), cols, squeeze=False).ravel()
    for j, (ax, row) in enumerate(zip(axes, rows)):
        start, end = row["sweep_window_s"]
        window = (t >= start) & (t < end)
        tw = t[window]
        ax.plot(tw, series["q_des"][window, j], color="gray", linestyle="--", label="q_des")
        ax.plot(tw, series["q"][window, j], color="C0", label="q")
        for k, value in enumerate(row["description_range_rad"]):
            ax.axhline(value, color="red", linewidth=0.8, label="description limit" if k == 0 else None)
        for k, value in enumerate(row["sweep_target_rad"]):
            ax.axhline(value, color="green", linewidth=0.8, linestyle=":", label="inner target" if k == 0 else None)
        ax.set_title(f"{row['joint']} [{row['window']}]", fontsize=10)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("q [rad]")
        ax.grid(True, alpha=0.3)
        tau_ax = ax.twinx()
        tau_ax.plot(tw, series["tau_command"][window, j], color="C1", linewidth=0.8, label="tau_command")
        tau_ax.plot(tw, series["tau_applied"][window, j], color="C3", linewidth=0.8, linestyle="--",
                    label="tau_applied")
        tau_ax.set_ylabel("tau [N*m]")
        if j == 0:
            handles = ax.get_legend_handles_labels()[0] + tau_ax.get_legend_handles_labels()[0]
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(title)
    return fig


class SceneRenderer:
    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480, *, world_frame: bool = False):
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._option = mujoco.MjvOption()
        # The hand model keeps collision meshes in geom group 2 and tip/IMU sites in groups 3-4; show visual meshes only.
        self._option.geomgroup[2] = 0
        self._option.sitegroup[:] = 0
        if world_frame:
            self._option.frame = mujoco.mjtFrame.mjFRAME_WORLD

    @property
    def option(self) -> mujoco.MjvOption:
        return self._option

    def render(self, data: mujoco.MjData, camera: str,
               extra: Callable[[mujoco.MjvScene], None] | None = None) -> np.ndarray:
        self._renderer.update_scene(data, camera=camera, scene_option=self._option)
        if extra is not None:
            extra(self._renderer.scene)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "SceneRenderer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _require_display() -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise conventions.ViewerError(
            f"no DISPLAY or WAYLAND_DISPLAY is set, so a MuJoCo viewer window cannot open; {conventions.headless_rerun_hint()}"
        )


class Display:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, view: "View", *, headless: bool):
        self._viewer = None
        if not headless:
            _require_display()
            # Imported only here: the passive viewer needs a desktop display that headless runs do not have.
            from mujoco import viewer

            try:
                # Computed before launch so a degenerate View fails without leaving a window open.
                camera_target = view.camera_target
                distance, azimuth, elevation = view.distance_m, view.azimuth_deg, view.elevation_deg
                # Side panels stay on for model/world info; Tab and Shift+Tab hide them.
                self._viewer = viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)
                with self._viewer.lock():
                    # As in SceneRenderer: by default the viewer draws the collision meshes (group 2) over the visual ones.
                    self._viewer.opt.geomgroup[2] = 0
                    self._viewer.opt.sitegroup[:] = 0
                    self._viewer.cam.lookat[:] = camera_target
                    self._viewer.cam.distance = distance
                    self._viewer.cam.azimuth = azimuth
                    self._viewer.cam.elevation = elevation
            except Exception as exc:
                self.close()
                raise conventions.ViewerError(f"could not open a MuJoCo viewer window: {exc}; {conventions.headless_rerun_hint()}") from exc

    def sync(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self) -> "Display":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def user_scn(self) -> mujoco.MjvScene:
        """The viewer's overlay scene, for a demo that injects extra geoms (e.g. other Warp worlds); needs an
        open viewer (headless=False)."""
        if self._viewer is None:
            raise RuntimeError("Display.user_scn needs an open viewer (headless=False)")
        return self._viewer.user_scn


def run_live(display: Display, dt: float, frame: Callable[[float], None]) -> None:
    """Run a live (non-headless) demo in real time: call `frame(t)` once per `dt` seconds of elapsed demo time,
    sync `display`, and pace to wall-clock real time, until the user closes the viewer window."""
    t = 0.0
    while display.is_running():
        started = time.perf_counter()
        frame(t)
        display.sync()
        time.sleep(max(0.0, dt - (time.perf_counter() - started)))
        t += dt
