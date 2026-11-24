# SPDX-License-Identifier: Apache-2.0
"""MotrixSim rendering glue shared by the motrixsim demos.

MotrixSim has one system camera per RenderApp, so the mujoco scene cameras (front, side, back,
preview) are reproduced through `renderer.system_camera.set_view` with orbit parameters derived
from the same numbers that place the mujoco cameras: the framing.View for the preview camera, the
scene.FRONT/SIDE/BACK_CAMERA_POS plus the tracked target body for the fixed cameras. A camera
whose fovy differs from the fovy the model was built with is framed by scaling the orbit distance
by the ratio of the half-fovy tangents, which keeps the same vertical extent around the target;
only the perspective differs slightly.
"""
import math
import os
import time
from collections.abc import Callable

import numpy as np
from motrixsim.render import RenderApp

from wuji_sim import conventions, framing, scene

FIXED_CAMERA_FOVY_DEG = 45.0
CAPTURE_ATTEMPTS = 8
WARMUP_SYNCS = 3
WARMUP_CAPTURES = 2


def orbit(position, target) -> tuple[float, float, float]:
    """Distance, azimuth and elevation (deg) that place the camera at `position` looking at `target`, the
    motrixsim.render set_view convention (azimuth 0 looks along +x; negative elevation is camera up)."""
    position, target = np.asarray(position, dtype=float), np.asarray(target, dtype=float)
    offset = position - target
    distance = float(np.linalg.norm(offset))
    if distance < 1e-9:
        raise ValueError(f"camera position {position.tolist()} coincides with its target {target.tolist()}")
    azimuth = math.degrees(math.atan2(-offset[1], -offset[0]))
    elevation = math.degrees(math.asin(np.clip(-offset[2] / distance, -1.0, 1.0)))
    return distance, azimuth, elevation


def _distance_scale(built_fovy_deg: float, camera_fovy_deg: float) -> float:
    return math.tan(math.radians(built_fovy_deg) * 0.5) / math.tan(math.radians(camera_fovy_deg) * 0.5)


def set_view_camera(app: RenderApp, camera: str, hand: str, view: framing.View, built_fovy_deg: float,
                    target=None) -> None:
    """Point the renderer's system camera at one of the scene cameras.

    `target` overrides the look-at point of a fixed camera (the mujoco versions track the middle-finger
    proximal body); the preview camera always looks at view.camera_target."""
    if camera == scene.PREVIEW_CAMERA:
        position, lookat, fovy = view.camera_position, np.asarray(view.camera_target, dtype=float), view.fovy_deg
    else:
        positions = {"front": scene.FRONT_CAMERA_POS[hand], "side": scene.SIDE_CAMERA_POS,
                     "back": scene.BACK_CAMERA_POS[hand]}
        if camera not in positions:
            raise ValueError(f"unknown camera {camera!r}, expected one of {(*positions, scene.PREVIEW_CAMERA)}")
        if target is None:
            raise ValueError(f"the fixed camera {camera!r} needs its tracked target body position")
        position = np.add(positions[camera], view.hand_roots[0])
        lookat, fovy = np.asarray(target, dtype=float), FIXED_CAMERA_FOVY_DEG
    distance, azimuth, elevation = orbit(position, lookat)
    app.system_camera.set_view(
        lookat=[float(value) for value in lookat],
        distance=distance * _distance_scale(built_fovy_deg, fovy),
        elevation=elevation,
        azimuth=azimuth,
    )


def capture_rgb(app: RenderApp, data) -> np.ndarray:
    """Capture the system camera as an HxWx3 uint8 array; the task completes on a sync(wait=True)."""
    task = app.system_camera.capture()
    image = None
    for _ in range(CAPTURE_ATTEMPTS):
        app.sync(data, wait=True)
        image = task.take_image()
        if image is not None:
            break
    if image is None:
        raise RuntimeError(f"the system camera capture did not return an image after {CAPTURE_ATTEMPTS} syncs")
    pixels = np.asarray(image.pixels)
    if pixels.ndim == 4:  # A batched capture returns one tile per world; the scene view is tile 0.
        pixels = pixels[0]
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise RuntimeError(f"unexpected capture shape {pixels.shape}")
    return np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)


def render_camera(app: RenderApp, data, camera: str, hand: str, view: framing.View, built_fovy_deg: float,
                  target=None) -> np.ndarray:
    set_view_camera(app, camera, hand, view, built_fovy_deg, target)
    return capture_rgb(app, data)


def hide_fixture_groups(app: RenderApp) -> None:
    # The hand model keeps its collision meshes and the wrist mount in geom group 2, as in the mujoco views.
    app.opt.set_group_vis(2, False)


class OffscreenRenderer:
    """A headless RenderApp bound to one loaded model; `render` returns one scene camera as RGB."""

    def __init__(self, loaded, view: framing.View, hand: str, built_fovy_deg: float):
        self.loaded = loaded
        self.view = view
        self.hand = hand
        self.built_fovy_deg = built_fovy_deg
        self.app: RenderApp | None = None

    def render(self, camera: str, target=None) -> np.ndarray:
        return render_camera(self.app, self.loaded.scene_data, camera, self.hand, self.view, self.built_fovy_deg,
                             target)

    def sync(self) -> None:
        self.app.sync(self.loaded.scene_data, wait=True)

    def __enter__(self) -> "OffscreenRenderer":
        self.app = RenderApp(headless=True)
        self.app.launch(self.loaded.scene_model)
        hide_fixture_groups(self.app)
        # Warm the renderer up: the first frames after launch still load assets, and a capture raced
        # against that returns a partial frame (as in MotrixSim's batch-capture example).
        for _ in range(WARMUP_SYNCS):
            self.app.sync(self.loaded.scene_data, wait=True)
        for _ in range(WARMUP_CAPTURES):
            capture_rgb(self.app, self.loaded.scene_data)
        return self

    def __exit__(self, *exc_info) -> None:
        if self.app is not None:
            self.app.close()
            self.app = None


class LiveDisplay:
    """A windowed RenderApp, the motrixsim equivalent of scene.Display: set_view from the framing.View,
    sync after every frame, run until the user closes the window."""

    def __init__(self, loaded, view: framing.View, built_fovy_deg: float = None, *, batch: int = 1,
                 render_offset=None):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise conventions.ViewerError(
                f"no DISPLAY or WAYLAND_DISPLAY is set, so a MotrixSim viewer window cannot open; "
                f"{conventions.headless_rerun_hint()}"
            )
        self.loaded = loaded
        self.app = RenderApp(headless=False)
        self.app.launch(loaded.scene_model, batch=batch, render_offset=render_offset)
        hide_fixture_groups(self.app)
        fovy = view.fovy_deg if built_fovy_deg is None else built_fovy_deg
        distance, azimuth, elevation = orbit(view.camera_position, view.camera_target)
        self.app.system_camera.set_view(
            lookat=[float(value) for value in view.camera_target],
            distance=distance * _distance_scale(fovy, view.fovy_deg),
            elevation=elevation,
            azimuth=azimuth,
        )

    def sync(self) -> None:
        if self.app is not None:
            self.app.sync(self.loaded.scene_data)

    def is_running(self) -> bool:
        return self.app is not None and not self.app.is_closed

    def close(self) -> None:
        if self.app is not None:
            self.app.close()
            self.app = None

    def __enter__(self) -> "LiveDisplay":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def run_live(display: LiveDisplay, dt: float, frame: Callable[[float], None]) -> None:
    """Run a live (non-headless) demo in real time: call `frame(t)` once per `dt` seconds of elapsed demo
    time, sync `display`, and pace to wall-clock real time, until the user closes the viewer window."""
    t = 0.0
    while display.is_running():
        started = time.perf_counter()
        frame(t)
        display.sync()
        time.sleep(max(0.0, dt - (time.perf_counter() - started)))
        t += dt
