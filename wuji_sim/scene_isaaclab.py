# SPDX-License-Identifier: Apache-2.0
import math
from collections.abc import Mapping, Sequence

from wuji_sim import conventions, framing, profiles
from wuji_sim.platforms import isaaclab as platform

# Appearance only, not calibrated to MuJoCo; one stage renders all cameras, so a sky DomeLight replaces a headlight.
SCENE_SKY_RGB = (0.4, 0.475, 0.575)
SCENE_LIGHT_INTENSITY = 837.5
SCENE_LIGHTS = {
    "scene_key_light": {"dir": (0.0, 0.0, -1.0), "diffuse": 0.7, "specular": 0.3, "castshadow": True},
    "scene_fill_light": {"dir": (0.3, 0.3, -1.0), "diffuse": 0.3, "specular": 0.0, "castshadow": False},
}
# GroundPlaneCfg's checker material; a texture scale of 3 draws its 1 m checks a third as wide.
GROUND_SHADER_PRIM = "/World/ground/Looks/theGrid/Shader"
GROUND_TEXTURE_SCALE = (3.0, 3.0)
RTX_SETTINGS = {
    "/rtx/reflections/enabled": True,
    "/rtx/ambientOcclusion/enabled": True,
    "/rtx/shadows/enabled": True,
    "/rtx/directLighting/enabled": True,
    "/rtx/indirectDiffuse/enabled": True,
}
PATH_TRACING_SPP = 96
# One path-traced frame per capture: five differ by < 0.01 grey level on average, each at a full render's cost.
CAPTURE_RENDERS = 1
# PinholeCameraCfg's default; the field of view follows from its ratio to the focal length.
CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_CLIPPING_RANGE = (0.01, 10.0)


def _distant_light_rotation(direction: Sequence[float]) -> tuple[float, float, float, float]:
    # A UsdLux DistantLight emits along its local -Z axis: shortest-arc (x, y, z, w) rotation of -Z onto direction.
    norm = math.sqrt(sum(value * value for value in direction))
    x, y, z = (value / norm for value in direction)
    scale = math.sqrt(2.0 * (1.0 - z))
    return (y / scale, -x / scale, 0.0, (1.0 - z) / scale)


def _set_scene_light_responses(light_cfgs: Mapping[str, object]) -> None:
    from isaaclab.sim.utils.prims import safe_set_attribute_on_usd_schema
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = get_current_stage()
    for name, cfg in light_cfgs.items():
        light = SCENE_LIGHTS[name]
        prim = stage.GetPrimAtPath(cfg.prim_path)
        # Not in LightCfg; the schema setter raises on inputs UsdLux lacks, a prim setter authors ones RTX ignores.
        light_api = UsdLux.LightAPI(prim)
        shadow_api = UsdLux.ShadowAPI.Apply(prim)
        safe_set_attribute_on_usd_schema(light_api, "diffuse", light["diffuse"], camel_case=True)
        safe_set_attribute_on_usd_schema(light_api, "specular", light["specular"], camel_case=True)
        safe_set_attribute_on_usd_schema(shadow_api, "shadow_enable", light["castshadow"], camel_case=True)
        diffuse = light_api.GetDiffuseAttr().Get()
        specular = light_api.GetSpecularAttr().Get()
        shadow = shadow_api.GetShadowEnableAttr().Get()
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        emitted = transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
        if (
            not math.isclose(diffuse, light["diffuse"], abs_tol=1e-5)
            or not math.isclose(specular, light["specular"], abs_tol=1e-5)
            or shadow != light["castshadow"]
            or (emitted - Gf.Vec3d(*light["dir"]).GetNormalized()).GetLength() > 1e-5
        ):
            raise profiles.ProfileError(
                f"{cfg.prim_path}: UsdLux readback diffuse={diffuse!r} specular={specular!r} shadow:enable={shadow!r} "
                f"direction={tuple(emitted)!r} does not match {name} {light!r}"
            )


def _set_ground_texture_scale() -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf

    shader = get_current_stage().GetPrimAtPath(GROUND_SHADER_PRIM)
    attribute = shader.GetAttribute("inputs:texture_scale") if shader.IsValid() else None
    if attribute is None or not attribute.IsValid():
        raise profiles.ProfileError(f"{GROUND_SHADER_PRIM}: no inputs:texture_scale to set; GroundPlaneCfg's material changed")
    attribute.Set(Gf.Vec2f(*GROUND_TEXTURE_SCALE))
    readback = tuple(float(value) for value in attribute.Get())
    if readback != GROUND_TEXTURE_SCALE:
        raise profiles.ProfileError(
            f"{GROUND_SHADER_PRIM}: inputs:texture_scale reads back {readback}, not {GROUND_TEXTURE_SCALE}"
        )


def _apply_rtx_settings() -> None:
    import carb.settings

    settings = carb.settings.get_settings()
    for path, value in RTX_SETTINGS.items():
        settings.set_bool(path, value)
    readback = {path: settings.get(path) for path in RTX_SETTINGS}
    if readback != RTX_SETTINGS:
        raise profiles.ProfileError(f"RTX settings read back {readback}, not {RTX_SETTINGS}")


def _hide_wrist_mounts(num_envs: int) -> None:
    # A test-bench fixture, hidden as wuji_sim.scene.add_scene hides it in MuJoCo; its collision shapes stay.
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdGeom

    mounts = [
        prim for prim in get_current_stage().Traverse()
        if prim.GetPath().pathString.startswith("/World/envs/") and prim.GetName().endswith("_mount")
    ]
    if len(mounts) < num_envs:
        raise profiles.ProfileError(f"found {len(mounts)} wrist mount prim(s) for {num_envs} env(s)")
    for prim in mounts:
        UsdGeom.Imageable(prim).MakeInvisible()


def _check_fovy(fovy_deg: float) -> None:
    if not (profiles.is_number(fovy_deg) and 0.0 < fovy_deg < 180.0):
        raise ValueError(f"fovy_deg must be a vertical field of view in (0, 180) deg, got {fovy_deg!r}")


def _focal_length(aperture: float, width: int, height: int, fovy_deg: float) -> float:
    return 0.5 * aperture * height / width / math.tan(math.radians(fovy_deg) / 2.0)


def add_camera(loaded: platform.Loaded, name: str, *, width: int, height: int, fovy_deg: float):
    from isaaclab import sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    if not (profiles.is_count(width) and profiles.is_count(height)):
        raise ValueError(f"camera {name!r} needs a positive int width and height, got {width!r} x {height!r}")
    _check_fovy(fovy_deg)
    # One camera prim per env, all at one pose; scene partitioning is off so each camera sees every env.
    prim_path = f"/World/envs/env_[^/]+/{name}"
    camera = Camera(CameraCfg(
        prim_path=prim_path,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_focal_length(CAMERA_HORIZONTAL_APERTURE, width, height, fovy_deg),
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
    ))
    count = len(sim_utils.find_matching_prims(prim_path))
    if count != loaded.articulation.num_instances:
        raise profiles.ProfileError(f"camera {name!r} has {count} prim(s) for {loaded.articulation.num_instances} env(s)")
    return camera


def aim_camera(loaded: platform.Loaded, camera, eye: Sequence[float], target: Sequence[float]) -> None:
    import torch

    count = camera.num_instances
    eyes = torch.tensor([[float(value) for value in eye]] * count, dtype=torch.float32, device=loaded.device)
    targets = torch.tensor([[float(value) for value in target]] * count, dtype=torch.float32, device=loaded.device)
    camera.set_world_poses_from_view(eyes, targets)


def capture_rgb(loaded: platform.Loaded, camera, *, renders: int):
    import torch

    if not profiles.is_count(renders):
        raise ValueError(f"renders must be a positive int, got {renders!r}")
    # Headless, sim.render() alone does not run RTX; reading the camera does, once per new render generation.
    for _ in range(renders):
        loaded.sim.render()
        camera.update(0.0, force_recompute=True)
    rgb = camera.data.output["rgb"].torch[0]
    if rgb.numel() == 0 or rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError(f"camera returned no uint8 RGB image: {rgb.dtype} {tuple(rgb.shape)}")
    return rgb[..., :3].detach().to("cpu").contiguous()


def use_path_tracing(spp: int = PATH_TRACING_SPP) -> None:
    import carb.settings

    if not profiles.is_count(spp):
        raise ValueError(f"spp must be a positive int, got {spp!r}")
    settings = carb.settings.get_settings()
    settings.set_string("/rtx/rendermode", "PathTracing")
    settings.set_int("/rtx/pathtracing/spp", spp)
    readback = (settings.get("/rtx/rendermode"), settings.get("/rtx/pathtracing/spp"))
    if readback != ("PathTracing", spp):
        raise profiles.ProfileError(f"RTX render mode and samples per pixel read back {readback}, not ('PathTracing', {spp})")


def _set_viewport_camera(loaded: platform.Loaded, eye: Sequence[float], target: Sequence[float], fovy_deg: float) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Usd, UsdGeom

    _check_fovy(fovy_deg)
    loaded.sim.set_camera_view(tuple(float(value) for value in eye), tuple(float(value) for value in target))
    viewport = get_active_viewport()
    resolution = None if viewport is None else viewport.resolution
    if not resolution or not all(resolution):
        raise conventions.ViewerError(f"the Isaac Sim window has no viewport resolution to fit the view to: {resolution!r}")
    stage = get_current_stage()
    camera = UsdGeom.Camera(stage.GetPrimAtPath(str(viewport.camera_path)))
    if not camera:
        raise conventions.ViewerError(f"the Isaac Sim window's viewport camera {viewport.camera_path} is not a camera prim")
    # The viewport keeps pixels square; Kit authors its camera in the session layer, which outranks the root layer.
    width, height = resolution
    focal_length = _focal_length(camera.GetHorizontalApertureAttr().Get(), width, height, fovy_deg)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera.GetFocalLengthAttr().Set(focal_length)
    if not math.isclose(camera.GetFocalLengthAttr().Get(), focal_length, rel_tol=1e-6):
        raise conventions.ViewerError(f"the viewport camera {viewport.camera_path} kept its focal length instead of {focal_length}")


class Scene:
    """The preview look, passed to the Isaac Lab platform's load(scene=...): assets() gives the sky and lights to
    spawn with the scene, finish() completes the look once they are on the stage."""

    def assets(self) -> dict:
        from isaaclab import sim as sim_utils
        from isaaclab.assets import AssetBaseCfg

        sky_cfg = AssetBaseCfg(
            prim_path="/World/scene_skybox",
            spawn=sim_utils.DomeLightCfg(color=SCENE_SKY_RGB, intensity=SCENE_LIGHT_INTENSITY),
        )
        self._light_cfgs = {
            name: AssetBaseCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.DistantLightCfg(intensity=SCENE_LIGHT_INTENSITY),
                init_state=AssetBaseCfg.InitialStateCfg(rot=_distant_light_rotation(light["dir"])),
            )
            for name, light in SCENE_LIGHTS.items()
        }
        return {"scene_skybox": sky_cfg, **self._light_cfgs}

    def finish(self, num_envs: int) -> None:
        _set_scene_light_responses(self._light_cfgs)
        _set_ground_texture_scale()
        _hide_wrist_mounts(num_envs)
        _apply_rtx_settings()


class Display:
    """The Isaac Sim window on `view`, for scene.run_live; platform.launch(headless=False) owns the window."""

    def __init__(self, loaded: platform.Loaded, view: framing.View):
        self._sim = loaded.sim
        _set_viewport_camera(loaded, view.camera_position, view.camera_target, view.fovy_deg)

    def sync(self) -> None:
        self._sim.render()

    def is_running(self) -> bool:
        return self._sim.is_running()
