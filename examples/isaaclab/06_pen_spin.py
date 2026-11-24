# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene, scene_isaaclab
from wuji_sim.platforms import isaaclab as platform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

import pen_spin  # noqa: E402

DEMO = "06_pen_spin"
RENDERING = "Isaac Lab physics / Isaac Sim RTX camera rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"
PEN_PRIM = "/World/envs/env_0/Pen"
WRIST_TOLERANCE_M = 1e-4
WRIST_TOLERANCE_DEG = 1e-2
# Image (column, row) axes per box face in PEN_END_TEXTURES order (+x, -x, +y, -y, +z, -z), as MuJoCo maps a cube.
CUBE_FACE_AXES = (((0, 0, -1), (0, -1, 0)), ((0, 0, 1), (0, -1, 0)), ((1, 0, 0), (0, 0, 1)),
                  ((1, 0, 0), (0, 0, -1)), ((1, 0, 0), (0, -1, 0)), ((-1, 0, 0), (0, -1, 0)))
# (column, row) corner signs in counterclockwise order seen from outside the face.
FACE_CORNERS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def steps_per_frame(period: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * period)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {period} s control steps")
    return steps


def isaac_pose(pose) -> list[float]:
    x, y, z, qw, qx, qy, qz = (float(value) for value in pose)
    return [x, y, z, qx, qy, qz, qw]


def wuji_pose(values) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = (float(value) for value in values)
    return np.array([x, y, z, qw, qx, qy, qz])


def label_material(stage, path: str, image):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, f"{path}/image")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(image))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface = UsdShade.Shader.Define(stage, f"{path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def add_label_block(stage, path: str, center: float, textures, materials) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    points, st = [], []
    for column, row in CUBE_FACE_AXES:
        column, row = np.array(column), np.array(row)
        normal = np.cross(row, column)
        for sign_column, sign_row in FACE_CORNERS:
            points.append(Gf.Vec3f(*(normal + sign_column * column + sign_row * row).tolist()))
            st.append(Gf.Vec2f((sign_column + 1) / 2, (1 - sign_row) / 2))
    block = UsdGeom.Mesh.Define(stage, path)
    block.CreatePointsAttr(points)
    block.CreateFaceVertexCountsAttr([4] * len(CUBE_FACE_AXES))
    block.CreateFaceVertexIndicesAttr(list(range(len(points))))
    block.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(block).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray(st))
    block.AddTranslateOp().Set(Gf.Vec3d(center, 0.0, 0.0))
    qw, qx, qy, qz = pen_spin.PEN_END_QUAT
    block.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
    block.AddScaleOp().Set(Gf.Vec3f(*pen_spin.PEN_END_HALF_M))
    for face, image in enumerate(textures):
        subset = UsdGeom.Subset.CreateGeomSubset(block, f"face_{face}", UsdGeom.Tokens.face, [face],
                                                 UsdShade.Tokens.materialBind)
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(materials[image])


class PenScene(scene_isaaclab.Scene):
    def __init__(self, pen_pose_w):
        self._pen_pose_w = pen_pose_w
        self.pen = None

    def finish(self, num_envs: int) -> None:
        super().finish(num_envs)
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom, UsdPhysics

        shape = pen_spin.pen_shape()
        stage = get_current_stage()
        white = [Gf.Vec3f(1.0, 1.0, 1.0)]
        x, y, z, qx, qy, qz, qw = isaac_pose(self._pen_pose_w)
        body = UsdGeom.Xform.Define(stage, PEN_PRIM)
        body.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(pen_spin.PEN_MASS_KG)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*shape["inertia"]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf.GetIdentity())

        shaft = UsdGeom.Cylinder.Define(stage, f"{PEN_PRIM}/shaft")
        shaft.CreateAxisAttr(UsdGeom.Tokens.x)
        shaft.CreateHeightAttr(2 * shape["shaft_half"])
        shaft.CreateRadiusAttr(shape["radius"])
        shaft.CreateDisplayColorAttr(white)

        sides = pen_spin.PEN_SHAFT_SIDES
        collision = UsdGeom.Mesh.Define(stage, f"{PEN_PRIM}/shaft_collision")
        collision.CreatePointsAttr([Gf.Vec3f(*vertex) for vertex in shape["shaft_vertices"]])
        collision.CreateFaceVertexCountsAttr([4] * sides + [sides, sides])
        collision.CreateFaceVertexIndicesAttr(
            [index for k in range(sides) for index in (k, (k + 1) % sides, sides + (k + 1) % sides, sides + k)]
            + list(range(2 * sides)))
        collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

        materials = {image: label_material(stage, f"{PEN_PRIM}/looks/{image.removesuffix('.png')}",
                                           pen_spin.PEN_TEXTURE_DIR / image)
                     for image in sorted({image for images in pen_spin.PEN_END_TEXTURES.values() for image in images})}
        half = pen_spin.PEN_END_HALF_M
        for name, center in (("end_a", -shape["end_center"]), ("end_b", shape["end_center"])):
            add_label_block(stage, f"{PEN_PRIM}/{name}", center, pen_spin.PEN_END_TEXTURES[f"pen_{name}"], materials)
            block = UsdGeom.Cube.Define(stage, f"{PEN_PRIM}/{name}_collision")
            block.CreateSizeAttr(1.0)
            block.AddTranslateOp().Set(Gf.Vec3d(center, 0.0, 0.0))
            block.AddScaleOp().Set(Gf.Vec3f(2 * half[2], 2 * half[0], 2 * half[1]))
            block.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(block.GetPrim())

        self.pen = RigidObject(RigidObjectCfg(prim_path=PEN_PRIM, spawn=None))


def place_wrist(loaded: platform.Loaded, wrist: np.ndarray) -> None:
    import torch

    articulation = loaded.articulation
    name = conventions.HAND_PREFIXES[pen_spin.HAND] + "wrist"
    body_ids, body_names = articulation.find_bodies([name], preserve_order=True)
    if tuple(body_names) != (name,):
        raise pen_spin.PenSpinError(f"the hand model has no body {name!r}: {body_names!r}")

    def poses() -> tuple[np.ndarray, np.ndarray]:
        data = articulation.data
        return (wuji_pose(data.root_link_pose_w.torch[0].tolist()),
                wuji_pose(data.body_link_pose_w.torch[0, int(body_ids[0])].tolist()))

    root, current = poses()
    target = pen_spin.to_world(pen_spin.to_wrist_local(root, current), wrist)
    articulation.write_root_pose_to_sim_index(
        root_pose=torch.tensor([isaac_pose(target)], dtype=torch.float32, device=loaded.device))
    loaded.sim.step(render=False)
    articulation.update(loaded.dt)
    current = poses()[1]
    offset = float(np.linalg.norm(current[:3] - wrist[:3]))
    angle = float(pen_spin.rotation_deg(current[3:], wrist[3:]))
    if offset > WRIST_TOLERANCE_M or angle > WRIST_TOLERANCE_DEG:
        raise pen_spin.PenSpinError(
            f"the wrist is {offset:.2e} m and {angle:.2e} deg from the pose the policy was trained with")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a published pen-spinning policy on the right hand under MIT torque control on Isaac Lab."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    pen_spin.add_cli_args(parser)
    args = parser.parse_args(argv[1:])
    if args.hand != pen_spin.HAND:
        parser.error(f"the published policy drives the {pen_spin.HAND} hand; rerun with --hand {pen_spin.HAND}")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    task = pen_spin.load(args.group, args.clip, args.cache_dir)
    print(task.describe())
    reference = task.reference
    policy = pen_spin.Policy(task.model)
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "pen_spin")
    wrist = pen_spin.wrist_pose(view.hand_roots[0])
    with platform.launch(headless=args.headless, enable_cameras=True):
        try:
            import torch

            pen_scene = PenScene(pen_spin.to_world(reference.pen_pose(0), wrist))
            loaded = platform.load(asset.usd, hand=args.hand, description_version=asset.expected_version,
                                   floor_z=view.floor_z, scene=pen_scene)
            pen = pen_scene.pen
            joint_names = loaded.joint_names
            pen_spin.check_joint_order(joint_names, args.hand)
            timestep = float(loaded.dt)
            substeps = scene.whole_steps(pen_spin.CONTROL_DT, timestep)
            if args.headless:
                width, height = view.render_size
                camera = scene_isaaclab.add_camera(loaded, VIDEO_CAMERA, width=width, height=height,
                                                   fovy_deg=view.fovy_deg)
            loaded.sim.reset()

            kp, kd = pen_spin.gains()
            qd_des = np.zeros(len(joint_names))
            tau_ff = np.zeros(len(joint_names))
            stability = platform.mit_stability(loaded, kp, kd)
            platform.set_mit_gains(loaded, kp, kd)
            print(f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
                  f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})")
            place_wrist(loaded, wrist)
            residual = pen_spin.ResidualAction(task.config)
            last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

            control = {
                "law": CONTROL_LAW,
                "kp": kp.tolist(),
                "kd": kd.tolist(),
                "qd_des": 0.0,
                "tau_ff": 0.0,
                "gains_source": "the gains the policy was trained with",
                "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m",
                          "control_period": "s", "physics_timestep": "s", "q": "rad", "qd": "rad/s"},
                "control_period": pen_spin.CONTROL_DT,
                "physics_timestep": timestep,
                "physics_steps_per_control_step": substeps,
                "joint_order": list(joint_names),
                "actuator_ids": list(loaded.joint_ids),
                "tau_limit": list(loaded.tau_limit),
                "stability": stability,
            }

            def pen_pose() -> np.ndarray:
                return wuji_pose(pen.data.root_link_pose_w.torch[0].tolist())

            def reset(step: int) -> None:
                nonlocal last_action
                pen.write_root_pose_to_sim_index(root_pose=torch.tensor(
                    [isaac_pose(pen_spin.to_world(reference.pen_pose(step), wrist))], dtype=torch.float32,
                    device=loaded.device))
                pen.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=loaded.device))
                platform.set_joint_state(loaded, reference.joint_pos(step), qd_des)
                residual.reset()
                last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

            def advance(step: int) -> None:
                nonlocal last_action
                if step > 0 and reference.restart(step):
                    reset(step)
                q, qd = platform.joint_state(loaded)
                pen_local = pen_spin.to_wrist_local(pen_pose(), wrist)
                obs = pen_spin.observation(reference.command(step), q, qd, pen_local, last_action)
                last_action = policy(obs)
                q_des = residual.step(last_action, reference.joint_pos(step))
                platform.set_command(loaded, q_des, qd_des, tau_ff)
                for _ in range(substeps):
                    platform.step(loaded)
                    pen.update(timestep)

            reset(0)
            scene_results = pen_spin.scene_record(args.hand, view, wrist)
            pen_results = {**pen_spin.pen_record(), "geometry": (
                "a round white shaft with a block on each end that shows the marker labels of the real pen, built "
                f"in code; the shaft collides as a {pen_spin.PEN_SHAFT_SIDES}-sided prism mesh, cooked as a convex "
                "hull; mass and inertia are the training setup's values with the mass spread by volume")}
            if args.headless:
                frame_steps = steps_per_frame(pen_spin.CONTROL_DT)
                scene_isaaclab.aim_camera(loaded, camera, view.camera_position, view.camera_target)
                scene_isaaclab.use_path_tracing()

                def capture() -> np.ndarray:
                    return scene_isaaclab.capture_rgb(loaded, camera, renders=scene_isaaclab.CAPTURE_RENDERS).numpy()

                log = pen_spin.PenLog(reference, wrist)
                log.start(pen_pose())
                with run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video:
                    video.append(capture())
                    for step in range(reference.num_steps):
                        advance(step)
                        log.update(step, pen_pose())
                        if (step + 1) % frame_steps == 0:
                            video.append(capture())
                    frames = video.frames
                    video_path = video.path

                replay = {"steps": reference.num_steps, "duration_s": reference.num_steps * pen_spin.CONTROL_DT,
                          **log.record()}
                results = {
                    "scene": scene_results,
                    "policy": {**task.record(), "inference": policy.record()},
                    "control": control,
                    "pen": pen_results,
                    "replay": replay,
                    "video": {"path": str(video_path), "camera": VIDEO_CAMERA, "fps": VIDEO_FPS, "frames": frames,
                              "control_steps_per_frame": frame_steps},
                }
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                        rendering=RENDERING)
                print(f"replay: {pen_spin.describe_replay(replay)}")
                print(f"record: {record_path}")
                print("artifacts: 1")
            else:
                results = {"scene": scene_results, "policy": {**task.record(), "inference": policy.record()},
                           "control": control, "pen": pen_results}
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                        rendering=RENDERING)
                print(f"record: {record_path}")
                print("artifacts: 0")
                display = scene_isaaclab.Display(loaded, view)

                def frame(t: float) -> None:
                    advance(round(t / pen_spin.CONTROL_DT))

                scene.run_live(display, pen_spin.CONTROL_DT, frame)
        except (assets.AssetError, pen_spin.PenSpinError, profiles.ProfileError, mit.MitStabilityError,
                records.RecordError, conventions.ViewerError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, pen_spin.PenSpinError, profiles.ProfileError, mit.MitStabilityError,
            records.RecordError, conventions.ViewerError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
