# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
import math
import sys
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from wuji_sim import conventions, framing, scene

HANDS = ("left", "right")
PREFIX = {"left": "l_", "right": "r_"}
FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
RIGHT_NAMES = tuple("r_" + suffix for suffix in conventions.JOINT_SUFFIXES)


@pytest.fixture(params=HANDS)
def hand(request):
    return request.param


def description_spec(description_root, hand):
    return mujoco.MjSpec.from_file(str(description_root / "mjcf" / f"{hand}.xml"))


def scene_view(description_root, hand, platform="mujoco", kind="tracking", worlds=1):
    return framing.view(description_root / "mjcf" / f"{hand}.xml", hand, platform, kind, worlds=worlds)


def scene_model(description_root, hand, view=None):
    spec = description_spec(description_root, hand)
    scene.add_scene(spec, hand, view or scene_view(description_root, hand))
    return spec.compile()


def hinge_names(model):
    return [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def set_joints(model, data, q):
    names = hinge_names(model)
    data.qpos[[model.jnt_qposadr[model.joint(n).id] for n in names]] = q
    mujoco.mj_forward(model, data)


def posed_data(model, pose):
    data = mujoco.MjData(model)
    set_joints(model, data, scene.pose_vector(hinge_names(model), pose))
    return data


def contact_pairs(model, data):
    return [
        (model.body(int(model.geom_bodyid[c.geom1])).name, model.body(int(model.geom_bodyid[c.geom2])).name, c.dist)
        for c in data.contact[: data.ncon]
    ]


def test_joint_suffixes_follow_description_order(description_root, hand):
    model = description_spec(description_root, hand).compile()
    assert tuple(PREFIX[hand] + s for s in conventions.JOINT_SUFFIXES) == tuple(hinge_names(model))


def test_presets_are_the_four_poses():
    assert scene.POSES == ("open", "half_grip", "fist", "pinch")
    assert set(scene.PRESET_POSES) == set(scene.POSES)


def test_half_grip_is_midway_between_open_and_fist():
    open_q, fist_q = (scene.pose_vector(RIGHT_NAMES, pose) for pose in ("open", "fist"))
    np.testing.assert_allclose(scene.pose_vector(RIGHT_NAMES, "half_grip"), (open_q + fist_q) / 2, atol=1e-12)


@pytest.mark.parametrize("pose", scene.POSES)
def test_presets_cover_all_joints_within_description_range(description_root, hand, pose):
    assert set(scene.PRESET_POSES[pose]) == set(conventions.JOINT_SUFFIXES)
    model = description_spec(description_root, hand).compile()
    names = hinge_names(model)
    q = scene.pose_vector(names, pose)
    assert q.shape == (20,)
    assert q.dtype == np.float64
    for name, value in zip(names, q):
        lo, hi = model.jnt_range[model.joint(name).id]
        assert lo <= value <= hi, (name, value, lo, hi)


def test_pose_vector_follows_joint_name_order():
    forward = scene.pose_vector(RIGHT_NAMES, "fist")
    backward = scene.pose_vector(RIGHT_NAMES[::-1], "fist")
    assert backward.tolist() == forward[::-1].tolist()
    assert forward.tolist() == [scene.PRESET_POSES["fist"][s] for s in conventions.JOINT_SUFFIXES]
    left = scene.pose_vector(tuple("l_" + s for s in conventions.JOINT_SUFFIXES), "fist")
    assert left.tolist() == forward.tolist()


@pytest.mark.parametrize(
    "names",
    [
        RIGHT_NAMES[:-1],
        tuple("x_" + s for s in conventions.JOINT_SUFFIXES),
        tuple(conventions.JOINT_SUFFIXES),
        ("l_" + conventions.JOINT_SUFFIXES[0],) + RIGHT_NAMES[1:],
        RIGHT_NAMES + ("r_thumb_cmc_flex",),
        RIGHT_NAMES[:-1] + ("r_pinky_tip",),
    ],
    ids=["missing", "bad-prefix", "no-prefix", "mixed-prefix", "duplicate", "unknown"],
)
def test_pose_vector_rejects_bad_joint_names(names):
    with pytest.raises(ValueError):
        scene.pose_vector(names, "open")


def test_pose_vector_rejects_unknown_pose():
    with pytest.raises(ValueError, match="wave"):
        scene.pose_vector(RIGHT_NAMES, "wave")


def test_add_scene_adds_cameras_floor_and_lights(description_root, hand):
    view = scene_view(description_root, hand)
    spec = description_spec(description_root, hand)
    before = {
        "geoms": {g.name for g in spec.geoms},
        "lights": {x.name for x in spec.lights},
        "textures": {x.name for x in spec.textures},
        "materials": {x.name for x in spec.materials},
        "cameras": {x.name for x in spec.cameras},
        "bodies": {x.name for x in spec.bodies},
    }
    scene.add_scene(spec, hand, view)
    added = {
        "geoms": {g.name for g in spec.geoms} - before["geoms"],
        "lights": {x.name for x in spec.lights} - before["lights"],
        "textures": {x.name for x in spec.textures} - before["textures"],
        "materials": {x.name for x in spec.materials} - before["materials"],
        "cameras": {x.name for x in spec.cameras} - before["cameras"],
        "bodies": {x.name for x in spec.bodies} - before["bodies"],
    }
    assert added["cameras"] == {*scene.CAMERAS, scene.PREVIEW_CAMERA}
    assert added["geoms"] == {"scene_floor"}
    assert added["bodies"] == {scene.PREVIEW_TARGET_BODY}
    for kind in ("lights", "textures", "materials"):
        assert added[kind], kind
        assert all(name.startswith("scene_") for name in added[kind]), added[kind]

    model = spec.compile()
    floor = model.geom("scene_floor")
    assert model.geom_type[floor.id] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.geom_pos[floor.id].tolist() == [0.0, 0.0, view.floor_z]
    assert model.geom_contype[floor.id] and model.geom_conaffinity[floor.id]
    target = model.body(f"{PREFIX[hand]}middle_finger_proximal").id
    for name in scene.CAMERAS:
        cam = model.camera(name)
        assert model.cam_mode[cam.id] == mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        assert model.cam_targetbodyid[cam.id] == target
        assert model.cam_fovy[cam.id] == 45.0
    root = np.array(view.hand_roots[0])
    front_side = 1.0 if hand == "right" else -1.0
    np.testing.assert_allclose(model.cam_pos[model.camera("front").id] - root, [0.03, front_side * 0.35, 0.20])
    np.testing.assert_allclose(model.cam_pos[model.camera("side").id] - root, [0.36, 0.0, 0.20])
    np.testing.assert_allclose(model.cam_pos[model.camera("back").id] - root, [-0.017, -front_side * 0.334, 0.20])


def test_add_scene_aims_the_preview_camera_and_the_lights_at_the_palm(description_root, hand):
    view = scene_view(description_root, hand)
    model = scene_model(description_root, hand, view)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = model.camera(scene.PREVIEW_CAMERA).id
    target = model.body(scene.PREVIEW_TARGET_BODY).id
    assert model.cam_mode[camera] == mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    assert model.cam_targetbodyid[camera] == target
    assert model.body_geomnum[target] == 0
    assert model.cam_fovy[camera] == model.vis.global_.fovy == view.fovy_deg
    np.testing.assert_allclose(data.cam_xpos[camera], view.camera_position, atol=1e-12)
    np.testing.assert_allclose(data.xpos[target], view.camera_target, atol=1e-12)
    looking = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(looking, np.subtract(view.camera_target, view.camera_position) / view.distance_m,
                               atol=1e-12)
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == view.render_size
    side = framing.CAMERA_SIDE[hand]
    key = model.light("scene_key_light").id
    assert model.light_castshadow[key]
    assert model.light_pos[key][1] * side > 0.0 and model.light_dir[key][1] * side < 0.0
    assert model.light_dir[model.light("scene_fill_light").id][1] * side > 0.0


def test_add_scene_puts_the_hand_root_on_the_first_grid_root(description_root):
    view = scene_view(description_root, "right", kind="grid", worlds=4)
    model = scene_model(description_root, "right", view)
    root = model.body("r_mount").id
    np.testing.assert_allclose(model.body_pos[root], view.hand_roots[0])
    np.testing.assert_allclose(view.hand_roots[0], [0.15, -0.15, 0.0])
    np.testing.assert_allclose(model.cam_pos[model.camera(scene.PREVIEW_CAMERA).id], view.camera_position)


def test_back_camera_mirrors_front_across_the_hand(description_root, hand):
    model = scene_model(description_root, hand)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target = data.xpos[model.body(f"{PREFIX[hand]}middle_finger_proximal").id]
    front, back = (data.cam_xpos[model.camera(name).id] - target for name in ("front", "back"))
    np.testing.assert_allclose(back, [-front[0], -front[1], front[2]], atol=1e-3)
    view_dirs = [data.cam_xmat[model.camera(name).id].reshape(3, 3)[:, 2] for name in ("front", "back")]
    assert np.dot(*view_dirs) < -0.8


def test_default_scene_fixes_wrist_and_finger_axis(description_root, hand):
    view = scene_view(description_root, hand)
    model = scene_model(description_root, hand, view)
    roots = [body for body in range(1, model.nbody) if model.body_parentid[body] == 0]
    assert roots
    root = roots[0]
    np.testing.assert_allclose(model.body_quat[root], scene.HAND_ROOT_QUAT)
    np.testing.assert_allclose(model.body_pos[root], view.hand_roots[0])
    np.testing.assert_allclose(model.opt.gravity, conventions.GRAVITY)
    assert scene.scene_record(hand, view) == {
        "wrist": "fixed",
        "finger_axis": "+z",
        "gravity": conventions.GRAVITY,
        "floor_z": view.floor_z,
        "cameras": [*scene.CAMERAS, scene.PREVIEW_CAMERA],
        "preview": view.record(),
    }


def test_add_scene_rejects_unknown_hand(description_root):
    with pytest.raises(ValueError, match="both"):
        scene.add_scene(description_spec(description_root, "right"), "both", scene_view(description_root, "right"))


@pytest.mark.parametrize(("view_hand", "platform", "match"), [("left", "mujoco", "left"), ("right", "isaaclab", "isaaclab")])
def test_add_scene_and_scene_record_reject_a_view_of_another_hand_or_platform(description_root, view_hand, platform,
                                                                              match):
    view = scene_view(description_root, view_hand, platform)
    with pytest.raises(ValueError, match=match):
        scene.add_scene(description_spec(description_root, "right"), "right", view)
    if view_hand != "right":
        with pytest.raises(ValueError, match=match):
            scene.scene_record("right", view)


def test_add_scene_hides_the_wrist_mount_with_the_collision_meshes(description_root, hand):
    model = scene_model(description_root, hand)
    mount = model.body(f"{PREFIX[hand]}mount").id
    mount_groups = {int(model.geom_group[g]) for g in range(model.ngeom) if model.geom_bodyid[g] == mount}
    assert mount_groups == {2}
    visible_hand_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] not in (0, mount) and model.geom_group[g] < 2]
    assert visible_hand_geoms


def test_env_grid_offsets_match_isaac_lab_cloner_grid():
    np.testing.assert_allclose(
        scene.env_grid_offsets(4, 0.3),
        [[0.15, -0.15, 0.0], [0.15, 0.15, 0.0], [-0.15, -0.15, 0.0], [-0.15, 0.15, 0.0]],
    )
    np.testing.assert_allclose(scene.env_grid_offsets(1, 0.3), [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(scene.env_grid_offsets(3, 0.3), [[0.15, -0.15, 0.0], [0.15, 0.15, 0.0], [-0.15, -0.15, 0.0]])
    with pytest.raises(ValueError, match="count"):
        scene.env_grid_offsets(0, 0.3)
    for spacing in (0.0, -0.3, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="spacing"):
            scene.env_grid_offsets(4, spacing)


@pytest.mark.parametrize("world_frame", [False, True])
def test_renderer_outputs_non_uniform_rgb(description_root, hand, world_frame):
    model = scene_model(description_root, hand)
    data = posed_data(model, "open")
    with scene.SceneRenderer(model, world_frame=world_frame) as renderer:
        for camera in scene.CAMERAS:
            rgb = renderer.render(data, camera)
            assert rgb.shape == (480, 640, 3)
            assert rgb.dtype == np.uint8
            assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 1


def test_renderer_renders_the_preview_camera_at_the_view_size(description_root, hand):
    view = scene_view(description_root, hand)
    model = scene_model(description_root, hand, view)
    with scene.SceneRenderer(model, *view.render_size) as renderer:
        rgb = renderer.render(posed_data(model, "open"), scene.PREVIEW_CAMERA)
    assert rgb.shape == (view.render_size[1], view.render_size[0], 3)
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 1


def test_renderer_runs_extra_on_the_updated_scene_before_rendering(description_root):
    model = scene_model(description_root, "right")
    data = posed_data(model, "open")
    updated = []

    def hide_every_geom(scn):
        updated.append(scn.ngeom)
        scn.ngeom = 0

    with scene.SceneRenderer(model) as renderer:
        full = renderer.render(data, "front")
        empty = renderer.render(data, "front", extra=hide_every_geom)
        assert renderer.option.geomgroup[2] == 0
        assert not renderer.option.sitegroup.any()
        with pytest.raises(AttributeError):
            renderer.option = mujoco.MjvOption()
    assert len(updated) == 1 and updated[0] > 0
    # Two renders of one scene differ in a few pixels; hiding every geom changes most of the frame.
    assert np.abs(full.astype(np.int16) - empty).mean() > 10.0


@pytest.mark.parametrize("pose", scene.POSES)
def test_presets_are_contact_free_along_the_path_from_open(description_root, hand, pose):
    model = scene_model(description_root, hand)
    data = mujoco.MjData(model)
    names = hinge_names(model)
    start, goal = scene.pose_vector(names, "open"), scene.pose_vector(names, pose)
    for s in np.linspace(0.0, 1.0, 51):
        set_joints(model, data, start + s * (goal - start))
        assert contact_pairs(model, data) == [], s


def test_parallel_cycle_is_contact_free_in_every_segment(description_root, hand):
    model = scene_model(description_root, hand)
    data = mujoco.MjData(model)
    names = hinge_names(model)
    period = scene.MOVE_S + scene.HOLD_S
    # World 0 starts the cycle on its first pose, so its reference runs through every segment once.
    for segment in range(len(scene.TRACKING_SEQUENCE)):
        for s in np.linspace(0.0, 1.0, 51):
            set_joints(model, data, scene.parallel_reference(names, 0, segment * period + s * scene.MOVE_S)[0])
            assert contact_pairs(model, data) == [], (segment, s)


@pytest.mark.parametrize("suffix", ["thumb_cmc_flex", "thumb_mcp", "thumb_ip"])
def test_fist_thumb_keeps_a_margin_to_the_four_fingers(description_root, hand, suffix):
    model = scene_model(description_root, hand)
    data = mujoco.MjData(model)
    names = hinge_names(model)
    q = scene.pose_vector(names, "fist")
    q[names.index(PREFIX[hand] + suffix)] += 0.05
    set_joints(model, data, q)
    thumb = PREFIX[hand] + "thumb"
    fingers = tuple(PREFIX[hand] + finger for finger in FINGERS)
    hits = [
        pair
        for pair in contact_pairs(model, data)
        if any(a.startswith(thumb) and b.startswith(fingers) for a, b in (pair[:2], pair[1::-1]))
    ]
    assert hits == []


def test_tracking_reference_hits_poses_at_segment_ends():
    period = scene.MOVE_S + scene.HOLD_S
    assert scene.tracking_duration() == pytest.approx((len(scene.TRACKING_SEQUENCE) - 1) * period)
    for k in range(1, len(scene.TRACKING_SEQUENCE)):
        start = (k - 1) * period
        previous = scene.pose_vector(RIGHT_NAMES, scene.TRACKING_SEQUENCE[k - 1])
        target = scene.pose_vector(RIGHT_NAMES, scene.TRACKING_SEQUENCE[k])
        for t, expected in ((start, previous), (start + scene.MOVE_S, target), (k * period, target)):
            q, qd = scene.tracking_reference(RIGHT_NAMES, t)
            np.testing.assert_allclose(q, expected, atol=1e-12)
            np.testing.assert_allclose(qd, 0.0, atol=1e-12)
        q_mid, qd_mid = scene.tracking_reference(RIGHT_NAMES, start + scene.MOVE_S / 2)
        np.testing.assert_allclose(q_mid, (previous + target) / 2, atol=1e-12)
        assert np.abs(qd_mid).max() > 0.0
    q_end, qd_end = scene.tracking_reference(RIGHT_NAMES, scene.tracking_duration() + 1.0)
    np.testing.assert_allclose(q_end, scene.pose_vector(RIGHT_NAMES, scene.TRACKING_SEQUENCE[-1]))
    np.testing.assert_allclose(qd_end, 0.0)


@pytest.mark.parametrize(("duration", "timestep", "steps"), [(1.0, 0.01, 100), (1.0, 0.002, 500), (4.8, 0.01, 480)])
def test_whole_steps_counts_physics_steps(duration, timestep, steps):
    assert scene.whole_steps(duration, timestep) == steps


@pytest.mark.parametrize(("duration", "timestep"), [(1.0, 0.003), (1.0, 0.3), (0.001, 0.01)])
def test_whole_steps_rejects_partial_steps(duration, timestep):
    with pytest.raises(ValueError, match="whole number"):
        scene.whole_steps(duration, timestep)


def test_tracking_reference_rejects_negative_time():
    with pytest.raises(ValueError):
        scene.tracking_reference(RIGHT_NAMES, -0.1)


def test_tracking_constants():
    assert scene.TRACKING_SEQUENCE == ("open", "half_grip", "fist", "open", "pinch", "open")
    assert (scene.MOVE_S, scene.HOLD_S, scene.DEMO_KP, scene.DEMO_KD) == (0.6, 0.2, 1.0, 0.03)
    assert scene.CAMERAS == ("front", "side", "back")
    assert scene.PREVIEW_CAMERA == "preview"


def test_parallel_reference_starts_each_world_on_its_pose_and_cycles_the_tracking_sequence():
    period = scene.MOVE_S + scene.HOLD_S
    sequence = scene.TRACKING_SEQUENCE
    count = len(sequence)
    assert scene.parallel_duration() == pytest.approx(count * period)
    starts = []
    for world in range(len(scene.POSES) + 2):
        start = sequence.index(scene.POSES[world % len(scene.POSES)])
        starts.append(start)
        poses = [scene.pose_vector(RIGHT_NAMES, sequence[(start + k) % count]) for k in range(count + 1)]
        q, qd = scene.parallel_reference(RIGHT_NAMES, world, 0.0)
        np.testing.assert_allclose(q, scene.pose_vector(RIGHT_NAMES, scene.POSES[world % len(scene.POSES)]),
                                   atol=1e-12)
        np.testing.assert_allclose(qd, 0.0, atol=1e-12)
        q_mid, qd_mid = scene.parallel_reference(RIGHT_NAMES, world, scene.MOVE_S / 2)
        np.testing.assert_allclose(q_mid, (poses[0] + poses[1]) / 2, atol=1e-12)
        assert np.abs(qd_mid).max() > 0.0
        for k in range(count):
            for t in (k * period + scene.MOVE_S, (k + 1) * period - 1e-9):
                q, qd = scene.parallel_reference(RIGHT_NAMES, world, t)
                np.testing.assert_allclose(q, poses[k + 1], atol=1e-9)
                np.testing.assert_allclose(qd, 0.0, atol=1e-6)
        for t in (0.1, 1.3, 2.9, 4.5):
            np.testing.assert_allclose(scene.parallel_reference(RIGHT_NAMES, world, t + scene.parallel_duration())[0],
                                       scene.parallel_reference(RIGHT_NAMES, world, t)[0], atol=1e-9)
    assert starts == [0, 1, 2, 4, 0, 1]


@pytest.mark.parametrize(("world", "t"), [(0, -0.1), (-1, 0.0)])
def test_parallel_reference_rejects_negative_time_or_world(world, t):
    with pytest.raises(ValueError):
        scene.parallel_reference(RIGHT_NAMES, world, t)


def test_add_cli_args_headless_flag():
    parser = argparse.ArgumentParser()
    scene.add_cli_args(parser)
    assert parser.parse_args(["--headless"]).headless is True
    assert parser.parse_args([]).headless is False


def test_headless_display_is_inert(description_root):
    view = scene_view(description_root, "right")
    model = scene_model(description_root, "right", view)
    data = mujoco.MjData(model)
    viewer_loaded = "mujoco.viewer" in sys.modules
    with scene.Display(model, data, view, headless=True) as display:
        assert display.is_running() is False
        display.sync()
    assert ("mujoco.viewer" in sys.modules) == viewer_loaded


def test_live_display_opens_on_the_preview_camera_with_side_panels(description_root):
    view = scene_view(description_root, "left")
    model = scene_model(description_root, "left", view)
    data = mujoco.MjData(model)
    fake_viewer = MagicMock()
    locked = []
    fake_viewer.lock.return_value.__enter__.side_effect = lambda: locked.append(True)
    fake_viewer.lock.return_value.__exit__.side_effect = lambda *exc: locked.append(False)
    fake_viewer.opt.geomgroup.__setitem__.side_effect = lambda *args: locked.append("write")
    with patch("wuji_sim.scene._require_display", lambda: None), \
         patch("mujoco.viewer.launch_passive", return_value=fake_viewer) as launch:
        with scene.Display(model, data, view, headless=False) as display:
            assert display.user_scn is fake_viewer.user_scn
    launch.assert_called_once_with(model, data, show_left_ui=True, show_right_ui=True)
    assert locked == [True, "write", False]
    fake_viewer.cam.lookat.__setitem__.assert_called_once_with(slice(None), view.camera_target)
    assert (fake_viewer.cam.distance, fake_viewer.cam.azimuth, fake_viewer.cam.elevation) == (
        view.distance_m, view.azimuth_deg, view.elevation_deg
    )
    fake_viewer.opt.geomgroup.__setitem__.assert_called_once_with(2, 0)
    fake_viewer.close.assert_called_once()


def test_live_display_closes_the_viewer_when_setup_fails(description_root):
    view = scene_view(description_root, "left")
    model = scene_model(description_root, "left", view)
    data = mujoco.MjData(model)
    fake_viewer = MagicMock()
    fake_viewer.opt.geomgroup.__setitem__.side_effect = RuntimeError("boom")
    with patch("wuji_sim.scene._require_display", lambda: None), \
         patch("mujoco.viewer.launch_passive", return_value=fake_viewer):
        with pytest.raises(conventions.ViewerError, match="boom.*--headless"):
            scene.Display(model, data, view, headless=False)
    fake_viewer.close.assert_called_once()


def test_live_display_rejects_a_degenerate_view_before_opening(description_root):
    view = scene_view(description_root, "left")
    model = scene_model(description_root, "left", view)
    data = mujoco.MjData(model)
    degenerate = dataclasses.replace(view, camera_position=view.camera_target)
    with patch("wuji_sim.scene._require_display", lambda: None), \
         patch("mujoco.viewer.launch_passive") as launch:
        with pytest.raises(conventions.ViewerError, match="--headless"):
            scene.Display(model, data, degenerate, headless=False)
    launch.assert_not_called()


def description_ranges(description_root, hand):
    return scene.description_joint_ranges(description_root / "mjcf" / f"{hand}.xml")


def expected_groups(prefix):
    thumb = [(suffix, [prefix + suffix]) for suffix in ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip")]
    fingers = [
        (f"fingers_{role}", [f"{prefix}{finger}_{role}" for finger in scene.SWEEP_FINGERS])
        for role in ("mcp_flex", "mcp_abd", "pip", "dip")
    ]
    return thumb + fingers


def test_sweep_groups_split_description_joints_into_thumb_joints_and_finger_roles(description_root, hand):
    names = list(description_ranges(description_root, hand))
    groups = scene.sweep_groups(names)
    assert [(label, [names[i] for i in members]) for label, members in groups] == expected_groups(PREFIX[hand])
    assert sorted(i for _, members in groups for i in members) == list(range(len(names)))


def test_sweep_groups_rejects_non_description_names():
    with pytest.raises(ValueError):
        scene.sweep_groups(RIGHT_NAMES[:-1] + ("r_pinky_tip",))


def test_sweep_reference_moves_one_window_at_a_time_inside_inner_limits(description_root, hand):
    names = tuple(PREFIX[hand] + suffix for suffix in conventions.JOINT_SUFFIXES)
    table = description_ranges(description_root, hand)
    ranges = np.array([table[name] for name in names])
    inner = scene.sweep_targets(names, ranges)
    margin = scene.SWEEP_MARGIN * (ranges[:, 1] - ranges[:, 0])
    np.testing.assert_allclose(inner, ranges + np.stack([margin, -margin], axis=1))
    groups = scene.sweep_groups(names)
    assert len(groups) == 8
    open_q = scene.pose_vector(names, "open")
    period = scene.SWEEP_MOVE_S + scene.SWEEP_HOLD_S
    moves = len(scene.SWEEP_WAYPOINTS) - 1
    seen, together = set(), set()
    for t in np.arange(0.0, scene.sweep_duration(names) + 0.5, 0.004):
        q, qd, window = scene.sweep_reference(names, ranges, t)
        assert window == min(int(t // period) // moves, len(groups) - 1)
        assert np.all(inner[:, 0] - 1e-12 <= q) and np.all(q <= inner[:, 1] + 1e-12), t
        moved = set(np.flatnonzero((q != open_q) | (qd != 0.0)).tolist())
        members = set(groups[window][1])
        assert moved in (set(), members), (t, moved, members)
        seen.add(window)
        if moved and len(members) == len(scene.SWEEP_FINGERS):
            together.add(window)
    assert seen == set(range(len(groups)))
    assert together == {window for window, (label, _) in enumerate(groups) if label.startswith("fingers_")}


def test_sweep_reference_reaches_each_joints_inner_limits_at_waypoints():
    ranges = np.array([[-1.0 - 0.01 * j, 2.0 + 0.02 * j] for j in range(len(RIGHT_NAMES))])
    inner = scene.sweep_targets(RIGHT_NAMES, ranges)
    period = scene.SWEEP_MOVE_S + scene.SWEEP_HOLD_S
    assert scene.SWEEP_WAYPOINTS == ("open", "inner_lower", "inner_upper", "open")
    assert scene.sweep_window_duration() == pytest.approx(3 * period)
    groups = scene.sweep_groups(RIGHT_NAMES)
    schedule = scene.sweep_schedule(RIGHT_NAMES)
    assert [entry["label"] for entry in schedule] == [label for label, _ in groups]
    assert scene.sweep_duration(RIGHT_NAMES) == pytest.approx(len(groups) * 3 * period)
    for window, (label, members) in enumerate(groups):
        members = list(members)
        others = [i for i in range(len(RIGHT_NAMES)) if i not in members]
        start = window * scene.sweep_window_duration()
        assert schedule[window]["t_s"] == pytest.approx([start, start + 3 * period])
        assert schedule[window]["joints"] == [RIGHT_NAMES[i] for i in members]
        q, qd, got = scene.sweep_reference(RIGHT_NAMES, ranges, start)
        assert got == window
        np.testing.assert_allclose(q, 0.0)
        np.testing.assert_allclose(qd, 0.0)
        expected = (inner[members, 0], inner[members, 1], np.zeros(len(members)))
        for k, value in enumerate(expected):
            t = start + k * period + scene.SWEEP_MOVE_S
            assert schedule[window]["waypoint_reached_s"][k] == pytest.approx(t)
            q, qd, got = scene.sweep_reference(RIGHT_NAMES, ranges, t)
            assert got == window
            np.testing.assert_allclose(q[members], value, atol=1e-12)
            np.testing.assert_allclose(q[others], 0.0)
            np.testing.assert_allclose(qd, 0.0, atol=1e-12)
    q_end, qd_end, window = scene.sweep_reference(RIGHT_NAMES, ranges, scene.sweep_duration(RIGHT_NAMES) + 1.0)
    assert window == len(groups) - 1
    np.testing.assert_allclose(q_end, 0.0, atol=1e-12)
    np.testing.assert_allclose(qd_end, 0.0)


@pytest.mark.parametrize(
    "ranges",
    [
        np.tile([0.1, 1.0], (20, 1)),
        np.tile([-0.01, 1.0], (20, 1)),
        np.tile([1.0, -1.0], (20, 1)),
        np.tile([-1.0, 1.0], (19, 1)),
        np.tile([-1.0, np.nan], (20, 1)),
    ],
    ids=["open-below-range", "open-inside-margin", "inverted", "wrong-count", "nan"],
)
def test_sweep_targets_rejects_bad_ranges(ranges):
    with pytest.raises(ValueError):
        scene.sweep_targets(RIGHT_NAMES, ranges)


def test_sweep_reference_rejects_negative_time():
    with pytest.raises(ValueError):
        scene.sweep_reference(RIGHT_NAMES, np.tile([-1.0, 1.0], (20, 1)), -0.1)


def test_description_joint_ranges_reads_radians_and_converts_degrees(tmp_path):
    template = (
        '<mujoco><compiler angle="{angle}"/><worldbody><body><joint name="j" type="hinge" {attrs}/>'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    cases = {
        "radian": ('radian', 'range="-1.0 0.5"', [-1.0, 0.5]),
        "degree": ('degree', 'range="-90 45"', [-np.pi / 2, np.pi / 4]),
    }
    for label, (angle, attrs, expected) in cases.items():
        path = tmp_path / f"{label}.xml"
        path.write_text(template.format(angle=angle, attrs=attrs))
        got = scene.description_joint_ranges(path)
        assert list(got) == ["j"]
        np.testing.assert_allclose(got["j"], expected)
    path = tmp_path / "unlimited.xml"
    path.write_text(template.format(angle="radian", attrs='limited="false" range="-1 1"'))
    with pytest.raises(scene.JointTableError, match="unlimited"):
        scene.description_joint_ranges(path)


def test_check_joint_table_accepts_the_compiled_description_model(description_root, hand):
    description = description_ranges(description_root, hand)
    model = scene_model(description_root, hand)
    compiled = {name: model.jnt_range[model.joint(name).id].tolist() for name in hinge_names(model)}
    scene.check_joint_table(hand, description, compiled)
    device = {name: np.asarray(value, dtype=np.float32).astype(float).tolist() for name, value in compiled.items()}
    scene.check_joint_table(hand, description, device)


def _drop_last(table):
    return dict(list(table.items())[:-1])


def _rename_first(table):
    items = list(table.items())
    items[0] = ("r_thumb_extra", items[0][1])
    return dict(items)


def _swap_first_two(table):
    items = list(table.items())
    items[0], items[1] = items[1], items[0]
    return dict(items)


def _shift_upper_limit(table):
    shifted = dict(table)
    lo, hi = shifted["r_index_finger_pip"]
    shifted["r_index_finger_pip"] = [lo, hi + 1e-3]
    return shifted


@pytest.mark.parametrize(
    "mutate", [_drop_last, _rename_first, _swap_first_two, _shift_upper_limit], ids=lambda f: f.__name__[1:]
)
def test_check_joint_table_rejects_compiled_mismatch(description_root, mutate):
    description = description_ranges(description_root, "right")
    with pytest.raises(scene.JointTableError, match="does not match description"):
        scene.check_joint_table("right", description, mutate(description))


@pytest.mark.parametrize("mutate", [_drop_last, _rename_first, _swap_first_two], ids=lambda f: f.__name__[1:])
def test_check_joint_table_rejects_description_with_unexpected_joints(description_root, mutate):
    changed = mutate(description_ranges(description_root, "right"))
    with pytest.raises(scene.JointTableError, match="description MJCF has"):
        scene.check_joint_table("right", changed, changed)


def test_check_joint_table_rejects_the_other_hand(description_root):
    description = description_ranges(description_root, "right")
    with pytest.raises(scene.JointTableError):
        scene.check_joint_table("left", description, description)


def test_sweep_joint_table_reports_reach_error_and_saturation_per_window():
    n = len(RIGHT_NAMES)
    ranges = {name: [-1.0, 2.0] for name in RIGHT_NAMES}
    windows = len(scene.sweep_groups(RIGHT_NAMES))
    active = np.repeat(np.arange(windows), 3)
    q = np.zeros((3 * windows, n))
    q[active == 5, 5] = [-0.5, 0.0, 0.7]
    error = np.zeros_like(q)
    error[0, 9] = -0.06
    saturated = np.zeros_like(q, dtype=bool)
    saturated[-1, n - 1] = True
    rows = scene.sweep_joint_table(RIGHT_NAMES, ranges, ranges, active, q, error, saturated)
    assert [row["joint"] for row in rows] == list(RIGHT_NAMES)
    assert [rows[i]["window"] for i in (0, 3, 5, 9, 19)] == [
        "thumb_cmc_flex", "thumb_ip", "fingers_mcp_abd", "fingers_mcp_abd", "fingers_dip"
    ]
    assert (rows[5]["reached_min_rad"], rows[5]["reached_max_rad"]) == (-0.5, 0.7)
    assert rows[5]["sweep_target_rad"] == pytest.approx([-0.85, 1.85])
    duration = scene.sweep_window_duration()
    assert rows[5]["sweep_window_s"] == rows[9]["sweep_window_s"] == pytest.approx([5 * duration, 6 * duration])
    assert (rows[9]["max_abs_error_active_rad"], rows[9]["max_abs_error_run_rad"]) == (0.0, 0.06)
    assert rows[n - 1]["saturation_fraction_active"] == pytest.approx(1 / 3)
    assert rows[n - 1]["saturation_fraction_run"] == pytest.approx(1 / (3 * windows))
    assert scene.sweep_flags(rows) == ["r_middle_finger_mcp_abd", RIGHT_NAMES[-1]]
