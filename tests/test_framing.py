# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import math

import mujoco
import numpy as np
import pytest

from wuji_sim import assets, framing, scene

HANDS = ("left", "right")
# The framing behind the README previews on MuJoCo: camera position and target (m), distance (m), render size (px).
README_FRAMING = {
    ("right", "tracking", 1): ((0.02838, -0.45701, 0.34745), (0.02838, -0.03284, 0.19306), 0.45140, (870, 960)),
    ("left", "tracking", 1): ((0.02838, 0.45701, 0.34745), (0.02838, 0.03284, 0.19306), 0.45139, (870, 960)),
    ("left", "sweep", 1): ((0.01216, 0.53686, 0.37182), (0.01216, 0.00063, 0.17665), 0.57064, (968, 960)),
    ("right", "grid", 4): ((0.02838, -0.83046, 0.78186), (0.02838, -0.03284, 0.11257), 1.04123, (1286, 960)),
}
# The README previews on Isaac Lab put the floor here (m), FLOOR_CLEARANCE_M below the lowest point the hand reaches.
ISAAC_LAB_FLOOR_Z = {
    ("right", "tracking", 1): -0.06447,
    ("left", "tracking", 1): -0.06447,
    ("left", "sweep", 1): -0.09752,
    ("right", "grid", 4): -0.06447,
}
PALM_VIEW_MAX_DEG = 60.0


def model_path(description_root, hand):
    return description_root / assets.MJCF_ENTRIES[hand]


def preview(description_root, hand, kind, platform="mujoco", worlds=1):
    return framing.view(model_path(description_root, hand), hand, platform, kind, worlds=worlds)


def palm_and_nail_centroids(description_root, hand):
    spec = mujoco.MjSpec.from_file(str(model_path(description_root, hand)))
    for body in spec.worldbody.bodies:
        body.quat = scene.HAND_ROOT_QUAT
    model = spec.compile()
    data = mujoco.MjData(model)
    names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    data.qpos[[model.jnt_qposadr[model.joint(name).id] for name in names]] = scene.pose_vector(names, "open")
    mujoco.mj_forward(model, data)
    centres = {"palm_cover": [], "nail_cap": []}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] >= 2:
            continue
        mesh = int(model.geom_dataid[g])
        part = next((part for part in centres if part in model.mesh(mesh).name), None)
        if part is None:
            continue
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        centres[part].append((vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]).mean(axis=0))
    assert [len(found) for found in centres.values()] == [5, 5]
    return np.mean(centres["palm_cover"], axis=0), np.mean(centres["nail_cap"], axis=0)


@pytest.mark.parametrize(("hand", "kind", "worlds"), list(README_FRAMING))
def test_view_reproduces_the_readme_framing(description_root, hand, kind, worlds):
    position, target, distance, render_size = README_FRAMING[hand, kind, worlds]
    view = preview(description_root, hand, kind, worlds=worlds)
    assert (view.platform, view.hand, view.kind) == ("mujoco", hand, kind)
    np.testing.assert_allclose(view.camera_position, position, atol=1e-4)
    np.testing.assert_allclose(view.camera_target, target, atol=1e-4)
    assert view.distance_m == pytest.approx(distance, abs=1e-4)
    assert view.render_size == render_size
    assert view.fovy_deg == framing.FOVY_DEG == 27.0


@pytest.mark.parametrize(("hand", "kind", "worlds"), list(ISAAC_LAB_FLOOR_Z))
def test_floor_sits_at_the_origin_on_mujoco_and_below_the_hand_on_isaac_lab(description_root, hand, kind, worlds):
    for platform in ("mujoco", "mujoco_warp"):
        assert preview(description_root, hand, kind, platform, worlds).floor_z == framing.MUJOCO_FLOOR_Z == 0.0
    isaac_view = preview(description_root, hand, kind, "isaaclab", worlds)
    assert isaac_view.floor_z == pytest.approx(ISAAC_LAB_FLOOR_Z[hand, kind, worlds], abs=1e-4)


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize("kind", ["tracking", "sweep"])
def test_single_hand_views_raise_the_mujoco_hand_and_keep_the_isaac_lab_hand_at_the_origin(description_root, hand,
                                                                                          kind):
    mujoco_view = preview(description_root, hand, kind)
    warp_view = preview(description_root, hand, kind, platform="mujoco_warp")
    isaac_view = preview(description_root, hand, kind, platform="isaaclab")
    assert dataclasses.replace(warp_view, platform="mujoco") == mujoco_view
    assert mujoco_view.hand_roots == ((0.0, 0.0, 0.08),)
    assert isaac_view.hand_roots == ((0.0, 0.0, 0.0),)
    np.testing.assert_allclose(isaac_view.camera_position, np.subtract(mujoco_view.camera_position, [0.0, 0.0, 0.08]),
                               atol=1e-12)
    np.testing.assert_allclose(isaac_view.camera_target, np.subtract(mujoco_view.camera_target, [0.0, 0.0, 0.08]),
                               atol=1e-12)
    assert isaac_view.render_size == mujoco_view.render_size


@pytest.mark.parametrize("platform", ["mujoco", "mujoco_warp", "isaaclab"])
def test_grid_view_lays_the_hands_out_on_the_env_grid(description_root, platform):
    view = preview(description_root, "right", "grid", platform=platform, worlds=4)
    np.testing.assert_allclose(view.hand_roots, scene.env_grid_offsets(4, scene.PARALLEL_SPACING_M), atol=1e-15)
    assert dataclasses.replace(view, platform="mujoco", floor_z=0.0) == preview(description_root, "right", "grid", worlds=4)
    assert preview(description_root, "right", "grid", worlds=1).hand_roots == ((0.0, 0.0, 0.0),)


@pytest.mark.parametrize(
    ("hand", "kind", "worlds"),
    [(hand, kind, 4 if kind == "grid" else 1) for hand in HANDS for kind in framing.KINDS if kind != "pen_spin"],
)
def test_preview_camera_sees_the_palm(description_root, hand, kind, worlds):
    view = preview(description_root, hand, kind, worlds=worlds)
    palm, nails = palm_and_nail_centroids(description_root, hand)
    # The palm covers lie on the palm side of the nail caps.
    palm_side = math.copysign(1.0, palm[1] - nails[1])
    assert palm_side == framing.CAMERA_SIDE[hand]
    for root in view.hand_roots:
        to_camera = np.subtract(view.camera_position, palm + root)
        to_camera /= np.linalg.norm(to_camera)
        angle = math.degrees(math.acos(to_camera[1] * palm_side))
        assert angle < PALM_VIEW_MAX_DEG, (root, angle)


@pytest.mark.parametrize(
    ("hand", "kind", "worlds"), [("right", "tracking", 1), ("left", "sweep", 1), ("right", "grid", 4)]
)
def test_view_angles_place_a_free_camera_on_the_preview_camera(description_root, hand, kind, worlds):
    view = preview(description_root, hand, kind, worlds=worlds)
    azimuth, elevation = math.radians(view.azimuth_deg), math.radians(view.elevation_deg)
    forward = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation)])
    np.testing.assert_allclose(np.subtract(view.camera_target, view.distance_m * forward), view.camera_position,
                               atol=1e-12)
    assert view.elevation_deg == pytest.approx(-framing.PITCH_DEG[kind])
    assert view.azimuth_deg == pytest.approx(90.0 if hand == "right" else -90.0)


def test_view_record_round_trips_through_json(description_root):
    view = preview(description_root, "right", "grid", worlds=4)
    assert json.loads(json.dumps(view.record())) == {
        "platform": "mujoco",
        "hand": "right",
        "kind": "grid",
        "camera_position": list(view.camera_position),
        "camera_target": list(view.camera_target),
        "distance_m": view.distance_m,
        "azimuth_deg": view.azimuth_deg,
        "elevation_deg": view.elevation_deg,
        "fovy_deg": 27.0,
        "render_size": [1286, 960],
        "floor_z": view.floor_z,
        "hand_roots": [list(root) for root in view.hand_roots],
    }


@pytest.mark.parametrize(
    ("hand", "platform", "kind", "worlds"),
    [
        ("both", "mujoco", "tracking", 1),
        ("right", "genesis", "tracking", 1),
        ("right", "mujoco", "orbit", 1),
        ("right", "mujoco", "tracking", 4),
        ("right", "mujoco", "sweep", 2),
        ("right", "mujoco", "grid", 0),
    ],
)
def test_view_rejects_bad_arguments(description_root, hand, platform, kind, worlds):
    with pytest.raises(ValueError):
        framing.view(model_path(description_root, "right"), hand, platform, kind, worlds=worlds)


def test_view_rejects_the_other_hands_model(description_root):
    with pytest.raises(ValueError, match="l_"):
        framing.view(model_path(description_root, "left"), "right", "mujoco", "tracking")


@pytest.mark.parametrize("platform", ["mujoco", "mujoco_warp", "isaaclab"])
def test_pen_spin_view_looks_down_on_the_palm_from_behind_the_wrist(description_root, platform):
    view = framing.view(model_path(description_root, "right"), "right", platform, "pen_spin")
    assert view.kind == "pen_spin"
    assert view.platform == platform
    assert view.camera_target == (-0.11, 0.0, 0.53)
    assert view.distance_m == pytest.approx(0.70, abs=1e-9)
    assert view.azimuth_deg == pytest.approx(-140.0, abs=1e-9)
    assert view.elevation_deg == pytest.approx(-45.0, abs=1e-9)
    # camera_target - distance * (cos(el) cos(az), cos(el) sin(az), sin(el)), worked out by hand.
    np.testing.assert_allclose(view.camera_position, (0.2691727, 0.3181636, 1.0249747), atol=1e-6)
    assert view.fovy_deg == 27.0
    assert view.render_size == (1152, 960)
    assert view.floor_z == 0.0
    assert view.hand_roots == ((0.0, 0.0, 0.0),)


def test_pen_spin_view_refuses_the_left_hand(description_root):
    with pytest.raises(ValueError):
        framing.view(model_path(description_root, "left"), "left", "mujoco", "pen_spin")
    with pytest.raises(ValueError):
        framing.view(model_path(description_root, "right"), "right", "mujoco", "tracking", worlds=2)
    with pytest.raises(ValueError):
        framing.view(model_path(description_root, "right"), "right", "mujoco", "pen_spin", worlds=2)
