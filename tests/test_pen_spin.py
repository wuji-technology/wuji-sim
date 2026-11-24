# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import io
import math
import urllib.request
from pathlib import Path
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np
import pen_spin
import pytest

PUBLIC = Path(__file__).resolve().parents[1]
GOLD = PUBLIC / "tests" / "data" / "pen_spin_training_env_obs.npz"
ARCHIVE = f"{pen_spin.RELEASE_NAME}.zip"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture(scope="module")
def gold() -> dict:
    with np.load(GOLD) as data:
        return {key: data[key] for key in data.files}


def reference_of(gold: dict) -> pen_spin.Reference:
    return pen_spin.Reference(gold["reference_qpos"], gold["wrist_pose_w"], gold["object_pose_w"], gold["link_pos_w"])


def test_observation_matches_the_training_environment_at_reset(gold):
    reference = reference_of(gold)
    joint_pos = gold["reference_qpos"][0]
    joint_vel = (gold["reference_qpos"][1] - gold["reference_qpos"][0]) / pen_spin.CONTROL_DT
    # The training environment advances the command once at reset, so its first observation carries frame 1's command.
    obs = pen_spin.observation(reference.command(1), joint_pos, joint_vel, reference.pen_pose(0),
                               np.zeros(pen_spin.ACTION_DIM, dtype=np.float32))
    assert obs.shape == (pen_spin.OBS_DIM,)
    assert np.abs(obs - gold["reset_obs"]).max() <= 1e-6


def test_observation_matches_the_training_environment_after_one_control_step(gold):
    reference = reference_of(gold)
    pen_local = pen_spin.to_wrist_local(gold["next_pen_pose_w"], pen_spin.wrist_pose(np.zeros(3)))
    obs = pen_spin.observation(reference.command(2), gold["next_joint_pos"], np.zeros(pen_spin.JOINTS), pen_local,
                               gold["reset_action"])
    # The joint velocities of that state were not recorded; every other dimension is compared.
    compared = np.ones(pen_spin.OBS_DIM, dtype=bool)
    compared[pen_spin.COMMAND_DIM + pen_spin.JOINTS:pen_spin.COMMAND_DIM + 2 * pen_spin.JOINTS] = False
    assert np.abs(obs - gold["next_obs"])[compared].max() <= 1e-6


def test_observation_refuses_non_finite_state(gold):
    reference = reference_of(gold)
    joint_pos = gold["reference_qpos"][0].copy()
    joint_pos[3] = np.nan
    with pytest.raises(pen_spin.PenSpinError, match="finite"):
        pen_spin.observation(reference.command(0), joint_pos, np.zeros(pen_spin.JOINTS), reference.pen_pose(0),
                             np.zeros(pen_spin.ACTION_DIM))


def test_reference_replays_the_clip_from_its_first_frame(gold):
    reference = reference_of(gold)
    assert reference.num_steps == 3
    assert [reference.restart(step) for step in range(7)] == [True, False, False, True, False, False, True]
    assert np.array_equal(reference.joint_pos(4), gold["reference_qpos"][1])
    assert np.array_equal(reference.command(4), reference.command(1))


def test_reference_owns_its_tables_and_hands_out_read_only_rows(gold):
    qpos = gold["reference_qpos"].astype(np.float32)
    reference = pen_spin.Reference(qpos, gold["wrist_pose_w"], gold["object_pose_w"], gold["link_pos_w"])
    joint_pos, command = reference.joint_pos(0).copy(), reference.command(0).copy()
    qpos[0] += 1.0
    assert np.array_equal(reference.joint_pos(0), joint_pos)
    assert np.array_equal(reference.command(0), command)
    for row in (reference.command(0), reference.joint_pos(0), reference.pen_pose(0)):
        with pytest.raises(ValueError, match="read-only"):
            row[0] = 0.0


def test_residual_action_follows_the_published_mapping():
    config = {"action_scale": 0.5, "action_residual_clip": [-0.6, 0.6], "action_ema_alpha": 0.5}
    action = pen_spin.ResidualAction(config)
    raw = np.full(pen_spin.ACTION_DIM, 4.0, dtype=np.float32)
    raw[0] = 0.2
    reference = np.full(pen_spin.ACTION_DIM, 0.1, dtype=np.float32)
    first = action.step(raw, reference)
    second = action.step(raw, reference)
    assert first.shape == (pen_spin.ACTION_DIM,)
    assert first[:2] == pytest.approx([0.1 + 0.05, 0.1 + 0.3])
    assert second[:2] == pytest.approx([0.1 + 0.075, 0.1 + 0.45])
    action.reset()
    assert action.step(raw, reference)[:2] == pytest.approx([0.1 + 0.05, 0.1 + 0.3])


def test_pen_mass_and_inertia_follow_the_training_formula():
    model_source = mujoco.MjSpec()
    pen_spin.add_pen(model_source, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    model = model_source.compile()
    pen = model.body(pen_spin.PEN_BODY).id
    radius = pen_spin.PEN_DIAMETER_M / 2
    shaft_length = pen_spin.PEN_LENGTH_M - 0.072
    end_center = pen_spin.PEN_LENGTH_M / 2 - 0.018
    shaft_volume = math.pi * radius**2 * shaft_length
    box_volume = 0.018 * 0.018 * 0.036
    density = pen_spin.PEN_MASS_KG / (shaft_volume + 2 * box_volume)
    shaft_mass, box_mass = density * shaft_volume, density * box_volume
    axial = shaft_mass * radius**2 / 2 + 2 * box_mass * (0.018**2 + 0.018**2) / 12
    transverse = (shaft_mass * (3 * radius**2 + shaft_length**2) / 12
                  + 2 * box_mass * ((0.018**2 + 0.036**2) / 12 + end_center**2))
    assert model.body_mass[pen] == pytest.approx(0.030, rel=1e-9)
    assert np.sort(model.body_inertia[pen]) == pytest.approx(sorted([axial, transverse, transverse]), rel=1e-6)
    assert model.body_ipos[pen] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_pen_shaft_collides_as_a_prism_mesh_and_shows_as_a_cylinder():
    model_source = mujoco.MjSpec()
    pen_spin.add_pen(model_source, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    model = model_source.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    collision, shown = model.geom("pen_shaft_collision").id, model.geom("pen_shaft").id
    # MuJoCo Warp gives several contact points only to mesh and box pairs; a colliding cylinder would get one.
    assert model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[collision] and model.geom_conaffinity[collision]
    assert model.geom_group[collision] == 2
    assert model.geom_type[shown] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert not model.geom_contype[shown] and not model.geom_conaffinity[shown]
    assert model.geom_group[shown] != 2
    mesh = model.geom_dataid[collision]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(np.float64)
    # The compiler recentres and realigns mesh vertices; the geom pose puts them back.
    points = vertices @ data.geom_xmat[collision].reshape(3, 3).T + data.geom_xpos[collision]
    assert len(points) == 2 * pen_spin.PEN_SHAFT_SIDES
    assert np.abs(points[:, 0]).max() == pytest.approx(0.089, rel=1e-6)
    assert np.linalg.norm(points[:, 1:], axis=1).max() == pytest.approx(pen_spin.PEN_DIAMETER_M / 2, rel=1e-6)


def test_pen_shows_a_white_shaft_and_the_label_faces_of_the_training_pen():
    textures = Path(pen_spin.__file__).resolve().parent / "textures" / "pen_spin"
    assert sorted(path.name for path in textures.glob("*.png")) == [
        "bottom_back.png", "bottom_down.png", "bottom_left.png", "bottom_right.png", "bottom_up.png",
        "top_down.png", "top_front.png", "top_left.png", "top_right.png", "top_up.png", "white.png"]
    model_source = mujoco.MjSpec()
    pen_spin.add_pen(model_source, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    model = model_source.compile()
    # The textures of the training pen model, written as in its MJCF: which file goes on which face of each block.
    training = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <compiler texturedir={quoteattr(str(textures))}/>
          <asset>
            <texture name="pen_end_a" type="cube" fileright="bottom_right.png" fileleft="bottom_left.png"
                     fileup="bottom_up.png" filedown="bottom_down.png" filefront="white.png"
                     fileback="bottom_back.png"/>
            <texture name="pen_end_b" type="cube" fileright="top_right.png" fileleft="top_left.png"
                     fileup="top_up.png" filedown="top_down.png" filefront="top_front.png" fileback="white.png"/>
          </asset>
        </mujoco>""")

    def pixels(compiled: mujoco.MjModel, texture: int) -> np.ndarray:
        start = compiled.tex_adr[texture]
        size = compiled.tex_height[texture] * compiled.tex_width[texture] * compiled.tex_nchannel[texture]
        return compiled.tex_data[start:start + size]

    shaft = model.geom_matid[model.geom("pen_shaft").id]
    assert shaft >= 0
    assert model.mat_texid[shaft, mujoco.mjtTextureRole.mjTEXROLE_RGB] == -1
    assert model.mat_rgba[shaft] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    for name in ("pen_end_a", "pen_end_b"):
        material = model.geom_matid[model.geom(name).id]
        assert material >= 0
        texture = model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB]
        assert texture >= 0
        assert model.tex_type[texture] == mujoco.mjtTexture.mjTEXTURE_CUBE
        assert np.array_equal(pixels(model, texture), pixels(training, training.texture(name).id))


def test_cached_archive_with_a_wrong_checksum_is_refused(tmp_path):
    path = tmp_path / ARCHIVE
    path.write_bytes(b"not the release")
    with pytest.raises(pen_spin.PenSpinError, match="SHA-256"):
        pen_spin.release_zip(tmp_path)
    assert path.read_bytes() == b"not the release"


def test_download_with_a_wrong_checksum_is_refused_and_not_cached(tmp_path):
    calls = []

    def opener(url, timeout=None):
        calls.append(url)
        return io.BytesIO(b"tampered")

    with pytest.raises(pen_spin.PenSpinError, match="SHA-256"):
        pen_spin.release_zip(tmp_path, opener=opener)
    assert calls == [pen_spin.RELEASE_URL]
    assert list(tmp_path.iterdir()) == []


def test_download_longer_than_the_release_is_refused_and_not_cached(tmp_path, monkeypatch):
    payload = b"release bytes"
    monkeypatch.setattr(pen_spin, "RELEASE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(pen_spin, "RELEASE_BYTES", len(payload) - 1)

    with pytest.raises(pen_spin.PenSpinError, match="exceeds"):
        pen_spin.release_zip(tmp_path, opener=lambda url, timeout=None: io.BytesIO(payload))
    assert list(tmp_path.iterdir()) == []


def test_verified_download_is_cached_and_reused(tmp_path, monkeypatch):
    payload = b"release bytes"
    monkeypatch.setattr(pen_spin, "RELEASE_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def opener(url, timeout=None):
        calls.append(url)
        return io.BytesIO(payload)

    first = pen_spin.release_zip(tmp_path, opener=opener)
    second = pen_spin.release_zip(tmp_path, opener=opener)
    assert first == second == tmp_path / ARCHIVE
    assert first.read_bytes() == payload
    assert len(calls) == 1
    assert [path.name for path in tmp_path.iterdir()] == [ARCHIVE]


def test_published_policy_reproduces_the_recorded_action(gold):
    archive = pen_spin.default_cache_dir() / ARCHIVE
    if not archive.is_file():
        pytest.skip(f"{archive} is not cached; run example 06 once to download it")
    # The recorded action comes from group 01's policy.
    task = pen_spin.load("01")
    policy = pen_spin.Policy(task.model)
    action = policy(gold["reset_obs"])
    assert action.shape == (pen_spin.ACTION_DIM,)
    assert policy.session_runs == 1
    # The training-side recording of this model's output for the reset observation.
    assert np.abs(action - gold["reset_action"]).max() <= 1e-5


def test_default_replay_is_a_multi_turn_clip_and_other_groups_start_at_their_first_clip():
    archive = pen_spin.default_cache_dir() / ARCHIVE
    if not archive.is_file():
        pytest.skip(f"{archive} is not cached; run example 06 once to download it")
    task = pen_spin.load(pen_spin.DEFAULT_GROUP)
    assert task.clip == pen_spin.DEFAULT_CLIP
    axis, spin = pen_spin.reference_spin(task.reference)
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert spin.shape == (task.reference.num_steps,) and spin[0] == 0.0
    assert spin[-1] > 720.0
    assert pen_spin.load("01").clip == "clip_001"


def load_example(simulator: str):
    path = PUBLIC / "examples" / simulator / "06_pen_spin.py"
    module_source = importlib.util.spec_from_file_location(f"{simulator}_06_pen_spin_test", path)
    module = importlib.util.module_from_spec(module_source)
    module_source.loader.exec_module(module)
    return module


@pytest.mark.parametrize("simulator", ["mujoco", "mujoco_warp", "isaaclab"])
def test_examples_refuse_the_left_hand_before_loading_the_policy(simulator, tmp_path, capsys):
    if simulator == "mujoco_warp":
        pytest.importorskip("mujoco_warp")
    flags = ["--hand", "left", "--headless", "--output-dir", str(tmp_path / "out"), "--cache-dir", str(tmp_path / "cache")]
    with pytest.raises(SystemExit):
        load_example(simulator).main(flags if simulator == "mujoco_warp" else ["06_pen_spin.py", *flags])
    assert "--hand right" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() and not (tmp_path / "cache").exists()


def test_pen_log_reports_the_net_rotation_apart_from_the_wobble(gold):
    reference = reference_of(gold)
    wrist = pen_spin.wrist_pose(np.zeros(3))
    start = pen_spin.to_world(reference.pen_pose(0), wrist)

    def turned(deg: float) -> np.ndarray:
        twist, quat = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(twist, np.array([0.0, 0.0, 1.0]), np.radians(deg))
        mujoco.mju_mulQuat(quat, twist, start[3:])
        return np.concatenate([start[:3], quat])

    log = pen_spin.PenLog(reference, wrist)
    log.start(start)
    # The pen swings to +30 deg, back to -30 deg and ends at +10 deg from where it started.
    for step, deg in enumerate((30.0, -30.0, 10.0)):
        log.update(step, turned(deg))
    record = log.record()
    assert record["pen_net_rotation_deg"] == pytest.approx(10.0, abs=1e-6)
    assert record["pen_path_rotation_total_deg"] == pytest.approx(30.0 + 60.0 + 40.0, abs=1e-6)
    frames = [reference.pen_pose(step)[3:] for step in range(reference.num_steps)]
    assert record["reference_net_rotation_deg"] == pytest.approx(float(pen_spin.rotation_deg(frames[0], frames[-1])))
    assert record["reference_path_rotation_total_deg"] >= record["reference_net_rotation_deg"]
    assert not any("turn" in key for key in record)


def test_pen_log_counts_whole_turns_about_the_reference_spin_axis():
    def about_z(deg: float) -> list[float]:
        half = math.radians(deg) / 2
        return [math.cos(half), 0.0, 0.0, math.sin(half)]

    identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    reference = pen_spin.Reference(np.zeros((4, pen_spin.JOINTS)), [identity] * 4,
                                   [[0.0, 0.0, 0.1, *about_z(deg)] for deg in (0.0, 40.0, 80.0, 120.0)],
                                   np.zeros((4, pen_spin.LINKS, 3)))
    log = pen_spin.PenLog(reference, np.array(identity))
    log.start([0.0, 0.0, 0.1, *about_z(0.0)])
    for step, deg in enumerate((120.0, 240.0, 360.0, 370.0)):
        log.update(step, [0.0, 0.0, 0.1, *about_z(deg)])
    record = log.record()
    assert record["spin_axis"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert record["reference_spin_rotation_deg"] == pytest.approx(120.0, abs=1e-6)
    assert record["pen_spin_rotation_deg"] == pytest.approx(370.0, abs=1e-6)
    # The net rotation folds the whole turn away, which is why the replay line reports the spin rotation.
    assert record["pen_net_rotation_deg"] == pytest.approx(10.0, abs=1e-6)
    assert "1.0 turns (370 deg)" in pen_spin.describe_replay({"steps": 4, **record})
