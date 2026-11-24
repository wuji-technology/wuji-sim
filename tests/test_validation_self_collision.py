# SPDX-License-Identifier: Apache-2.0
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from validation import check_self_collision as check
from validation import common
from wuji_sim import assets, conventions

PUBLIC = Path(__file__).resolve().parents[1]
MODULE = "validation.check_self_collision"
SCRIPT = PUBLIC / "validation" / "check_self_collision.py"
HAND = "left"
SIM_TIMEOUT_S = 900.0


def run_check(*flags: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", MODULE, *flags], cwd=PUBLIC, capture_output=True, text=True,
                          timeout=timeout)


def base_flags(description_root: Path, output_dir: Path, simulator: str) -> list[str]:
    return ["--hand", HAND, "--asset-root", str(description_root), "--strict", "--simulator", simulator,
            "--output-dir", str(output_dir)]


def read_record(output_dir: Path) -> tuple[Path, dict]:
    [path] = output_dir.glob("*.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def zero_thresholds(tmp_path: Path) -> Path:
    text = common.THRESHOLDS_PATH.read_text(encoding="utf-8")
    old = 'max_penetration = { value = 0.002, unit = "m" }'
    assert old in text
    path = tmp_path / "thresholds-zero.toml"
    path.write_text(text.replace(old, 'max_penetration = { value = 0.0, unit = "m" }'), encoding="utf-8")
    assert tomllib.loads(path.read_text(encoding="utf-8"))["self_collision"]["push"]["max_penetration"]["value"] == 0.0
    return path


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def assert_consistent_record(result: subprocess.CompletedProcess, output_dir: Path, simulator: str) -> dict:
    assert result.returncode in (common.EXIT_PASS, common.EXIT_FAIL), result.stderr
    path, record = read_record(output_dir)
    assert result.returncode == common.exit_code(record["verdict"])
    assert record["verdict"] == common.record_verdict(record["verdicts"])
    assert f"record: {path}" in result.stdout
    for key in ("record_format", "check", "simulator", "hand", "sim_commit", "backend_versions", "asset", "thresholds",
                "protocol", "values", "verdicts", "verdict", "artifacts"):
        assert key in record
    assert (record["check"], record["simulator"], record["hand"]) == ("self_collision", simulator, HAND)
    assert finite_numbers(record["values"])
    assert [Path(entry["path"]).name for entry in record["artifacts"]] == [f"{path.stem}-push.png"]
    assert all(Path(entry["path"]).is_file() for entry in record["artifacts"])
    pairs = record["values"]["collision_pairs"]
    assert pairs["summary"]["pairs"] == len(pairs["pairs"]) > 0
    push = record["values"]["push"]
    limit = common.criterion(record["thresholds"], "self_collision", "push", "max_penetration", unit="m")
    assert list(push) == [scenario.name for scenario in check.SCENARIOS]
    for name, entry in push.items():
        for item in ("contact_reported", "max_penetration"):
            assert f"push.{name}.{item}" in record["verdicts"]
        assert entry["kinematic_valid"] == (entry["kinematic_penetration_at_target_m"] < -limit)
        if entry["applicable"]:
            assert entry["kinematic_valid"]
            assert entry["mujoco_other_contact_pairs"] == []
            assert entry["max_penetration_m"] >= 0.0
            assert set(entry["final_q"]) == set(entry["push_joints"])
    return record


SMALL_LINKS = ("palm", "a", "b", "b_cover", "c", "d", "e")


def small_spec(*, pair: bool = True) -> mujoco.MjSpec:
    """Overlapping spheres, so every body pair the collision filters leave in touches."""
    spec = mujoco.MjSpec()
    sphere = {"type": mujoco.mjtGeom.mjGEOM_SPHERE, "size": [0.05, 0.0, 0.0]}

    def body(parent, name, *, joint=True, **geom):
        child = parent.add_body(name=name)
        if joint:
            child.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 0.0, 1.0])
        child.add_geom(name=name, **sphere, **geom)
        return child

    palm = body(spec.worldbody, "palm", joint=False)
    a = body(palm, "a")
    b = body(a, "b")
    body(b, "b_cover", joint=False)
    body(b, "c")
    body(palm, "d")
    body(palm, "e", contype=2, conaffinity=2)
    spec.add_exclude(bodyname1="a", bodyname2="d")
    if pair:
        spec.add_pair(geomname1="a", geomname2="d")
    return spec


def contact_pairs(model: mujoco.MjModel) -> set[tuple[str, str]]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {check.pair_key(model.body(int(model.geom_bodyid[g1])).name, model.body(int(model.geom_bodyid[g2])).name)
            for g1, g2 in data.contact.geom[:data.ncon]}


def test_mujoco_rules_name_the_filter_and_agree_with_the_contacts_mujoco_finds():
    model = small_spec().compile()
    rules = check.mujoco_pair_rules(model, SMALL_LINKS)
    assert len(rules) == 21
    assert rules[("b", "b_cover")] == (False, "same_weld")
    assert rules[("a", "b")] == (False, "weld_parent_child")
    assert rules[("b_cover", "c")] == (False, "weld_parent_child")
    assert rules[("a", "d")] == (True, "explicit_pair")
    assert rules[("a", "palm")] == (True, "collidable")
    assert rules[("a", "e")] == (False, "affinity_mask")
    assert {pair for pair, (collidable, _) in rules.items() if collidable} == contact_pairs(model)

    model = small_spec(pair=False).compile()
    rules = check.mujoco_pair_rules(model, SMALL_LINKS)
    assert rules[("a", "d")] == (False, "explicit_exclude")
    assert {pair for pair, (collidable, _) in rules.items() if collidable} == contact_pairs(model)


def test_warp_rules_read_the_filtered_geom_pairs():
    geom_links = ["world", "a", "b", "c", "d"]
    geom_types = [0, 7, 7, 7, 2]
    geom_pairs = [[0, 1], [1, 2], [1, 3], [2, 3], [3, 4], [1, 4]]
    pair_ids = [[-1, -1], [-1, -1], [-2, 0], [0, -1], [-1, -1], [-1, -1]]
    supported = {(7, 7), (0, 7)}
    rules = check.warp_pair_rules(geom_links, geom_types, geom_pairs, pair_ids, supported, ["a", "b", "c", "d"])
    assert rules[("a", "b")] == (True, "warp_supported")
    assert rules[("a", "c")] == (False, "warp_filtered")
    assert rules[("b", "c")] == (True, "warp_supported")
    assert rules[("c", "d")] == (False, "unsupported_type_pair")
    assert rules[("b", "d")] == (False, "warp_filtered")
    assert len(rules) == 6
    with pytest.raises(common.ValidationError):
        check.warp_pair_rules(geom_links, geom_types, geom_pairs, pair_ids[:-1], supported, ["a"])


def test_isaac_rules_apply_in_order():
    links = ["a", "b", "c", "d", "e"]
    has_collider = {"a": True, "b": True, "c": True, "d": True, "e": False}
    pair = frozenset
    rules = check.isaac_pair_rules(links, has_collider, True, {pair(("a", "b"))},
                                   {pair(("a", "b")), pair(("a", "c"))}, {pair(("a", "c")), pair(("b", "c"))})
    assert rules[("a", "b")] == (False, "joint_adjacent")
    assert rules[("a", "c")] == (False, "filtered_pair")
    assert rules[("b", "c")] == (False, "collision_group")
    assert rules[("a", "d")] == (True, "collidable")
    assert rules[("a", "e")] == (False, "no_collider")
    off = check.isaac_pair_rules(links, has_collider, False, set(), set(), set())
    assert off[("a", "d")] == (False, "self_collision_disabled")
    assert off[("d", "e")] == (False, "no_collider")


def test_collision_groups_filter_follows_the_scene_inversion():
    lists_itself = {"env": [({"env"}, False)]}
    assert check.collision_groups_filter({"env"}, {"env"}, lists_itself, scene_inverted=False)
    assert not check.collision_groups_filter({"env"}, {"env"}, lists_itself, scene_inverted=True)
    assert check.collision_groups_filter({"env"}, {"other"}, lists_itself, scene_inverted=True)
    assert not check.collision_groups_filter({"env"}, {"other"}, lists_itself, scene_inverted=False)
    inverted = {"env": [({"env"}, True)]}
    assert check.collision_groups_filter({"env"}, {"other"}, inverted, scene_inverted=False)
    assert not check.collision_groups_filter(set(), {"env"}, inverted, scene_inverted=True)


def test_merged_collision_groups_keep_every_member_filter():
    # One merged member filters x, the other filters everything but x: together they filter both.
    merged = {"m": [({"x"}, False), ({"x"}, True)], "x": [], "y": []}
    assert check.collision_groups_filter({"m"}, {"x"}, merged, scene_inverted=False)
    assert check.collision_groups_filter({"m"}, {"y"}, merged, scene_inverted=False)
    union = {"m": [({"x", "y"}, False)], "x": [], "y": [], "z": []}
    assert check.collision_groups_filter({"m"}, {"y"}, union, scene_inverted=False)
    assert not check.collision_groups_filter({"m"}, {"z"}, union, scene_inverted=False)


def test_pair_filters_cover_the_collision_shapes_below_each_end():
    shapes = ["/hand/a/col0", "/hand/a/col1", "/hand/ab/col", "/hand/b/col", "/hand/c/col"]
    pairs = check.filtered_shape_pairs([("/hand/a/col0", ["/hand/b"]), ("/hand/c", ["/hand/a"])], shapes)
    assert pairs == {frozenset(pair) for pair in [("/hand/a/col0", "/hand/b/col"), ("/hand/c/col", "/hand/a/col0"),
                                                  ("/hand/c/col", "/hand/a/col1")]}


def test_a_link_pair_is_blocked_only_when_no_enabled_shape_pair_is_left():
    shapes_by_link = {"a": ["a0", "a1"], "b": ["b0"], "c": ["c0"], "d": ["d0"]}
    pair = frozenset
    filtered = {pair(("a0", "b0")), pair(("a0", "c0")), pair(("a1", "c0"))}
    groups = {pair(("a1", "b0")), pair(("b0", "d0"))}
    by_pairs, by_groups = check.blocked_link_pairs(shapes_by_link, filtered,
                                                   lambda x, y: pair((x, y)) in groups)
    assert by_pairs == {pair(("a", "c"))}
    assert by_groups == {pair(("a", "b")), pair(("b", "d"))}
    by_pairs, _ = check.blocked_link_pairs({"a": ["a0", "a1"], "b": ["b0"]}, {pair(("a0", "b0"))},
                                           lambda x, y: False)
    assert by_pairs == set()


def test_one_shape_per_link_and_link_level_filters_give_the_link_pairs_of_the_filters():
    shapes_by_link = {name: [f"/hand/{name}/collisions"] for name in ("palm", "thumb", "index", "middle")}
    filters = [("/hand/palm", ["/hand/thumb", "/hand/index"]), ("/hand/thumb", ["/hand/index"])]
    shapes = [shape for link in shapes_by_link.values() for shape in link]
    by_pairs, by_groups = check.blocked_link_pairs(shapes_by_link, check.filtered_shape_pairs(filters, shapes),
                                                   lambda x, y: False)
    assert by_pairs == {frozenset(pair) for pair in [("palm", "thumb"), ("palm", "index"), ("thumb", "index")]}
    assert by_groups == set()


def test_hand_links_are_grouped_by_the_finger_they_are_welded_to(description_root):
    model = check.HandGeometry(description_root / assets.MJCF_ENTRIES[HAND], HAND).model
    links = check.mujoco_links(model, HAND)
    counts = {}
    for entry in links.values():
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    assert counts == {"palm": 9, "thumb": 7, "index_finger": 8, "middle_finger": 8, "ring_finger": 8, "pinky": 8}
    assert links["l_thumb_palm_cover"] == {"category": "thumb", "weld_root": "l_thumb_proximal", "collision_shapes": 1}
    assert links["l_mount"]["category"] == links["l_thenar_cover"]["category"] == "palm"
    assert check.pair_category("palm", "palm") == "palm_palm"
    assert check.pair_category("thumb", "palm") == check.pair_category("palm", "pinky") == "finger_palm"
    assert check.pair_category("thumb", "thumb") == "same_finger"
    assert check.pair_category("thumb", "pinky") == "different_fingers"
    assert check.pair_category("thumb", check.UNKNOWN) == check.UNKNOWN


def test_geom_distance_measures_separations_and_penetrations(description_root):
    geometry = check.HandGeometry(description_root / assets.MJCF_ENTRIES["right"], "right")
    model, data = geometry.model, geometry.data
    data.qpos[model.joint("r_index_finger_mcp_abd").qposadr[0]] = -0.5
    mujoco.mj_kinematics(model, data)
    g1 = geometry.geoms[model.body("r_index_finger_proximal").id][0]
    g2 = geometry.geoms[model.body("r_middle_palm_cover").id][0]
    separation = check.geom_distance(model, data, g1, g2, check.GEOM_DISTANCE_MAX_M)
    assert 0.0 < separation < check.GEOM_DISTANCE_MAX_M
    assert mujoco.mj_geomDistance(model, data, g1, g2, separation - 2e-6, None) == separation - 2e-6
    assert mujoco.mj_geomDistance(model, data, g1, g2, separation + 2e-6, None) < separation + 2e-6

    names = [model.joint(j).name for j in range(model.njnt)]
    scenario = check.SCENARIOS[0]
    data.qpos[:] = 0.0
    for name, value in zip(names, check.target_vector(names, scenario)):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    g1 = geometry.geoms[model.body("r_thumb_tip_sensor_frame").id][0]
    g2 = geometry.geoms[model.body("r_index_finger_tip_sensor_frame").id][0]
    depth = check.geom_distance(model, data, g1, g2, check.GEOM_DISTANCE_MAX_M)
    assert depth < 0.0
    assert depth == mujoco.mj_geomDistance(model, data, g1, g2, 0.0, None)


def test_scenarios_stay_inside_the_joint_ranges_and_apply_alike_to_both_hands(description_root):
    limit = common.criterion(common.load_thresholds(), "self_collision", "push", "max_penetration", unit="m")
    applicable = {}
    for hand in assets.HANDS:
        geometry = check.HandGeometry(description_root / assets.MJCF_ENTRIES[hand], hand)
        model = geometry.model
        names = tuple(conventions.HAND_PREFIXES[hand] + suffix for suffix in conventions.JOINT_SUFFIXES)
        for scenario in check.SCENARIOS:
            target = check.target_vector(names, scenario)
            for name, value in zip(names, target):
                low, high = model.jnt_range[model.joint(name).id]
                assert low < value < high, (scenario.name, name, value)
            pairs = check.scenario_pairs(scenario, geometry.prefix, geometry.categories)
            distance, _ = geometry.distance(names, target, pairs)
            applicable.setdefault(scenario.name, set()).add(distance < -limit)
    assert all(len(flags) == 1 for flags in applicable.values()), applicable
    with pytest.raises(common.ValidationError):
        check.resolve_links("no_such_link", "l_", {"l_wrist": "palm"})


def test_fingertip_palm_pairs_are_the_ones_the_description_lets_collide(description_root):
    for hand in assets.HANDS:
        geometry = check.HandGeometry(description_root / assets.MJCF_ENTRIES[hand], hand)
        rules = check.mujoco_pair_rules(geometry.model, geometry.categories)
        palm = check.resolve_links(check.PALM, geometry.prefix, geometry.categories)
        for tip, links in check.FINGERTIP_PALM_LINKS.items():
            collidable = {link[2:] for link in palm if rules[check.pair_key(geometry.prefix + tip, link)][0]}
            assert collidable == set(links), (hand, tip)


def test_scenario_pairs_resolve_link_names_and_categories(description_root):
    geometry = check.HandGeometry(description_root / assets.MJCF_ENTRIES[HAND], HAND)
    by_name = {scenario.name: scenario for scenario in check.SCENARIOS}
    thumb_index = check.scenario_pairs(by_name["thumb_tip_to_index_tip"], geometry.prefix, geometry.categories)
    assert thumb_index == [(f"l_thumb_{a}", f"l_index_finger_{b}") for a in ("distal", "tip_sensor_frame")
                           for b in ("distal", "tip_sensor_frame")]
    index_middle = check.scenario_pairs(by_name["index_side_to_middle"], geometry.prefix, geometry.categories)
    assert len(index_middle) == 8 * 8
    assert {geometry.categories[a] for a, _ in index_middle} == {"index_finger"}
    assert {geometry.categories[b] for _, b in index_middle} == {"middle_finger"}


def test_contact_log_reports_hold_contacts_penetration_and_other_pairs():
    log = check.ContactLog([("p", "t1"), ("p", "t2")], dt=0.5, steps=4, hold_start=2)
    log.add(0, [("p", "t1", 0.004), ("x", "y", -0.001)])
    log.add(1, [("t1", "p", -0.0005), ("p", "t2", 0.0)])
    log.add(2, [("x", "z", 0.002)])
    summary = log.summary()
    assert not summary["contact_reported"]
    assert summary["first_contact_time_s"] == 0.5
    assert summary["min_separation_m"] == -0.0005
    assert summary["max_penetration_m"] == 0.0005
    assert [(row["link_a"], row["link_b"]) for row in summary["pusher_target_pairs"]] == [("p", "t1"), ("p", "t2")]
    assert [(row["link_a"], row["link_b"]) for row in summary["other_contact_pairs"]] == [("x", "y")]
    np.testing.assert_array_equal(log.min_distance, [0.004, -0.0005, np.nan, np.nan])

    log.add(3, [("p", "t2", -0.001)])
    summary = log.summary()
    assert summary["contact_reported"]
    assert summary["max_penetration_m"] == 0.001

    empty = check.ContactLog([("p", "t")], dt=0.5, steps=2, hold_start=1).summary()
    assert (empty["contact_reported"], empty["first_contact_time_s"], empty["min_separation_m"],
            empty["max_penetration_m"]) == (False, None, None, 0.0)


def test_stop_state_needs_still_positions_short_of_the_target():
    q = np.array([[0.0], [0.5], [0.9], [0.9], [0.9]])
    reported = np.full_like(q, 2.0)
    state = check.stop_state(q, reported, np.array([1.0]), window=2, dt=0.5)
    assert state["stopped"]
    assert (state["final_max_speed_rad_s"], state["final_max_reported_speed_rad_s"]) == (0.0, 2.0)
    assert state["final_offset_rad"] == pytest.approx(0.1)
    assert not check.stop_state(q, reported, np.array([0.905]), window=2, dt=0.5)["stopped"]
    state = check.stop_state(q, reported, np.array([1.0]), window=3, dt=0.5)
    assert not state["stopped"]
    assert state["final_max_speed_rad_s"] == pytest.approx(0.8)


def test_a_scenario_applies_only_when_its_mujoco_run_touches_nothing_else():
    results = {
        "clean": {"kinematic_valid": True, "other_contact_pairs": []},
        "blocked": {"kinematic_valid": True, "other_contact_pairs": [{"link_a": "ring", "link_b": "thumb"}]},
        "short": {"kinematic_valid": False},
    }
    check.settle_applicability(results, None, 0.002)
    assert [results[name]["applicable"] for name in results] == [True, False, False]
    assert results["blocked"]["mujoco_other_contact_pairs"] == [["ring", "thumb"]]
    assert results["short"]["mujoco_other_contact_pairs"] is None

    other = {"clean": {"kinematic_valid": True, "other_contact_pairs": [{"link_a": "x", "link_b": "y"}]},
             "blocked": {"kinematic_valid": True, "other_contact_pairs": []},
             "short": {"kinematic_valid": False}}
    check.settle_applicability(other, results, 0.002)
    assert [other[name]["applicable"] for name in other] == [True, False, False]
    with pytest.raises(common.ValidationError):
        check.settle_applicability({"clean": {"kinematic_valid": True}}, {}, 0.002)


def test_a_push_the_reference_did_not_run_needs_a_new_reference():
    reference = {"deep": {"kinematic_valid": True, "other_contact_pairs": []},
                 "loose": {"kinematic_valid": True, "other_contact_pairs": []},
                 "shallow": {"kinematic_valid": False}}
    replay = {"deep": {"kinematic_valid": True}, "loose": {"kinematic_valid": False},
              "shallow": {"kinematic_valid": True}}
    with pytest.raises(common.ValidationError, match=r"shallow.*\(0\.0 m\).*regenerate the MuJoCo reference"):
        check.settle_applicability(replay, reference, 0.0)

    replay = {"deep": {"kinematic_valid": True}, "loose": {"kinematic_valid": False},
              "shallow": {"kinematic_valid": False}}
    check.settle_applicability(replay, reference, 0.0)
    assert replay["deep"]["applicable"] and replay["deep"]["mujoco_other_contact_pairs"] == []
    assert [replay[name]["applicable"] for name in ("loose", "shallow")] == [False, False]


def test_push_verdicts_need_contact_and_shallow_penetration():
    criteria = {"contact_required": True, "max_penetration": {"value": 0.002, "unit": "m"}}

    def verdicts(**result):
        items = check.push_verdicts("s", {"applicable": True, **result}, criteria)
        return items["push.s.contact_reported"]["verdict"], items["push.s.max_penetration"]["verdict"]

    assert verdicts(contact_reported=True, max_penetration_m=0.002) == ("pass", "pass")
    assert verdicts(contact_reported=True, max_penetration_m=0.0021) == ("pass", "fail")
    assert verdicts(contact_reported=False, max_penetration_m=0.0) == ("fail", "pass")
    items = check.push_verdicts("s", {"applicable": False}, criteria)
    assert {item["verdict"] for item in items.values()} == {"not_applicable"}
    assert common.record_verdict(items) == "pass"
    relaxed = {**criteria, "contact_required": False}
    items = check.push_verdicts("s", {"applicable": True, "contact_reported": False, "max_penetration_m": 0.0},
                                relaxed)
    assert items["push.s.contact_reported"]["verdict"] == "report_only"


def test_comparison_lists_differences_with_both_rules_and_fails_on_unknown_links():
    reference_links = {name: {"category": category, "collision_shapes": 1}
                       for name, category in (("a", "palm"), ("b", "thumb"), ("c", "thumb"))}
    reference_rules = {("a", "b"): (True, "collidable"), ("a", "c"): (False, "explicit_exclude"),
                       ("b", "c"): (False, "weld_parent_child")}
    links = {"a": {"collision_shapes": 1}, "b": {"collision_shapes": 1}, "c": {"collision_shapes": 1},
             "x": {"collision_shapes": 0}}
    rules = {("a", "b"): (True, "collidable"), ("a", "c"): (False, "filtered_pair"), ("b", "c"): (True, "collidable")}
    comparison = check.compare_pairs("isaaclab", reference_links, reference_rules, links, rules)
    assert comparison["consistent"] is False
    assert comparison["differences"] == 1
    assert comparison["differences_by_category"]["same_finger"] == 1
    assert comparison["difference_list"] == [{
        "link_a": "b", "link_b": "c", "category": "same_finger",
        "mujoco": {"collidable": False, "rule": "weld_parent_child"},
        "isaaclab": {"collidable": True, "rule": "collidable"},
    }]
    assert comparison["unclassified_pairs"] == []
    assert check.pair_verdict(comparison)["verdict"] == "pass"

    links["y"] = {"collision_shapes": 1}
    rules.update({("a", "y"): (True, "collidable"), ("b", "y"): (True, "collidable"), ("c", "y"): (True, "collidable")})
    comparison = check.compare_pairs("isaaclab", reference_links, reference_rules, links, rules)
    assert comparison["unknown_links"] == {"mujoco": ["y"], "isaaclab": []}
    assert len(comparison["unclassified_pairs"]) == 3
    assert check.pair_verdict(comparison) == {"value": 3, "threshold": 0, "unit": "pairs", "verdict": "fail"}
    assert check.pair_verdict(None)["verdict"] == "reference"


def test_reference_flag_is_checked_before_anything_runs(tmp_path, description_root):
    result = run_check(*base_flags(description_root, tmp_path / "out", "mujoco_warp"))
    assert result.returncode == common.EXIT_ERROR
    assert "--reference" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def mujoco_reference(tmp_path_factory, description_root) -> tuple[subprocess.CompletedProcess, Path]:
    output_dir = tmp_path_factory.mktemp("self-collision-mujoco")
    result = run_check(*base_flags(description_root, output_dir, "mujoco"))
    return result, output_dir


@pytest.fixture(scope="module")
def mujoco_zero_reference(tmp_path_factory, description_root) -> tuple[subprocess.CompletedProcess, Path, Path]:
    tmp_path = tmp_path_factory.mktemp("self-collision-mujoco-zero")
    thresholds = zero_thresholds(tmp_path)
    output_dir = tmp_path / "zero"
    result = run_check(*base_flags(description_root, output_dir, "mujoco"), "--thresholds", str(thresholds))
    return result, output_dir, thresholds


def test_mujoco_run_writes_the_reference_pair_table_and_push_results(mujoco_reference):
    result, output_dir = mujoco_reference
    record = assert_consistent_record(result, output_dir, "mujoco")
    assert record["verdicts"]["collision_pairs.unclassified"]["verdict"] == "reference"
    pairs = record["values"]["collision_pairs"]
    assert pairs["comparison"] is None
    assert len(pairs["links"]) == 48
    assert pairs["summary"]["pairs"] == 48 * 47 // 2
    assert pairs["model"]["nexclude"] > 0
    assert set(pairs["summary"]["by_rule"]) <= set(check.MUJOCO_RULES)


def test_mujoco_run_with_zero_thresholds_exits_1(mujoco_zero_reference):
    result, output_dir, _ = mujoco_zero_reference
    assert result.returncode == common.EXIT_FAIL, result.stderr
    _, record = read_record(output_dir)
    assert record["verdict"] == "fail"


THUMB_INDEX_TIPS = tuple((thumb, index) for thumb in ("thumb_distal", "thumb_tip_sensor_frame")
                         for index in ("index_finger_distal", "index_finger_tip_sensor_frame"))


def test_mujoco_run_fails_when_the_model_lets_the_thumb_pass_through_the_index_tip(tmp_path, description_root,
                                                                                    monkeypatch):
    from wuji_sim.platforms import mujoco as mujoco_platform

    prefix = conventions.HAND_PREFIXES[HAND]
    load = mujoco_platform.load

    def load_with_excludes(mjcf, *, scene=None, **kwargs):
        def add_scene_and_excludes(spec):
            scene(spec)
            for thumb, index in THUMB_INDEX_TIPS:
                spec.add_exclude(bodyname1=prefix + thumb, bodyname2=prefix + index)

        return load(mjcf, scene=add_scene_and_excludes, **kwargs)

    monkeypatch.setattr(mujoco_platform, "load", load_with_excludes)
    output_dir = tmp_path / "excluded"
    assert check.main([str(SCRIPT), *base_flags(description_root, output_dir, "mujoco")]) == common.EXIT_FAIL
    _, record = read_record(output_dir)
    rows = {(row["link_a"], row["link_b"]): row for row in record["values"]["collision_pairs"]["pairs"]}
    for thumb, index in THUMB_INDEX_TIPS:
        assert rows[check.pair_key(prefix + thumb, prefix + index)]["rule"] == "explicit_exclude"
    push = record["values"]["push"]["thumb_tip_to_index_tip"]
    assert push["applicable"] and not push["contact_reported"]
    assert record["verdicts"]["push.thumb_tip_to_index_tip.contact_reported"]["verdict"] == "fail"
    assert record["verdict"] == "fail"


@pytest.mark.gpu
def test_warp_pair_rules_match_mujoco_on_a_small_model():
    mjw = pytest.importorskip("mujoco_warp")
    from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE

    for pair in (True, False):
        model = small_spec(pair=pair).compile()
        m = mjw.put_model(model)
        rules = check.warp_pair_rules([model.body(int(body)).name for body in model.geom_bodyid], model.geom_type,
                                      m.nxn_geom_pair_filtered.numpy(), m.nxn_pairid_filtered.numpy(),
                                      {tuple(sorted((int(a), int(b)))) for a, b in MJ_COLLISION_TABLE}, SMALL_LINKS)
        expected = check.mujoco_pair_rules(model, SMALL_LINKS)
        assert {key: collidable for key, (collidable, _) in rules.items()} == {
            key: collidable for key, (collidable, _) in expected.items()
        }


@pytest.mark.gpu
def test_warp_run_matches_the_mujoco_pair_table(tmp_path, description_root, mujoco_reference, mujoco_zero_reference):
    pytest.importorskip("mujoco_warp")
    reference_path, _ = read_record(mujoco_reference[1])
    output_dir = tmp_path / "warp"
    result = run_check(*base_flags(description_root, output_dir, "mujoco_warp"), "--reference", str(reference_path),
                       timeout=SIM_TIMEOUT_S)
    record = assert_consistent_record(result, output_dir, "mujoco_warp")
    comparison = record["values"]["collision_pairs"]["comparison"]
    assert comparison["consistent"] is True
    assert comparison["compared_pairs"] == 48 * 47 // 2
    assert record["reference"]["path"] == str(reference_path)

    _, zero_reference_dir, zero = mujoco_zero_reference
    stale_dir = tmp_path / "warp-zero-stale"
    result = run_check(*base_flags(description_root, stale_dir, "mujoco_warp"), "--reference", str(reference_path),
                       "--thresholds", str(zero), timeout=SIM_TIMEOUT_S)
    assert result.returncode == common.EXIT_ERROR, result.stderr
    assert "regenerate the MuJoCo reference with the same thresholds" in result.stderr
    assert list(stale_dir.glob("*.json")) == []

    zero_dir = tmp_path / "warp-zero"
    result = run_check(*base_flags(description_root, zero_dir, "mujoco_warp"), "--reference",
                       str(read_record(zero_reference_dir)[0]), "--thresholds", str(zero), timeout=SIM_TIMEOUT_S)
    assert result.returncode == common.EXIT_FAIL, result.stderr
    assert read_record(zero_dir)[1]["verdict"] == "fail"


@pytest.mark.isaac
def test_isaac_run_lists_its_differences_and_exits_with_the_verdict(tmp_path, description_root, mujoco_reference,
                                                                     mujoco_zero_reference):
    pytest.importorskip("isaaclab")
    reference_path, _ = read_record(mujoco_reference[1])
    output_dir = tmp_path / "isaac"
    result = run_check(*base_flags(description_root, output_dir, "isaaclab"), "--reference", str(reference_path),
                       timeout=SIM_TIMEOUT_S)
    record = assert_consistent_record(result, output_dir, "isaaclab")
    pairs = record["values"]["collision_pairs"]
    comparison = pairs["comparison"]
    assert comparison["unclassified_pairs"] == []
    assert comparison["differences"] == len(comparison["difference_list"])
    for entry in comparison["difference_list"]:
        assert entry["mujoco"]["rule"] in check.MUJOCO_RULES
        assert entry["isaaclab"]["rule"] in check.ISAAC_RULES
    assert set(pairs["stage"]["enabled_self_collisions"].values()) == {True}
    assert "joint_adjacent" in record["protocol"]["collision_pairs"]["rule_source"]
    assert any(entry["contact_reported"] for entry in record["values"]["push"].values() if entry["applicable"])

    _, zero_reference_dir, zero = mujoco_zero_reference
    zero_dir = tmp_path / "isaac-zero"
    result = run_check(*base_flags(description_root, zero_dir, "isaaclab"), "--reference",
                       str(read_record(zero_reference_dir)[0]), "--thresholds", str(zero), timeout=SIM_TIMEOUT_S)
    assert result.returncode == common.EXIT_FAIL, result.stderr
    assert read_record(zero_dir)[1]["verdict"] == "fail"


# Run from a file: Isaac Sim crashes (SIGSEGV, no output) when this source is passed through python -c.
SELF_COLLISIONS_OFF = """\
import os
import runpy
import sys

from wuji_sim.platforms import isaaclab as platform

# As python -m would, so the validation package in the working directory is importable.
sys.path.insert(0, os.getcwd())

load = platform.load


def load_without_self_collisions(*args, **kwargs):
    from isaaclab.sim.utils.stage import get_current_stage

    loaded = load(*args, **kwargs)
    root = get_current_stage().GetPrimAtPath(loaded.report["self_collision_readback"]["prim_path"])
    root.GetAttribute("physxArticulation:enabledSelfCollisions").Set(False)
    loaded.sim.stop()
    loaded.sim.reset()
    return loaded


platform.load = load_without_self_collisions
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
"""


@pytest.mark.isaac
def test_isaac_run_fails_when_self_collisions_are_switched_off(tmp_path, description_root, mujoco_reference):
    pytest.importorskip("isaaclab")
    reference_path, _ = read_record(mujoco_reference[1])
    wrapper = tmp_path / "self_collisions_off.py"
    wrapper.write_text(SELF_COLLISIONS_OFF, encoding="utf-8")
    output_dir = tmp_path / "isaac-off"
    result = subprocess.run([sys.executable, str(wrapper), MODULE,
                             *base_flags(description_root, output_dir, "isaaclab"),
                             "--reference", str(reference_path)],
                            cwd=PUBLIC, capture_output=True, text=True, timeout=SIM_TIMEOUT_S)
    record = assert_consistent_record(result, output_dir, "isaaclab")
    assert result.returncode == common.EXIT_FAIL
    pairs = record["values"]["collision_pairs"]
    assert set(pairs["stage"]["enabled_self_collisions"].values()) == {False}
    assert pairs["summary"]["collidable"] == 0
    comparison = pairs["comparison"]
    assert comparison["consistent"] is False
    assert {entry["isaaclab"]["rule"] for entry in comparison["difference_list"]} == {"self_collision_disabled"}
    for name in ("thumb_tip_to_index_tip", "index_side_to_middle"):
        assert record["values"]["push"][name]["applicable"]
        assert record["verdicts"][f"push.{name}.contact_reported"]["verdict"] == "fail"
