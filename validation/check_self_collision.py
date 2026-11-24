# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from validation import common
from wuji_sim import assets, conventions, framing, mit, profiles, scene, scene_isaaclab

CHECK = "self_collision"
VIEW_KIND = "tracking"
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
PALM = "palm"
CATEGORIES = ("same_finger", "different_fingers", "finger_palm", "palm_palm")
UNKNOWN = "unknown"
SCENE_BODY_PREFIX = "scene_"

MUJOCO_RULES = ("explicit_pair", "no_collision_shape", "same_weld", "weld_parent_child", "explicit_exclude",
                "affinity_mask", "collidable")
WARP_RULES = ("warp_filtered", "unsupported_type_pair", "warp_supported")
ISAAC_RULES = ("no_collider", "self_collision_disabled", "joint_adjacent", "filtered_pair", "collision_group",
               "collidable")
COLLIDABLE_RULES = frozenset({"explicit_pair", "collidable", "warp_supported"})
ISAAC_RULE_SOURCES = {
    "joint_adjacent": (
        "documented, not measured here: with physxArticulation:enabledSelfCollisions on, PhysX always filters "
        "collisions between parent and child articulation links (Omni Physics, Articulation and Robot Simulation "
        "Stability Guide, Articulation Link Self-Collision: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/"
        "latest/dev_guide/guides/articulation_stability_guide.html#articulation-link-self-collision); a joint "
        "excluded from the articulation keeps its bodies apart only with physics:collisionEnabled false"
    ),
    "filtered_pair": (
        "UsdPhysics.FilteredPairsAPI: a prim's physics:filteredPairs disables every collision shape at or below it "
        "against every one at or below each target; a link pair is filtered when no pair of their enabled shapes "
        "is left"
    ),
    "collision_group": (
        "UsdPhysics.CollisionGroup: a group filters the groups listed in physics:filteredGroups (all others with "
        "physics:invertFilteredGroups); physxScene:invertCollisionGroupFilter turns the listed groups into the only "
        "ones that collide; a collider in no group is never filtered by groups; groups sharing physics:mergeGroup "
        "pool their colliders and each keeps its own filter, so a shape pair is filtered when any of them filters "
        "it; a link pair is filtered when every pair of their enabled shapes is filtered by a group or a pair "
        "filter and at least one by a group"
    ),
}

PUSH_KP = 5.0  # N*m/rad
PUSH_KD = 0.05  # N*m*s/rad
PUSH_MOVE_S = 1.0
PUSH_HOLD_S = 1.0
GEOM_DISTANCE_MAX_M = 0.05
GEOM_DISTANCE_RESOLUTION_M = 1e-6
STOP_WINDOW_S = 0.2
STOP_SPEED_RAD_S = 0.05
STOP_OFFSET_RAD = 0.01
ISAAC_MAX_CONTACTS = 16384

THUMB_JOINTS = ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip")
FLEXION_JOINTS = ("mcp_flex", "pip", "dip")


@dataclass(frozen=True)
class Scenario:
    name: str
    # Link suffixes or categories (a finger, PALM); fixed here, so a collision filter dropping one fails the check.
    pairs: tuple[tuple[str, str], ...]
    push_joints: tuple[str, ...]
    # Joint suffix -> rad; every other joint stays at the open pose (0 rad).
    target: Mapping[str, float]


def fingertip_pairs(finger: str, palm_links: Mapping[str, tuple[str, ...]]) -> tuple[tuple[str, str], ...]:
    return tuple((tip, link) for tip in (f"{finger}_distal", f"{finger}_tip_sensor_frame") for link in palm_links[tip])


# The palm links each fingertip collides with in the hand's description MJCF (nearest palm covers are excluded).
FINGERTIP_PALM_LINKS = {
    "thumb_distal": ("adductor_pollicis_cover", "hypothenar_cover", "midpalm_cover", "mount", "thenar_cover", "wrist"),
    "thumb_tip_sensor_frame": ("adductor_pollicis_cover", "hypothenar_cover", "mount", "palm_index_cover",
                               "thenar_cover", "wrist"),
    "index_finger_distal": ("adductor_pollicis_cover", "hypothenar_cover", "mount", "palm_ring_cover", "wrist"),
    "index_finger_tip_sensor_frame": ("adductor_pollicis_cover", "hypothenar_cover", "mount", "palm_ring_cover",
                                      "wrist"),
    "middle_finger_distal": ("adductor_pollicis_cover", "mount", "palm_index_cover", "thenar_cover", "wrist"),
    "middle_finger_tip_sensor_frame": ("adductor_pollicis_cover", "hypothenar_cover", "mount", "palm_index_cover",
                                       "thenar_cover", "wrist"),
    "ring_finger_distal": ("adductor_pollicis_cover", "mount", "palm_index_cover", "palm_middle_cover",
                           "thenar_cover", "wrist"),
    "ring_finger_tip_sensor_frame": ("adductor_pollicis_cover", "mount", "palm_index_cover", "palm_middle_cover",
                                     "thenar_cover", "wrist"),
    "pinky_distal": ("adductor_pollicis_cover", "midpalm_cover", "mount", "palm_index_cover", "palm_middle_cover",
                     "palm_ring_cover", "thenar_cover", "wrist"),
    "pinky_tip_sensor_frame": ("adductor_pollicis_cover", "midpalm_cover", "mount", "palm_index_cover",
                               "palm_middle_cover", "palm_ring_cover", "thenar_cover", "wrist"),
}
# One set of values serves both hands (the hand model mirrors the left-hand joint axes).
FINGERTIP_DIP = {"index_finger": 1.25, "middle_finger": 0.56, "ring_finger": 1.22, "pinky": 1.45}
SCENARIOS = (
    # Pinch with thumb_cmc_abd moved toward the index; more flex from pinch slides the thumb tip past the index tip.
    Scenario(
        "thumb_tip_to_index_tip",
        pairs=tuple((thumb, index) for thumb in ("thumb_distal", "thumb_tip_sensor_frame")
                    for index in ("index_finger_distal", "index_finger_tip_sensor_frame")),
        push_joints=THUMB_JOINTS,
        target={"thumb_cmc_flex": 0.96, "thumb_cmc_abd": -0.35, "thumb_mcp": 0.34, "thumb_ip": 0.31,
                "index_finger_mcp_flex": 0.7, "index_finger_pip": 0.87, "index_finger_dip": 0.64},
    ),
    Scenario(
        "index_side_to_middle",
        pairs=(("index_finger", "middle_finger"),),
        push_joints=("index_finger_mcp_abd",),
        target={"index_finger_mcp_abd": 0.32},
    ),
    # Grid-picked so the thumb pad presses mid-palm with no other thumb link pair within 2 mm along the joint path.
    Scenario(
        "fingertip_to_palm_thumb",
        pairs=fingertip_pairs("thumb", FINGERTIP_PALM_LINKS),
        push_joints=THUMB_JOINTS,
        target={"thumb_cmc_flex": 1.04, "thumb_cmc_abd": -0.22, "thumb_mcp": 1.51766, "thumb_ip": 1.51766},
    ),
    # mcp_flex, pip 2 % inside the upper limit; dip where the tip goes deepest, since a fully bent dip curls it away.
    *(
        Scenario(
            f"fingertip_to_palm_{finger}",
            pairs=fingertip_pairs(finger, FINGERTIP_PALM_LINKS),
            push_joints=tuple(f"{finger}_{joint}" for joint in FLEXION_JOINTS),
            target={f"{finger}_mcp_flex": 1.51766, f"{finger}_pip": 2.03118, f"{finger}_dip": dip},
        )
        for finger, dip in FINGERTIP_DIP.items()
    ),
)


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def all_pairs(links: Iterable[str]) -> list[tuple[str, str]]:
    names = sorted(set(links))
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def pair_category(category_a: str, category_b: str) -> str:
    if UNKNOWN in (category_a, category_b):
        return UNKNOWN
    if category_a == PALM and category_b == PALM:
        return "palm_palm"
    if PALM in (category_a, category_b):
        return "finger_palm"
    return "same_finger" if category_a == category_b else "different_fingers"


def collision_geoms(model: mujoco.MjModel) -> dict[int, list[int]]:
    geoms: dict[int, list[int]] = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            geoms.setdefault(int(model.geom_bodyid[g]), []).append(g)
    return geoms


def mujoco_links(model: mujoco.MjModel, hand: str) -> dict[str, dict]:
    """Every hand body (not world, not scene_*) with its category, weld root and number of collision geoms."""
    prefix = conventions.HAND_PREFIXES[hand]
    geoms = collision_geoms(model)
    links = {}
    for body in range(1, model.nbody):
        name = model.body(body).name
        if name.startswith(SCENE_BODY_PREFIX):
            continue
        weld = int(model.body_weldid[body])
        root = model.body(weld).name
        if weld == 0:
            category = PALM
        else:
            category = next((finger for finger in FINGERS if root.startswith(f"{prefix}{finger}_")), None)
            if category is None:
                raise common.ValidationError(f"body {name!r} is welded to {root!r}, which is not a joint body of "
                                             f"one of the fingers {FINGERS}")
        links[name] = {"category": category, "weld_root": root, "collision_shapes": len(geoms.get(body, []))}
    return links


def mujoco_pair_rules(model: mujoco.MjModel, links: Iterable[str]) -> dict[tuple[str, str], tuple[bool, str]]:
    geoms = collision_geoms(model)
    filterparent = not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    excluded = {int(signature) for signature in model.exclude_signature}
    explicit = {frozenset((int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])))
                for g1, g2 in zip(model.pair_geom1, model.pair_geom2)}
    rules = {}
    for a, b in all_pairs(links):
        id_a, id_b = sorted((model.body(a).id, model.body(b).id))
        weld_a, weld_b = int(model.body_weldid[id_a]), int(model.body_weldid[id_b])
        weld_parent_a = int(model.body_weldid[model.body_parentid[weld_a]])
        weld_parent_b = int(model.body_weldid[model.body_parentid[weld_b]])
        # Explicit <pair> elements are collided whatever the filters below say.
        if frozenset((id_a, id_b)) in explicit:
            rule = "explicit_pair"
        elif id_a not in geoms or id_b not in geoms:
            rule = "no_collision_shape"
        elif weld_a == weld_b:
            rule = "same_weld"
        elif filterparent and weld_a and weld_b and (weld_a == weld_parent_b or weld_b == weld_parent_a):
            rule = "weld_parent_child"
        elif (id_a << 16) + id_b in excluded:
            rule = "explicit_exclude"
        elif not any(model.geom_contype[g1] & model.geom_conaffinity[g2]
                     or model.geom_contype[g2] & model.geom_conaffinity[g1]
                     for g1 in geoms[id_a] for g2 in geoms[id_b]):
            rule = "affinity_mask"
        else:
            rule = "collidable"
        rules[(a, b)] = (rule in COLLIDABLE_RULES, rule)
    return rules


def warp_pair_rules(geom_links: Sequence[str], geom_types: Sequence[int], geom_pairs, pair_ids,
                    supported: set[tuple[int, int]], links: Iterable[str]) -> dict[tuple[str, str], tuple[bool, str]]:
    links = sorted(set(links))
    geom_pairs = np.asarray(geom_pairs, dtype=np.int64).reshape(-1, 2)
    pair_ids = np.asarray(pair_ids, dtype=np.int64).reshape(-1, 2)
    if len(geom_pairs) != len(pair_ids):
        raise common.ValidationError(f"{len(geom_pairs)} filtered geom pairs but {len(pair_ids)} pair ids")
    names = set(links)
    candidates: dict[tuple[str, str], bool] = {}
    # Contact pair id -1: left in by the collision filters; >= 0: an explicit <pair>; -2: kept only for a sensor.
    for (g1, g2), contact_id in zip(geom_pairs, pair_ids[:, 0]):
        a, b = geom_links[g1], geom_links[g2]
        if contact_id < -1 or a == b or a not in names or b not in names:
            continue
        key = pair_key(a, b)
        types = tuple(sorted((int(geom_types[g1]), int(geom_types[g2]))))
        candidates[key] = candidates.get(key, False) or types in supported
    rules = {}
    for pair in all_pairs(links):
        if pair not in candidates:
            rules[pair] = (False, "warp_filtered")
        else:
            rules[pair] = (True, "warp_supported") if candidates[pair] else (False, "unsupported_type_pair")
    return rules


def isaac_pair_rules(links: Iterable[str], has_collider: Mapping[str, bool], self_collisions: bool,
                     adjacent: set[frozenset], filtered: set[frozenset],
                     group_filtered: set[frozenset]) -> dict[tuple[str, str], tuple[bool, str]]:
    rules = {}
    for a, b in all_pairs(links):
        key = frozenset((a, b))
        if not (has_collider[a] and has_collider[b]):
            rule = "no_collider"
        elif not self_collisions:
            rule = "self_collision_disabled"
        elif key in adjacent:
            rule = "joint_adjacent"
        elif key in filtered:
            rule = "filtered_pair"
        elif key in group_filtered:
            rule = "collision_group"
        else:
            rule = "collidable"
        rules[(a, b)] = (rule in COLLIDABLE_RULES, rule)
    return rules


def collision_groups_filter(groups_a: set[str], groups_b: set[str],
                            group_rules: Mapping[str, Sequence[tuple[set[str], bool]]], scene_inverted: bool) -> bool:
    """group_rules: per (merged) group, the (filtered groups, invertFilteredGroups) of each group merged into it."""
    if not groups_a or not groups_b:
        return False

    def lists(group: str, others: set[str]) -> bool:
        return any(bool(others & filtered) != invert for filtered, invert in group_rules.get(group, ()))

    listed = any(lists(group, groups_b) for group in groups_a) or any(lists(group, groups_a) for group in groups_b)
    return not listed if scene_inverted else listed


def under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def filtered_shape_pairs(filters: Iterable[tuple[str, Sequence[str]]], shapes: Iterable[str]) -> set[frozenset]:
    """filters: (path of a FilteredPairsAPI prim, its filteredPairs targets); shapes: collision shape paths."""
    shapes = list(shapes)
    pairs = set()
    for source, targets in filters:
        sources = [shape for shape in shapes if under(shape, source)]
        for target in targets:
            pairs |= {frozenset((a, b)) for a in sources for b in shapes if under(b, target) and a != b}
    return pairs


def blocked_link_pairs(shapes_by_link: Mapping[str, Sequence[str]], filtered_shapes: set[frozenset],
                       group_filtered: Callable[[str, str], bool]) -> tuple[set[frozenset], set[frozenset]]:
    """Link pairs no enabled shape pair of which can collide: by pair filters alone, and with a collision group."""
    by_pairs, by_groups = set(), set()
    for a, b in all_pairs(shapes_by_link):
        shape_pairs = [(sa, sb) for sa in shapes_by_link[a] for sb in shapes_by_link[b]]
        paired = [frozenset(pair) in filtered_shapes for pair in shape_pairs]
        grouped = [group_filtered(sa, sb) for sa, sb in shape_pairs]
        if all(paired):
            by_pairs.add(frozenset((a, b)))
        if any(grouped) and all(p or g for p, g in zip(paired, grouped)):
            by_groups.add(frozenset((a, b)))
    return by_pairs, by_groups


def pair_rows(rules: Mapping[tuple[str, str], tuple[bool, str]], categories: Mapping[str, str]) -> list[dict]:
    return [
        {"link_a": a, "link_b": b,
         "category": pair_category(categories.get(a, UNKNOWN), categories.get(b, UNKNOWN)),
         "collidable": collidable, "rule": rule}
        for (a, b), (collidable, rule) in sorted(rules.items())
    ]


def pair_summary(rows: Sequence[Mapping]) -> dict:
    by_rule: dict[str, int] = {}
    collidable_by_category = dict.fromkeys((*CATEGORIES, UNKNOWN), 0)
    for row in rows:
        by_rule[row["rule"]] = by_rule.get(row["rule"], 0) + 1
        collidable_by_category[row["category"]] += row["collidable"]
    return {
        "pairs": len(rows),
        "collidable": sum(row["collidable"] for row in rows),
        "collidable_by_category": collidable_by_category,
        "by_rule": dict(sorted(by_rule.items())),
    }


def reference_pairs(record: Mapping) -> tuple[dict[str, dict], dict[tuple[str, str], tuple[bool, str]]]:
    try:
        table = record["values"]["collision_pairs"]
        links = {name: {"category": str(entry["category"]), "collision_shapes": int(entry["collision_shapes"])}
                 for name, entry in table["links"].items()}
        rules = {pair_key(row["link_a"], row["link_b"]): (bool(row["collidable"]), str(row["rule"]))
                 for row in table["pairs"]}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise common.ValidationError(f"reference record has no readable values.collision_pairs table: {exc!r}") from exc
    return links, rules


def reference_push(record: Mapping) -> Mapping:
    push = record.get("values", {}).get("push") if isinstance(record.get("values"), Mapping) else None
    if not isinstance(push, Mapping):
        raise common.ValidationError("reference record has no values.push table")
    return push


def compared_links(reference_links: Mapping[str, Mapping], links: Mapping[str, Mapping]) -> list[str]:
    """Links with a collision shape in either simulator: the pairs of these are compared."""
    return sorted({name for table in (reference_links, links) for name, entry in table.items()
                   if entry["collision_shapes"] > 0})


def compare_pairs(simulator: str, reference_links: Mapping[str, Mapping],
                  reference_rules: Mapping[tuple[str, str], tuple[bool, str]], links: Mapping[str, Mapping],
                  rules: Mapping[tuple[str, str], tuple[bool, str]]) -> dict:
    reference = common.REFERENCE_SIMULATOR
    compared = compared_links(reference_links, links)
    differences, unclassified = [], []
    for a, b in all_pairs(compared):
        category = pair_category(reference_links.get(a, {}).get("category", UNKNOWN),
                                 reference_links.get(b, {}).get("category", UNKNOWN))
        reference_rule, rule = reference_rules.get((a, b)), rules.get((a, b))
        entry = {
            "link_a": a, "link_b": b, "category": category,
            reference: None if reference_rule is None else {"collidable": reference_rule[0], "rule": reference_rule[1]},
            simulator: None if rule is None else {"collidable": rule[0], "rule": rule[1]},
        }
        if reference_rule is None or rule is None or category == UNKNOWN:
            unclassified.append(entry)
        elif reference_rule[0] != rule[0]:
            differences.append(entry)
    by_category = dict.fromkeys(CATEGORIES, 0)
    for entry in differences:
        by_category[entry["category"]] += 1
    rule_pairs: dict[str, int] = {}
    for entry in differences:
        key = f"{entry[reference]['rule']} / {entry[simulator]['rule']}"
        rule_pairs[key] = rule_pairs.get(key, 0) + 1
    return {
        "compared_links": len(compared),
        "compared_pairs": len(compared) * (len(compared) - 1) // 2,
        "collidable": {
            reference: sum(reference_rules.get(pair, (False, ""))[0] for pair in all_pairs(compared)),
            simulator: sum(rules.get(pair, (False, ""))[0] for pair in all_pairs(compared)),
        },
        "unknown_links": {reference: sorted(set(compared) - set(reference_links)),
                          simulator: sorted(set(compared) - set(links))},
        "differences": len(differences),
        "differences_by_category": by_category,
        "differences_by_rules": dict(sorted(rule_pairs.items())),
        "only_collidable_in": {
            reference: sum(entry[reference]["collidable"] for entry in differences),
            simulator: sum(entry[simulator]["collidable"] for entry in differences),
        },
        "consistent": not differences and not unclassified,
        "unclassified_pairs": unclassified,
        "difference_list": differences,
    }


def resolve_links(selector: str, prefix: str, categories: Mapping[str, str]) -> list[str]:
    if selector in (*FINGERS, PALM):
        links = sorted(name for name, category in categories.items() if category == selector)
    else:
        links = [prefix + selector]
    missing = [name for name in links if name not in categories]
    if not links or missing:
        raise common.ValidationError(f"links {selector!r} resolve to {links}, missing from the model: {missing}")
    return links


def scenario_pairs(scenario: Scenario, prefix: str, categories: Mapping[str, str]) -> list[tuple[str, str]]:
    return sorted({(pusher, target) for pushers, targets in scenario.pairs
                   for pusher in resolve_links(pushers, prefix, categories)
                   for target in resolve_links(targets, prefix, categories)})


def target_vector(joint_names: Sequence[str], scenario: Scenario) -> np.ndarray:
    suffixes = [name[2:] for name in joint_names]
    unknown = sorted(set(scenario.target) - set(suffixes))
    if unknown:
        raise common.ValidationError(f"scenario {scenario.name} names unknown joints {unknown}")
    return np.array([float(scenario.target.get(suffix, 0.0)) for suffix in suffixes])


def geom_distance(model: mujoco.MjModel, data: mujoco.MjData, g1: int, g2: int, cap: float) -> float:
    """Signed distance between two geoms (negative: penetration depth), capped at cap."""
    depth = mujoco.mj_geomDistance(model, data, g1, g2, 0.0, None)
    if depth < 0.0:
        return float(depth)
    # mj_geomDistance may return 0.0 between meshes above the separation, but whether it finds a contact is reliable.
    if not mujoco.mj_geomDistance(model, data, g1, g2, cap, None) < cap:
        return cap
    low, high = 0.0, cap
    while high - low > GEOM_DISTANCE_RESOLUTION_M:
        middle = 0.5 * (low + high)
        if mujoco.mj_geomDistance(model, data, g1, g2, middle, None) < middle:
            high = middle
        else:
            low = middle
    return high


class HandGeometry:
    """The description MJCF alone, to measure how far the pushers would sink into the targets if nothing collided."""

    def __init__(self, mjcf, hand: str):
        self.model = mujoco.MjSpec.from_file(str(mjcf)).compile()
        self.data = mujoco.MjData(self.model)
        self.prefix = conventions.HAND_PREFIXES[hand]
        self.categories = {name: entry["category"] for name, entry in mujoco_links(self.model, hand).items()}
        self.geoms = collision_geoms(self.model)

    def distance(self, joint_names: Sequence[str], q: np.ndarray,
                 pairs: Iterable[tuple[str, str]]) -> tuple[float, list[str] | None]:
        self.data.qpos[:] = self.model.qpos0
        for name, value in zip(joint_names, q):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        mujoco.mj_kinematics(self.model, self.data)
        best, closest = GEOM_DISTANCE_MAX_M, None
        for pusher, target in pairs:
            for g1 in self.geoms.get(self.model.body(pusher).id, []):
                for g2 in self.geoms.get(self.model.body(target).id, []):
                    distance = geom_distance(self.model, self.data, g1, g2, GEOM_DISTANCE_MAX_M)
                    if distance < best:
                        best, closest = distance, [pusher, target]
        return best, closest


class ContactLog:
    def __init__(self, expected: Iterable[tuple[str, str]], dt: float, steps: int, hold_start: int):
        self.expected = {frozenset(pair) for pair in expected}
        self.dt = dt
        self.hold_start = hold_start
        self.min_distance = np.full(steps, np.nan)
        self.pairs: dict[tuple[str, str], dict] = {}

    def is_push(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.expected

    def add(self, step: int, contacts: Iterable[tuple[str, str, float]]) -> None:
        for a, b, distance in contacts:
            distance = float(distance)
            entry = self.pairs.setdefault(pair_key(a, b), {"first_touch": None, "touch_in_hold": False,
                                                           "min": math.inf, "steps": set()})
            entry["steps"].add(step)
            entry["min"] = min(entry["min"], distance)
            if distance <= 0.0:
                if entry["first_touch"] is None:
                    entry["first_touch"] = step
                entry["touch_in_hold"] |= step >= self.hold_start
            if self.is_push(a, b):
                self.min_distance[step] = np.fmin(self.min_distance[step], distance)

    def time(self, step: int | None) -> float | None:
        return None if step is None else round(step * self.dt, 9)

    def summary(self) -> dict:
        push = {pair: entry for pair, entry in self.pairs.items() if self.is_push(*pair)}
        touches = [entry["first_touch"] for entry in push.values() if entry["first_touch"] is not None]
        min_separation = min((entry["min"] for entry in push.values()), default=None)

        def rows(entries: Mapping[tuple[str, str], dict]) -> list[dict]:
            return [
                {"link_a": a, "link_b": b, "min_distance_m": entry["min"], "steps": len(entry["steps"]),
                 "first_touch_time_s": self.time(entry["first_touch"])}
                for (a, b), entry in sorted(entries.items(), key=lambda item: (item[1]["min"], item[0]))
            ]

        return {
            "contact_reported": any(entry["touch_in_hold"] for entry in push.values()),
            "first_contact_time_s": self.time(min(touches, default=None)),
            "max_penetration_m": 0.0 if min_separation is None else max(0.0, -min_separation),
            "min_separation_m": min_separation,
            "pusher_target_pairs": rows(push),
            "other_contact_pairs": rows({pair: entry for pair, entry in self.pairs.items()
                                         if pair not in push and entry["first_touch"] is not None}),
        }


def min_jerk(q0: np.ndarray, q1: np.ndarray, t: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    s = min(max(t / duration, 0.0), 1.0)
    return q0 + (q1 - q0) * s**3 * (10.0 - 15.0 * s + 6.0 * s**2), (q1 - q0) * 30.0 * s**2 * (1.0 - s) ** 2 / duration


def stop_state(q: np.ndarray, qd: np.ndarray, target: np.ndarray, window: int, dt: float) -> dict:
    """q, qd: (samples, push joints), sampled before every step and once after the last one."""
    # Speed from positions: in sustained contact Isaac Lab reports joint velocities of a few rad/s at rest.
    speed = float(np.max(np.abs(np.diff(q[-window - 1:], axis=0)))) / dt
    offset = float(np.max(np.abs(q[-1] - target)))
    return {"stopped": speed < STOP_SPEED_RAD_S and offset > STOP_OFFSET_RAD, "final_max_speed_rad_s": speed,
            "final_max_reported_speed_rad_s": float(np.max(np.abs(qd[-window - 1:]))), "final_offset_rad": offset}


def push_criteria(thresholds: Mapping) -> dict:
    contact_required = common.lookup(thresholds, CHECK, "push", "contact_required")
    if not isinstance(contact_required, bool):
        raise common.ValidationError(f"thresholds {thresholds['file']}: {CHECK}.push.contact_required must be "
                                     f"true or false, got {contact_required!r}")
    limit = common.criterion(thresholds, CHECK, "push", "max_penetration", unit="m")
    return {"contact_required": contact_required, "max_penetration": {"value": limit, "unit": "m"}}


def push_verdicts(name: str, result: Mapping, criteria: Mapping) -> dict:
    limit = criteria["max_penetration"]
    contact = {"value": None, "threshold": criteria["contact_required"], "unit": "flag"}
    penetration = {"value": None, "threshold": limit["value"], "unit": limit["unit"]}
    if not result["applicable"]:
        contact["verdict"] = penetration["verdict"] = "not_applicable"
    else:
        contact["value"] = result["contact_reported"]
        if criteria["contact_required"]:
            contact["verdict"] = "pass" if result["contact_reported"] else "fail"
        else:
            contact["verdict"] = "report_only"
        penetration["value"] = result["max_penetration_m"]
        penetration["verdict"] = common.judge(result["max_penetration_m"], limit["value"])
    return {f"push.{name}.contact_reported": contact, f"push.{name}.max_penetration": penetration}


def pair_verdict(comparison: Mapping | None) -> dict:
    if comparison is None:
        return {"value": 0, "threshold": 0, "unit": "pairs", "verdict": "reference"}
    count = len(comparison["unclassified_pairs"])
    return {"value": count, "threshold": 0, "unit": "pairs", "verdict": common.judge(count, 0)}


def run_scenario(sim, joint_names: Sequence[str], geometry: HandGeometry, scenario: Scenario, dt: float,
                 limit_m: float) -> tuple[dict, dict]:
    pairs = scenario_pairs(scenario, geometry.prefix, geometry.categories)
    target = target_vector(joint_names, scenario)
    push = [list(joint_names).index(geometry.prefix + suffix) for suffix in scenario.push_joints]
    distance, closest = geometry.distance(joint_names, target, pairs)
    result = {
        "pushers": sorted({pusher for pusher, _ in pairs}),
        "targets": sorted({target_link for _, target_link in pairs}),
        "pusher_target_pairs_expected": [list(pair) for pair in pairs],
        "push_joints": [joint_names[j] for j in push],
        "target_q": {name: float(value) for name, value in zip(joint_names, target) if value != 0.0},
        "kinematic_penetration_at_target_m": distance,
        "kinematic_closest_pair": closest,
        # The penetration criterion can only fail when the target overlaps by more than its limit.
        "kinematic_valid": distance < -limit_m,
    }
    if not result["kinematic_valid"]:
        return result, {}
    steps = scene.whole_steps(PUSH_MOVE_S + PUSH_HOLD_S, dt)
    log = ContactLog(pairs, dt, steps, scene.whole_steps(PUSH_MOVE_S, dt))
    open_q = np.zeros(len(joint_names))
    q = np.empty((steps + 1, len(joint_names)))
    qd = np.empty_like(q)
    q_des = np.empty((steps, len(joint_names)))
    sim.reset()
    for k in range(steps):
        q_des[k], qd_des = min_jerk(open_q, target, k * dt, PUSH_MOVE_S)
        q[k], qd[k] = sim.state()
        sim.step(q_des[k], qd_des)
        log.add(k, sim.contacts())
    q[steps], qd[steps] = sim.state()
    result.update(log.summary())
    result.update(stop_state(q[:, push], qd[:, push], target[push], scene.whole_steps(STOP_WINDOW_S, dt), dt))
    result["final_q"] = {joint_names[j]: float(q[steps, j]) for j in push}
    series = {"t": np.arange(steps + 1) * dt, "q": q[:, push], "q_des": q_des[:, push],
              "min_distance": log.min_distance, "labels": result["push_joints"]}
    return result, series


def settle_applicability(results: Mapping[str, dict], reference_push: Mapping | None, limit_m: float) -> None:
    """Applicable when the target sinks the pushers in far enough and the MuJoCo run touches no other link pair."""
    for name, result in results.items():
        source = result if reference_push is None else reference_push.get(name)
        if not isinstance(source, Mapping) or not isinstance(source.get("kinematic_valid"), bool):
            raise common.ValidationError(f"the reference record has no push result for scenario {name}")
        if result["kinematic_valid"] and not source["kinematic_valid"]:
            raise common.ValidationError(
                f"push scenario {name} overlaps its target by more than this run's {CHECK}.push.max_penetration "
                f"({limit_m} m), but the reference record did not run it; regenerate the MuJoCo reference with "
                f"the same thresholds")
        if not result["kinematic_valid"]:
            result["mujoco_other_contact_pairs"] = None
            result["applicable"] = False
            continue
        if not isinstance(source.get("other_contact_pairs"), list):
            raise common.ValidationError(f"the reference record has no other_contact_pairs for push scenario {name}")
        result["mujoco_other_contact_pairs"] = [[row["link_a"], row["link_b"]] for row in source["other_contact_pairs"]]
        result["applicable"] = not result["mujoco_other_contact_pairs"]


def run_push(sim, joint_names: Sequence[str], geometry: HandGeometry, dt: float, limit_m: float,
             reference_push: Mapping | None) -> tuple[dict, dict]:
    results, series = {}, {}
    for scenario in SCENARIOS:
        results[scenario.name], data = run_scenario(sim, joint_names, geometry, scenario, dt, limit_m)
        if data:
            series[scenario.name] = data
    settle_applicability(results, reference_push, limit_m)
    return results, series


def push_figure(results: Mapping[str, Mapping], series: Mapping[str, Mapping], limit_m: float, title: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12.0, 2.4 * len(results)), dpi=80, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(results), 2, squeeze=False)
    for (q_ax, d_ax), (name, result) in zip(axes, results.items()):
        q_ax.set_title(f"{name}: push joints" + ("" if result["applicable"] else " (not applicable)"), fontsize=10)
        d_ax.set_title(f"{name}: closest pusher-target contact", fontsize=10)
        if name not in series:
            for ax in (q_ax, d_ax):
                ax.set_axis_off()
                ax.text(0.5, 0.5, "not applicable: kinematic distance "
                        f"{result['kinematic_penetration_at_target_m']:.4g} m", ha="center", va="center",
                        transform=ax.transAxes)
            continue
        data = series[name]
        for j, label in enumerate(data["labels"]):
            q_ax.plot(data["t"], data["q"][:, j], color=f"C{j % 10}", label=label)
            q_ax.plot(data["t"][:-1], data["q_des"][:, j], color=f"C{j % 10}", linestyle="--", linewidth=0.8)
        q_ax.set_xlabel("t [s]")
        q_ax.set_ylabel("q [rad], dashed q_des")
        q_ax.legend(fontsize=7, loc="upper left")
        q_ax.grid(True, alpha=0.3)
        d_ax.plot(data["t"][:-1], data["min_distance"], color="C0", marker=".", markersize=2, linewidth=0.8,
                  label="contact distance")
        d_ax.axhline(-limit_m, color="red", linewidth=0.8, label="-max_penetration")
        d_ax.axhline(0.0, color="gray", linewidth=0.8)
        d_ax.set_xlim(data["t"][0], data["t"][-1])
        d_ax.set_xlabel("t [s]")
        d_ax.set_ylabel("distance [m]")
        d_ax.legend(fontsize=7, loc="lower left")
        d_ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    return fig


def _named_contacts(geom_links: Sequence[str], geom_pairs, distances) -> list[tuple[str, str, float]]:
    return [(geom_links[g1], geom_links[g2], float(distance)) for (g1, g2), distance in zip(geom_pairs, distances)]


class MujocoSim:
    def __init__(self, loaded):
        self.model, self.data = loaded.model, loaded.data
        self.qpos_adr, self.dof_adr = list(loaded.qpos_adr), list(loaded.dof_adr)
        self.actuator_ids = list(loaded.actuator_ids)
        self.geom_links = [self.model.body(int(body)).name for body in self.model.geom_bodyid]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self.qpos_adr].copy(), self.data.qvel[self.dof_adr].copy()

    def step(self, q_des: np.ndarray, qd_des: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = PUSH_KP * q_des + PUSH_KD * qd_des
        mujoco.mj_step(self.model, self.data)

    def contacts(self) -> list[tuple[str, str, float]]:
        n = self.data.ncon
        return _named_contacts(self.geom_links, self.data.contact.geom[:n], self.data.contact.dist[:n])


class WarpSim:
    WORLD = 0

    def __init__(self, loaded, platform, mjw):
        self.loaded, self.platform, self.mjw = loaded, platform, mjw
        self.qpos_adr, self.dof_adr = list(loaded.qpos_adr), list(loaded.dof_adr)
        self.geom_links = [loaded.model.body(int(body)).name for body in loaded.model.geom_bodyid]
        self.buffers = platform.mit_buffers(loaded)
        self.peak: dict = {}
        n = len(loaded.joint_names)
        self.gains = {"kp": np.full(n, PUSH_KP), "kd": np.full(n, PUSH_KD), "tau_ff": np.zeros(n)}

    def reset(self) -> None:
        self.mjw.reset_data(self.loaded.m, self.loaded.d)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.loaded.d
        return d.qpos.numpy()[self.WORLD, self.qpos_adr], d.qvel.numpy()[self.WORLD, self.dof_adr]

    def step(self, q_des: np.ndarray, qd_des: np.ndarray) -> None:
        cmd = mit.MitCommand(self.loaded.joint_names, q_des, qd_des, **self.gains)
        shape = (self.loaded.nworld, len(cmd.joint_names))
        for name in self.platform.MIT_FIELDS:
            value = np.broadcast_to(getattr(cmd, name), shape)
            getattr(self.buffers, name).assign(np.ascontiguousarray(value, dtype=np.float32))
        self.platform.apply_mit(self.loaded, self.buffers)
        self.mjw.step(self.loaded.m, self.loaded.d)
        self.platform.check_capacity(self.loaded, self.peak)

    def contacts(self) -> list[tuple[str, str, float]]:
        d = self.loaded.d
        n = int(d.nacon.numpy()[0])
        world = d.contact.worldid.numpy()[:n] == self.WORLD
        return _named_contacts(self.geom_links, d.contact.geom.numpy()[:n][world], d.contact.dist.numpy()[:n][world])


class IsaacSim:
    def __init__(self, loaded, platform, contacts: "IsaacContacts"):
        self.loaded, self.platform, self.reader = loaded, platform, contacts
        self.zeros = [0.0] * len(loaded.joint_names)

    def reset(self) -> None:
        self.platform.set_joint_state(self.loaded, self.zeros, self.zeros)
        self.platform.set_command(self.loaded, self.zeros, self.zeros, self.zeros)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        q, qd = self.platform.joint_state(self.loaded)
        return np.asarray(q, dtype=np.float64), np.asarray(qd, dtype=np.float64)

    def step(self, q_des: np.ndarray, qd_des: np.ndarray) -> None:
        self.platform.set_command(self.loaded, q_des, qd_des, self.zeros)
        self.platform.step(self.loaded)

    def contacts(self) -> list[tuple[str, str, float]]:
        return self.reader.read(self.loaded.dt)


class IsaacStage:
    """The hand links of the running stage, and which link a collider or joint body belongs to."""

    def __init__(self, loaded):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Sdf, Usd

        self.stage = get_current_stage()
        self.prims = list(self.stage.Traverse(Usd.TraverseInstanceProxies()))
        self.link_paths = [Sdf.Path(str(path)) for path in loaded.articulation.root_view.link_paths[0]]
        self.names = {path: path.name for path in self.link_paths}
        if len(set(self.names.values())) != len(self.names):
            raise common.ValidationError(f"articulation link names are not unique: {sorted(self.names.values())}")
        self.root = Sdf.Path.absoluteRootPath

    def owner(self, path) -> str | None:
        while not path.isEmpty and path != self.root:
            if path in self.names:
                return self.names[path]
            path = path.GetParentPath()
        return None


def isaac_collision_tables(loaded, reference_links: Mapping[str, Mapping]) -> tuple[dict, dict, dict]:
    from pxr import PhysxSchema, UsdPhysics

    stage = IsaacStage(loaded)
    colliders: dict = {}
    for prim in stage.prims:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            link = stage.owner(prim.GetPath())
            if link is not None:
                colliders[prim.GetPath()] = (link, UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
    links = {
        name: {"category": reference_links.get(name, {}).get("category", UNKNOWN),
               "collision_shapes": sum(link == name for link, _ in colliders.values()),
               "enabled_collision_shapes": sum(link == name and enabled is not False
                                               for link, enabled in colliders.values())}
        for name in sorted(stage.names.values())
    }

    adjacent, filters = set(), []
    groups: dict[str, dict] = {}
    scene_inverted = False
    articulations = {}
    for prim in stage.prims:
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            articulations[str(prim.GetPath())] = prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            ends = [stage.owner(targets[0]) if len(targets) == 1 else None
                    for targets in (joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets())]
            if None in ends or ends[0] == ends[1] or joint.GetJointEnabledAttr().Get() is False:
                continue
            if not joint.GetExcludeFromArticulationAttr().Get() or not joint.GetCollisionEnabledAttr().Get():
                adjacent.add(frozenset(ends))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
            filters.append((str(prim.GetPath()), [str(target) for target in targets]))
        if prim.IsA(UsdPhysics.Scene):
            scene_inverted |= bool(prim.GetAttribute("physxScene:invertCollisionGroupFilter").Get())
        if prim.IsA(UsdPhysics.CollisionGroup):
            group = UsdPhysics.CollisionGroup(prim)
            entry = groups.setdefault(group.GetMergeGroupNameAttr().Get() or str(prim.GetPath()),
                                      {"paths": set(), "members": set(), "rules": []})
            query = group.GetCollidersCollectionAPI().ComputeMembershipQuery()
            entry["paths"].add(prim.GetPath())
            entry["members"] |= {str(path) for path in colliders if query.IsPathIncluded(path)}
            entry["rules"].append((set(group.GetFilteredGroupsRel().GetTargets()),
                                   bool(group.GetInvertFilteredGroupsAttr().Get())))

    key_of = {path: key for key, entry in groups.items() for path in entry["paths"]}
    group_rules = {key: [({key_of[path] for path in targets if path in key_of}, invert)
                         for targets, invert in entry["rules"]]
                   for key, entry in groups.items()}
    membership = {str(path): {key for key, entry in groups.items() if str(path) in entry["members"]}
                  for path in colliders}
    shapes_by_link: dict[str, list[str]] = {}
    for path, (link, enabled) in colliders.items():
        if enabled is not False:
            shapes_by_link.setdefault(link, []).append(str(path))
    filtered, group_filtered = blocked_link_pairs(
        shapes_by_link, filtered_shape_pairs(filters, [str(path) for path in colliders]),
        lambda a, b: collision_groups_filter(membership[a], membership[b], group_rules, scene_inverted))

    if not articulations:
        raise common.ValidationError("the stage has no PhysxArticulationAPI prim to read "
                                     "physxArticulation:enabledSelfCollisions from")
    compared = [name for name in compared_links(reference_links, links) if name in links]
    rules = isaac_pair_rules(compared, {name: links[name]["enabled_collision_shapes"] > 0 for name in compared},
                             all(value is not False for value in articulations.values()), adjacent, filtered,
                             group_filtered)
    stage_values = {
        "enabled_self_collisions": articulations,
        "self_collision_readback_at_load": loaded.report["self_collision_readback"],
        "joint_adjacent_pairs": len(adjacent),
        "filtered_pairs": len(filtered),
        "collision_groups": [
            {"group": key, "colliders": len(entry["members"]),
             "merged_rules": [{"filtered_groups": sorted(targets), "invert_filtered_groups": invert}
                              for targets, invert in group_rules[key]]}
            for key, entry in sorted(groups.items())
        ],
        "invert_collision_group_filter": scene_inverted,
        "group_filtered_pairs": len(group_filtered),
    }
    return links, rules, stage_values


def enable_contact_reports(loaded) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import PhysxSchema

    stage = get_current_stage()
    for path in loaded.articulation.root_view.link_paths[0]:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.0)


class IsaacContacts:
    """Every contact of the hand links after the last step, from a PhysX rigid contact view."""

    def __init__(self, loaded):
        from isaaclab_physx.physics import PhysxManager

        self.stage = IsaacStage(loaded)
        self.sensors = [self.stage.names[path] for path in self.stage.link_paths]
        self.view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            [str(path) for path in self.stage.link_paths], max_contact_data_count=ISAAC_MAX_CONTACTS)
        if self.view is None:
            raise common.ValidationError("PhysX created no rigid contact view for the hand links; they need "
                                         "PhysxContactReportAPI before the simulation starts")
        self.actors: dict[int, str] = {}

    def _actor(self, path: str) -> str:
        from pxr import Sdf

        return self.stage.owner(Sdf.Path(path)) or path

    def read(self, dt: float) -> list[tuple[str, str, float]]:
        import warp as wp

        _, _, _, separation, count, start, other = self.view.get_raw_contact_data(dt)
        count, start = count.numpy().reshape(-1), start.numpy().reshape(-1)
        separation, other = separation.numpy().reshape(-1), other.numpy().reshape(-1)
        if count.sum() >= ISAAC_MAX_CONTACTS:
            raise common.ValidationError(f"{int(count.sum())} hand contacts fill the {ISAAC_MAX_CONTACTS} contact "
                                         "slots of the rigid contact view; contacts may be missing")
        contacts = []
        for sensor in np.nonzero(count)[0]:
            span = slice(int(start[sensor]), int(start[sensor] + count[sensor]))
            ids = [int(actor) for actor in other[span]]
            unknown = sorted(set(ids) - set(self.actors))
            if unknown:
                paths = self.view.get_other_actor_paths_from_ids(
                    wp.array(np.asarray(unknown, dtype=np.uint64), dtype=wp.uint64, device="cpu"))
                self.actors.update(zip(unknown, (self._actor(str(path)) for path in paths)))
            contacts += [(self.sensors[sensor], self.actors[actor], float(value))
                         for actor, value in zip(ids, separation[span])]
        return contacts


def push_protocol(stability: Mapping, dt: float, limit_m: float) -> dict:
    return {
        "control": {
            "law": "tau = clip(kp*(q_des-q) + kd*(qd_des-qd), -tau_limit, tau_limit)",
            "kp": PUSH_KP, "kd": PUSH_KD, "units": {"kp": "N*m/rad", "kd": "N*m*s/rad"},
            "control_period_s": dt, "stability": stability,
        },
        "motion": {"from": "open", "profile": "min-jerk", "move_s": PUSH_MOVE_S, "hold_s": PUSH_HOLD_S,
                   "steps": scene.whole_steps(PUSH_MOVE_S + PUSH_HOLD_S, dt)},
        "scenarios": [
            {"name": scenario.name, "pairs": [list(pair) for pair in scenario.pairs],
             "push_joints": list(scenario.push_joints), "target_rad": dict(scenario.target)}
            for scenario in SCENARIOS
        ],
        "scenario_pairs": "the (pusher, target) link pairs a scenario expects to collide, fixed in the check rather "
                          "than read from the collision filters of the model under test: link names after the hand "
                          "prefix, or a link category standing for every link of it (pusher_target_pairs_expected "
                          "lists them resolved); a contact counts as pusher-target only on these pairs",
        "kinematic_penetration_at_target_m": (
            "smallest signed distance (negative: penetration) between the collision geoms of the expected "
            "pusher-target pairs at the target pose, after mj_kinematics on the description MJCF alone (MuJoCo "
            f"{mujoco.__version__}), whatever the collision filters; penetration from mujoco.mj_geomDistance at "
            "distmax 0, separation as the smallest distmax at which it finds a contact (to "
            f"{GEOM_DISTANCE_RESOLUTION_M} m, capped at {GEOM_DISTANCE_MAX_M} m); kinematic_valid: below "
            f"-max_penetration ({-limit_m} m, {CHECK}.push.max_penetration of the thresholds)"
        ),
        "applicable": "kinematic_valid, and the MuJoCo run of the scenario (this record for MuJoCo, the --reference "
                      "record otherwise) has no other_contact_pairs (mujoco_other_contact_pairs), so the pushers "
                      "alone meet the targets; a scenario that is not applicable gets the verdict not_applicable",
        "contact_time": "contacts are read after every physics step and stamped with the time of the state that "
                        "step started from",
        "contact_reported": f"a pusher-target contact with distance <= 0 at a step of the hold (t >= {PUSH_MOVE_S} s)",
        "first_contact_time_s": "time of the first pusher-target contact with distance <= 0 over the whole run",
        "min_separation_m": "smallest distance of any reported pusher-target contact over the whole run (move and "
                            "hold; null without one)",
        "max_penetration_m": "max(0, -min_separation_m): a touch during the move counts even if the hold has none",
        "other_contact_pairs": "link pairs other than pusher-target with a contact of distance <= 0",
        "stopped": f"over the last {STOP_WINDOW_S} s every push joint moves slower than {STOP_SPEED_RAD_S} rad/s "
                   "(final_max_speed_rad_s, from the change of the sampled joint positions per step; "
                   "final_max_reported_speed_rad_s is the largest joint velocity the simulator reports), and the "
                   f"final pose is more than {STOP_OFFSET_RAD} rad from the target in some push joint",
    }


def pairs_protocol(simulator: str) -> dict:
    protocol = {
        "links": "hand links with a collision shape (MuJoCo: geom with contype or conaffinity, on any body but "
                 "world and scene_* bodies; Isaac Lab: UsdPhysics.CollisionAPI prim below the link)",
        "category": "a link belongs to the finger whose joint body it is welded to in the MuJoCo model, or to the "
                    "palm when it is welded to the world (wrist mount included)",
        "rules": {"mujoco": list(MUJOCO_RULES), "mujoco_warp": list(WARP_RULES), "isaaclab": list(ISAAC_RULES)},
        "rule_order": "the first rule that applies names the pair; collidable pairs are named explicit_pair or "
                      "collidable (MuJoCo), warp_supported (MuJoCo Warp) and collidable (Isaac Lab)",
        "comparison": "every unordered pair of the links with a collision shape in either simulator; a pair whose "
                      "collidable flag differs is a difference, a pair that either simulator cannot classify is "
                      "unclassified and fails the check",
    }
    if simulator == "isaaclab":
        protocol["rule_source"] = dict(ISAAC_RULE_SOURCES)
    return protocol


def pair_values(links: Mapping[str, Mapping], rules: Mapping, categories: Mapping[str, str]) -> dict:
    rows = pair_rows(rules, categories)
    return {"links": dict(links), "summary": pair_summary(rows), "pairs": rows}


def report(values: Mapping) -> None:
    pairs = values["collision_pairs"]
    line = f"collision pairs: {pairs['summary']['collidable']} of {pairs['summary']['pairs']} collidable"
    comparison = pairs.get("comparison")
    if comparison is not None:
        line += (f"; {comparison['differences']} differ from {common.REFERENCE_SIMULATOR} by category "
                 f"{comparison['differences_by_category']}, {len(comparison['unclassified_pairs'])} unclassified")
    print(line)
    for name, result in values["push"].items():
        if not result["kinematic_valid"]:
            print(f"push {name}: not applicable, kinematic distance {result['kinematic_penetration_at_target_m']:.4g} m")
            continue
        print(f"push {name}: kinematic {result['kinematic_penetration_at_target_m']:.4g} m, contact "
              f"{result['contact_reported']}, max penetration {result['max_penetration_m']:.4g} m, "
              f"stopped {result['stopped']}"
              + ("" if result["applicable"] else
                 f"; not applicable, the MuJoCo run also touches {result['mujoco_other_contact_pairs']}"))


def finish(record: common.ValidationRecord, asset: assets.ResolvedAsset, thresholds: Mapping,
           reference: Mapping | None, criteria: Mapping, outcome: Mapping, simulator: str, hand: str) -> int:
    figure = push_figure(outcome["push"], outcome["series"], criteria["max_penetration"]["value"],
                         f"{simulator}, {hand} hand: push tests")
    figure.savefig(record.artifact_path("push", ".png"), format="png")
    values = {"collision_pairs": {**outcome["pairs"], "comparison": outcome["comparison"]},
              "push": outcome["push"], **outcome["extra"]}
    verdicts = {"collision_pairs.unclassified": pair_verdict(outcome["comparison"])}
    for name, result in outcome["push"].items():
        verdicts.update(push_verdicts(name, result, criteria))
    protocol = {"collision_pairs": pairs_protocol(simulator),
                "push": push_protocol(outcome["stability"], outcome["dt"], criteria["max_penetration"]["value"])}
    path = record.write(asset=asset, thresholds=thresholds, reference=reference, protocol=protocol, values=values,
                        verdicts=verdicts)
    report(values)
    return common.print_result(path, verdicts)


def set_push_gains(platform, loaded) -> dict:
    n = len(loaded.joint_names)
    kp, kd = np.full(n, PUSH_KP), np.full(n, PUSH_KD)
    stability = platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    return stability


def run_mujoco(platform, asset: assets.ResolvedAsset, hand: str, geometry: HandGeometry, limit_m: float) -> dict:
    view = framing.view(asset.mjcf, hand, platform.PLATFORM, VIEW_KIND)
    loaded = platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, hand, view),
                           description_version=asset.expected_version)
    model = loaded.model
    links = mujoco_links(model, hand)
    pairs = pair_values(links, mujoco_pair_rules(model, links), {name: entry["category"] for name, entry in links.items()})
    pairs["model"] = {
        "nexclude": int(model.nexclude), "npair": int(model.npair),
        "filterparent": not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT,
    }
    stability = set_push_gains(platform, loaded)
    dt = float(model.opt.timestep)
    push, series = run_push(MujocoSim(loaded), loaded.joint_names, geometry, dt, limit_m, None)
    return {"pairs": pairs, "comparison": None, "push": push, "series": series, "stability": stability, "dt": dt,
            "extra": {}}


def run_warp(platform, asset: assets.ResolvedAsset, hand: str, geometry: HandGeometry, limit_m: float,
             reference: Mapping) -> dict:
    import mujoco_warp as mjw
    from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE

    view = framing.view(asset.mjcf, hand, platform.PLATFORM, VIEW_KIND)
    loaded = platform.load(asset.mjcf, nworld=1, scene=lambda spec: scene.add_scene(spec, hand, view),
                           description_version=asset.expected_version)
    model = loaded.model
    reference_links, reference_rules = reference_pairs(reference["record"])
    links = mujoco_links(model, hand)
    supported = {tuple(sorted((int(a), int(b)))) for a, b in MJ_COLLISION_TABLE}
    rules = warp_pair_rules([model.body(int(body)).name for body in model.geom_bodyid], model.geom_type,
                            loaded.m.nxn_geom_pair_filtered.numpy(), loaded.m.nxn_pairid_filtered.numpy(), supported,
                            [name for name in compared_links(reference_links, links) if name in links])
    categories = {name: entry["category"] for name, entry in reference_links.items()}
    stability = set_push_gains(platform, loaded)
    dt = float(model.opt.timestep)
    sim = WarpSim(loaded, platform, mjw)
    push, series = run_push(sim, loaded.joint_names, geometry, dt, limit_m, reference_push(reference["record"]))
    return {
        "pairs": pair_values(links, rules, categories),
        "comparison": compare_pairs(platform.PLATFORM, reference_links, reference_rules, links, rules),
        "push": push, "series": series, "stability": stability, "dt": dt,
        "extra": {"capacity_usage": sim.peak},
    }


def run_isaac(platform, asset: assets.ResolvedAsset, hand: str, geometry: HandGeometry, limit_m: float,
              reference: Mapping, finish_run, expected: tuple[type[BaseException], ...]) -> int:
    view = framing.view(asset.mjcf, hand, platform.PLATFORM, VIEW_KIND)
    with platform.launch(headless=True):
        # launch_simulation turns any exception but SystemExit into exit 1 (a failed check), so always sys.exit here.
        try:
            loaded = platform.load(asset.usd, hand=hand, floor_z=view.floor_z,
                                   description_version=asset.expected_version, scene=scene_isaaclab.Scene())
            reference_links, reference_rules = reference_pairs(reference["record"])
            links, rules, stage_values = isaac_collision_tables(loaded, reference_links)
            enable_contact_reports(loaded)
            stability = set_push_gains(platform, loaded)
            sim = IsaacSim(loaded, platform, IsaacContacts(loaded))
            push, series = run_push(sim, loaded.joint_names, geometry, float(loaded.dt), limit_m,
                                    reference_push(reference["record"]))
            categories = {name: entry["category"] for name, entry in reference_links.items()}
            pairs = pair_values(links, rules, categories)
            pairs["stage"] = stage_values
            code = finish_run({
                "pairs": pairs,
                "comparison": compare_pairs(platform.PLATFORM, reference_links, reference_rules, links, rules),
                "push": push, "series": series, "stability": stability, "dt": float(loaded.dt), "extra": {},
            })
        except Exception as exc:
            code = common.report_error(exc, expected)
        sys.exit(code)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare which hand link pairs can collide with the MuJoCo model, and push fingers into each "
                    "other and into the palm under MIT control."
    )
    common.add_cli_args(parser)
    args = common.parse_args(parser, argv[1:])
    platform = importlib.import_module(f"wuji_sim.platforms.{args.simulator}")
    expected = (*common.EXPECTED_ERRORS, profiles.ProfileError, mit.MitStabilityError)
    try:
        thresholds = common.load_thresholds(args.thresholds)
        criteria = push_criteria(thresholds)
        asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
        print(asset.describe())
        reference = (None if args.reference is None
                     else common.load_reference(args.reference, check=CHECK, hand=args.hand, asset=asset))
        record = common.ValidationRecord(check=CHECK, simulator=args.simulator, hand=args.hand,
                                         output_dir=args.output_dir, argv=argv)
        geometry = HandGeometry(asset.mjcf, args.hand)

        def finish_run(outcome: Mapping) -> int:
            return finish(record, asset, thresholds, reference, criteria, outcome, args.simulator, args.hand)

        limit_m = criteria["max_penetration"]["value"]
        if args.simulator == "isaaclab":
            return run_isaac(platform, asset, args.hand, geometry, limit_m, reference, finish_run, expected)
        if args.simulator == "mujoco_warp":
            return finish_run(run_warp(platform, asset, args.hand, geometry, limit_m, reference))
        return finish_run(run_mujoco(platform, asset, args.hand, geometry, limit_m))
    except Exception as exc:
        return common.report_error(exc, expected)


if __name__ == "__main__":
    try:
        code = main(sys.argv)
    except Exception as exc:
        code = common.report_error(exc)
    sys.exit(code)
