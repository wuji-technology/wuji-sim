# SPDX-License-Identifier: Apache-2.0
import shlex
import sys

GRAVITY = [0.0, 0.0, -9.81]
JOINT_SUFFIXES = (
    "thumb_cmc_flex",
    "thumb_cmc_abd",
    "thumb_mcp",
    "thumb_ip",
    "index_finger_mcp_flex",
    "index_finger_mcp_abd",
    "index_finger_pip",
    "index_finger_dip",
    "middle_finger_mcp_flex",
    "middle_finger_mcp_abd",
    "middle_finger_pip",
    "middle_finger_dip",
    "ring_finger_mcp_flex",
    "ring_finger_mcp_abd",
    "ring_finger_pip",
    "ring_finger_dip",
    "pinky_mcp_flex",
    "pinky_mcp_abd",
    "pinky_pip",
    "pinky_dip",
)
HAND_PREFIXES = {"left": "l_", "right": "r_"}


class ViewerError(RuntimeError):
    pass


def headless_rerun_hint() -> str:
    return "rerun with --headless to write files instead: " + shlex.join([sys.executable, *sys.argv, "--headless"])
