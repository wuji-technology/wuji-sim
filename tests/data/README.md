# Test data

## pen_spin_training_env_obs.npz

Reference values for `tests/test_pen_spin.py`, taken from the wuji-mjlab release v2026.10.22
(https://github.com/wuji-technology/wuji-mjlab, Apache-2.0):

- the first three frames of clip `clip_001` of the `pen_spin` motion group `01_single_axis`, as published;
- values recorded in that repository's training environment (play configuration, first-frame hold off, a single
  environment, seed 0) running the group's published policy: the observation at reset, the policy's raw action
  for that observation, and the observation, joint positions and pen pose after one control step.

The observation layout is the one built by `observation()` in `examples/_shared/pen_spin.py`.

| Array | Shape, dtype | Meaning |
| :--- | :--- | :--- |
| `reference_qpos` | (3, 20) float32 | reference joint positions of clip frames 0 to 2, rad |
| `wrist_pose_w` | (3, 7) float32 | recorded wrist pose of those frames: position in m, then a wxyz quaternion |
| `object_pose_w` | (3, 7) float32 | recorded pen pose of those frames, same layout |
| `link_pos_w` | (3, 21, 3) float32 | recorded positions of the hand's 21 links in those frames, m |
| `reset_obs` | (155,) float32 | policy observation right after reset |
| `reset_action` | (20,) float32 | raw policy output for `reset_obs` |
| `next_obs` | (155,) float32 | policy observation after the first control step |
| `next_joint_pos` | (20,) float32 | joint positions after the first control step, rad |
| `next_pen_pose_w` | (7,) float64 | pen pose in the world after the first control step: position in m, then a wxyz quaternion |
