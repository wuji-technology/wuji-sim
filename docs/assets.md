# Asset Lookup and Digest Check

`wuji_sim.assets.resolve(hand, asset_root=None, strict=False)` finds the hand model in this order:

1. `--asset-root` (the `asset_root` argument);
2. `$WUJI_DESCRIPTION_ROOT`;
3. the installed `wuji-hand2-2-description` package (every Pixi environment locks version `2026.11.18`).

The first two point at the `hand2/hand2_2/body` directory of a wuji-description checkout. If none of the three is
available, `assets.AssetError` is raised.

It then hashes every regular file under the root into a SHA-256 manifest digest and compares it against
`assets-expected.toml` for `hand2_2` (currently `v2026.11.18`). The two kinds of root use different file sets:

- A wuji-description checkout is hashed without the root-level `package.xml` and `CMakeLists.txt`, and compared
  against `digest`.
- The installed package (any root containing `manifest.json`) ships only `mjcf/`, `urdf/`, `meshes/` and `usd/`.
  It is hashed without its generated `__init__.py`, `_version.py` and `manifest.json` (and `__pycache__/`), and
  compared against `package_digest`. The recomputed digest must also equal the `digest` declared in the package's
  own `manifest.json`; otherwise `assets.AssetError` is raised.

- Digest matches: `asset.status == "matches-expected"`.
- Digest differs: a `AssetsModifiedWarning` is raised and `asset.status == "assets-modified"`, unless
  `--strict` is given, in which case `assets.AssetError` is raised instead.

Simulation adapters also load `wuji_sim/model_overrides.toml`. The file replaces the hand model's `joint_armature` (`kg*m^2`) and `joint_torque_limit` (`N*m`) groups; both are per-joint estimates not confirmed against hardware, and each group's `source` records how its values were obtained.

The `product` and `description_version` headers must match the resolved asset. Every effective value is recorded in the run record alongside its original hand-model value.
