# Contributing

## Development environment

Development needs Linux x86-64 with glibc 2.35 or newer (for example Ubuntu 22.04 or later) and Pixi 0.84.1 or
newer. Install Pixi, then install the environments used by the project:

```bash
pixi install --locked -e mujoco
pixi install --locked -e mujoco-warp
```

Pixi installs the hand model (`wuji-hand2-2-description`) with each environment; to test against a wuji-description checkout instead, set `WUJI_DESCRIPTION_ROOT` to its `hand2/hand2_2/body` directory.

Run the CPU test suite with:

```bash
pixi run -e mujoco pytest -q tests
```

Run the non-GPU tests in the MuJoCo Warp environment with:

```bash
pixi run -e mujoco-warp pytest -q -m 'not gpu' tests
```

GPU-marked tests require an NVIDIA GPU and should be run in the `mujoco-warp`
environment.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/), for example:

```text
feat(platforms): add integrated actuator support
fix(examples): correct sweep saturation statistics
```

## Pull requests

This repository is published from a separate development repository, and each release replaces its
contents as a whole, so pull requests are not merged here directly. They go through these steps:

1. Open an issue or a pull request in this repository; a maintainer reviews it here.
2. Once it is accepted, a maintainer makes the same change in the development repository.
3. The change ships in the next release.
4. The maintainer then closes your pull request with a comment naming the release that contains it.

## Bug reports

Include the following information so the issue can be reproduced:

- wuji-sim commit (`git rev-parse HEAD`)
- wuji-hand2-2-description version (for example, `2026.11.18`)
- Backend name and version
- The complete run record JSON produced by the example; it contains the working directory, the command line
  and absolute file paths, so replace any local paths or user names in it before posting
