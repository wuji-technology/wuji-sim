# SPDX-License-Identifier: Apache-2.0
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

INTEGRATED_CONTROL_PATH = "integrated actuator"
CONTROL_PATH_LABEL = "control path: integrated actuator"
EXPLICIT_MIT_STABILITY_LIMIT = 2.0
# Stability upper bound for the integrated spring term kp*dt^2/I, not a precision guarantee.
MIT_STABILITY_LIMIT = 4.0
MIT_STABILITY_EVIDENCE = "integrated actuator stability bound"


class MitStabilityError(ValueError):
    pass


_FIELDS = ("q_des", "qd_des", "kp", "kd", "tau_ff")


@dataclass(frozen=True)
class MitCommand:
    joint_names: tuple[str, ...]
    q_des: np.ndarray
    qd_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        n = len(self.joint_names)
        for name in _FIELDS:
            value = np.array(getattr(self, name), dtype=np.float64)
            if value.ndim == 0 or value.shape[-1] != n:
                raise ValueError(f"MitCommand.{name} has shape {value.shape}; last dim must be {n} (joint_names)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"MitCommand.{name} contains non-finite values")
            object.__setattr__(self, name, value)
        for name in ("kp", "kd"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"MitCommand.{name} must be >= 0")
        try:
            np.broadcast_shapes(*(getattr(self, name).shape for name in _FIELDS))
        except ValueError as exc:
            raise ValueError(f"MitCommand arrays do not broadcast together: {exc}") from exc


def _state(name: str, value, n: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != n:
        raise ValueError(f"{name} has shape {array.shape}; last dim must be {n}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def mit_torque(cmd: MitCommand, q: np.ndarray, qd: np.ndarray, tau_limit: np.ndarray) -> np.ndarray:
    n = len(cmd.joint_names)
    q = _state("q", q, n)
    qd = _state("qd", qd, n)
    limit = np.asarray(tau_limit, dtype=np.float64)
    if limit.shape != (n,):
        raise ValueError(f"tau_limit has shape {limit.shape}; expected ({n},)")
    if not np.all(limit > 0):
        raise ValueError(f"tau_limit must be > 0 for every joint, got {limit}")
    tau = cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.qd_des - qd) + cmd.tau_ff
    return np.clip(tau, -limit, limit)


def validate_gains(
    joint_names: Sequence[str], kp: Sequence[float], kd: Sequence[float]
) -> tuple[list[float], list[float]]:
    expected = len(joint_names)
    kp = [float(value) for value in kp]
    kd = [float(value) for value in kd]
    if len(kp) != expected:
        raise ValueError(f"kp has {len(kp)} values; expected one per joint {joint_names}")
    if len(kd) != expected:
        raise ValueError(f"kd has {len(kd)} values; expected one per joint {joint_names}")
    if not all(math.isfinite(value) for value in kp):
        raise ValueError("kp values must be finite")
    if not all(math.isfinite(value) and value >= 0.0 for value in kd):
        raise ValueError("kd values must be finite and non-negative")
    return kp, kd


def stability(
    joint_names: Sequence[str],
    joint_inertia: Sequence[float],
    dt: float,
    kp: Sequence[float],
    kd: Sequence[float],
    *,
    control_path: str,
    criterion: str,
) -> dict:
    kp = [float(value) for value in kp]
    kd = [float(value) for value in kd]
    if len(kp) != len(joint_names):
        raise ValueError(f"kp has {len(kp)} values; expected one per joint {joint_names}")
    if len(kd) != len(joint_names):
        raise ValueError(f"kd has {len(kd)} values; expected one per joint {joint_names}")
    kp_ratios = [gain * dt * dt / inertia for gain, inertia in zip(kp, joint_inertia)]
    kd_ratios = [gain * dt / inertia for gain, inertia in zip(kd, joint_inertia)]
    integrated = control_path == INTEGRATED_CONTROL_PATH
    if integrated:
        limit = MIT_STABILITY_LIMIT
        ratios = kp_ratios
        unstable = [index for index, ratio in enumerate(ratios) if ratio >= limit]
    else:
        limit = EXPLICIT_MIT_STABILITY_LIMIT
        ratios = kd_ratios
        unstable = [index for index, ratio in enumerate(ratios) if ratio > limit]
    joints = {
        name: {
            "kp": gain_kp,
            "kd": gain_kd,
            "inertia": inertia,
            "ratio": ratio,
            "kp_dt2_over_I": kp_ratio,
            "kd_dt_over_I": kd_ratio,
        }
        for name, gain_kp, gain_kd, inertia, ratio, kp_ratio, kd_ratio in zip(
            joint_names, kp, kd, joint_inertia, ratios, kp_ratios, kd_ratios
        )
    }
    worst_index = max(range(len(ratios)), key=ratios.__getitem__)
    if unstable:
        details = "; ".join(
            f"{joint_names[index]}: kp {kp[index]:g} N*m/rad, kd {kd[index]:g} N*m*s/rad, "
            f"dt {dt:g} s, I {joint_inertia[index]:.4e} kg*m^2, "
            f"kp*dt^2/I {kp_ratios[index]:.3f}, kd*dt/I {kd_ratios[index]:.3f}"
            for index in unstable
        )
        if integrated:
            raise MitStabilityError(
                f"MIT integrated actuator stability bound violated: kp*dt^2/I >= {limit:g} for {details}; "
                "lower kp or the timestep"
            )
        raise MitStabilityError(
            f"MIT explicit torque path would chatter: kd*dt/I > {limit:g} for {details}; "
            "lower kd or the timestep"
        )
    return {
        "control_path": control_path,
        "criterion": criterion,
        "evidence": MIT_STABILITY_EVIDENCE,
        "limit": limit,
        "dt": dt,
        "units": {
            "kp": "N*m/rad",
            "kd": "N*m*s/rad",
            "inertia": "kg*m^2",
            "dt": "s",
            "ratio": "1",
            "kp_dt2_over_I": "1",
            "kd_dt_over_I": "1",
        },
        "max_ratio": ratios[worst_index],
        "max_joint": joint_names[worst_index],
        "joints": joints,
    }
