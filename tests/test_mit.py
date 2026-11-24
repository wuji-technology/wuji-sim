# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from wuji_sim.mit import MitCommand, mit_torque

ONE = ("j0",)


def command(names=ONE, *, q_des=1.0, qd_des=0.0, kp=2.0, kd=0.1, tau_ff=0.05):
    n = len(names)
    return MitCommand(
        joint_names=names,
        q_des=np.full(n, q_des),
        qd_des=np.full(n, qd_des),
        kp=np.full(n, kp),
        kd=np.full(n, kd),
        tau_ff=np.full(n, tau_ff),
    )


def test_hand_computed_torque():
    tau = mit_torque(command(), np.array([0.5]), np.array([1.0]), np.array([1.0]))
    # 2 * (1 - 0.5) + 0.1 * (0 - 1) + 0.05
    np.testing.assert_allclose(tau, [0.95], rtol=0, atol=1e-12)


def test_positive_saturation():
    tau = mit_torque(command(), np.array([0.5]), np.array([1.0]), np.array([0.5]))
    np.testing.assert_array_equal(tau, [0.5])


def test_negative_saturation():
    tau = mit_torque(command(q_des=-1.0), np.array([0.5]), np.array([1.0]), np.array([0.5]))
    np.testing.assert_array_equal(tau, [-0.5])


def test_command_arrays_are_float64():
    cmd = MitCommand(joint_names=["a", "b"], q_des=[1, 2], qd_des=[0, 0], kp=[1, 1], kd=[0, 0], tau_ff=[0, 0])
    assert cmd.joint_names == ("a", "b")
    for name in ("q_des", "qd_des", "kp", "kd", "tau_ff"):
        value = getattr(cmd, name)
        assert isinstance(value, np.ndarray) and value.dtype == np.float64


def test_batch_broadcast_over_worlds():
    names = ("a", "b", "c")
    nworld = 4
    cmd = command(names, q_des=0.2, qd_des=0.0, kp=1.0, kd=0.0, tau_ff=0.0)
    q = np.arange(nworld * 3, dtype=float).reshape(nworld, 3) * 0.01
    qd = np.zeros((nworld, 3))
    tau = mit_torque(cmd, q, qd, np.full(3, 10.0))
    assert tau.shape == (nworld, 3)
    np.testing.assert_allclose(tau, 0.2 - q, rtol=0, atol=1e-12)


def test_batched_command_per_world():
    names = ("a", "b")
    q_des = np.array([[0.1, 0.2], [0.3, 0.4]])
    cmd = MitCommand(
        joint_names=names, q_des=q_des, qd_des=np.zeros(2), kp=np.ones(2), kd=np.zeros(2), tau_ff=np.zeros(2)
    )
    tau = mit_torque(cmd, np.zeros(2), np.zeros(2), np.ones(2))
    np.testing.assert_allclose(tau, q_des, rtol=0, atol=1e-12)


def test_command_last_dim_must_match_joint_names():
    with pytest.raises(ValueError):
        MitCommand(joint_names=("a", "b"), q_des=np.zeros(3), qd_des=np.zeros(2), kp=np.ones(2), kd=np.zeros(2),
                   tau_ff=np.zeros(2))


def test_state_shape_mismatch():
    with pytest.raises(ValueError):
        mit_torque(command(("a", "b")), np.zeros(3), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        mit_torque(command(("a", "b")), np.zeros(2), np.zeros((2, 3)), np.ones(2))


@pytest.mark.parametrize("field", ["kp", "kd"])
def test_negative_gains_rejected(field):
    with pytest.raises(ValueError):
        command(**{field: -0.1})


@pytest.mark.parametrize("field", ["q_des", "qd_des", "kp", "kd", "tau_ff"])
def test_nan_command_rejected(field):
    with pytest.raises(ValueError):
        command(**{field: np.nan})


def test_nan_state_rejected():
    with pytest.raises(ValueError):
        mit_torque(command(), np.array([np.nan]), np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        mit_torque(command(), np.array([0.0]), np.array([np.inf]), np.array([1.0]))


@pytest.mark.parametrize("limit", [[0.0], [-1.0], [np.nan], [1.0, 1.0], [[1.0]]])
def test_invalid_tau_limit_rejected(limit):
    with pytest.raises(ValueError):
        mit_torque(command(), np.array([0.0]), np.array([0.0]), np.array(limit))
