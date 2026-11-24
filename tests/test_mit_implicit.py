# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from wuji_sim import mit
from wuji_sim.platforms import _mjcf
from wuji_sim.platforms import mujoco as platform


XML = """
<mujoco>
  <option integrator="implicitfast"/>
  <worldbody>
    <body>
      <joint name="j" type="hinge" actuatorfrcrange="-0.5 0.5" armature="0.1"/>
      <geom size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <position name="p" joint="j" kp="2" kv="0.3"/>
  </actuator>
</mujoco>
"""


def compile_replacement(timestep: float):
    spec = mujoco.MjSpec.from_string(XML)
    spec.option.timestep = timestep
    replacement = _mjcf.replace_position_actuators(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return replacement, model, data


def test_replacement_uses_affine_general_actuator_and_forcerange_clamp():
    replacement, model, data = compile_replacement(0.005)
    platform.set_mit_gains(SimpleNamespace(model=model, joint_names=("j",), actuator_ids=(0,)), [2.0], [0.3])
    assert replacement["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert replacement["joints"][0]["kp"] == 2.0
    assert replacement["joints"][0]["kd"] == 0.3
    assert replacement["joints"][0]["tau_limit"] == 0.5
    assert model.actuator_gaintype.tolist() == [int(mujoco.mjtGain.mjGAIN_FIXED)]
    assert model.actuator_biastype.tolist() == [int(mujoco.mjtBias.mjBIAS_AFFINE)]
    assert model.actuator_gainprm[0, :3].tolist() == [1.0, 0.0, 0.0]
    assert model.actuator_biasprm[0, :3].tolist() == [0.0, -2.0, -0.3]
    assert model.actuator_forcelimited.tolist() == [1]
    assert model.actuator_ctrllimited.tolist() == [0]
    assert model.actuator_forcerange.tolist() == [[-0.5, 0.5]]

    q, qd = 0.2, -0.4
    q_des, qd_des, tau_ff = 0.7, 0.5, 0.2
    data.qpos[0] = q
    data.qvel[0] = qd
    data.ctrl[0] = 2.0 * q_des + 0.3 * qd_des + tau_ff
    mujoco.mj_forward(model, data)
    expected = np.clip(2.0 * (q_des - q) + 0.3 * (qd_des - qd) + tau_ff, -0.5, 0.5)
    assert data.qfrc_actuator[0] == pytest.approx(expected, abs=1e-6)

def test_runtime_mit_gains_replace_asset_bias_and_drive_actual_force():
    _, model, data = compile_replacement(0.005)
    loaded = SimpleNamespace(model=model, joint_names=("j",), actuator_ids=(0,))
    platform.set_mit_gains(loaded, [3.0], [0.7])

    q_des, qd_des, tau_ff = 0.2, 0.0, 0.01
    data.qpos[0] = 0.1
    data.qvel[0] = 0.1
    data.ctrl[0] = 3.0 * q_des + 0.7 * qd_des + tau_ff
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    expected = np.clip(3.0 * (q_des - data.qpos[0]) + 0.7 * (qd_des - data.qvel[0]) + tau_ff, -0.5, 0.5)
    assert data.qfrc_actuator[0] == pytest.approx(expected, abs=1e-6)
    assert model.actuator_biasprm[0, :3].tolist() == [0.0, -3.0, -0.7]


@pytest.mark.parametrize(
    ("kp", "kd", "match"),
    [
        ([3.0, 4.0], [0.7], "kp has 2 values"),
        ([float("nan")], [0.7], "kp values must be finite"),
        ([3.0], [-0.1], "kd values must be finite and non-negative"),
    ],
)
def test_set_mit_gains_rejects_invalid_values(kp, kd, match):
    _, model, _ = compile_replacement(0.005)
    loaded = SimpleNamespace(model=model, joint_names=("j",), actuator_ids=(0,))
    with pytest.raises(ValueError, match=match):
        platform.set_mit_gains(loaded, kp, kd)


# The second command asks for 3*(0.5-0.1) + 0.7*(0-0.1) + 0.2 = 1.33 N*m, above the 0.5 N*m forcerange.
@pytest.mark.parametrize(("q_des", "tau_ff", "force"), [(0.2, 0.01, 0.24), (0.5, 0.2, 0.5)])
def test_warp_runtime_mit_gains_drive_actual_force_without_gpu(q_des, tau_ff, force):
    mjw = pytest.importorskip("mujoco_warp")
    import warp as wp

    with wp.ScopedDevice("cpu"):
        spec = mujoco.MjSpec.from_string(XML)
        _mjcf.replace_position_actuators(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=1, nconmax=32, njmax=128)
        loaded = SimpleNamespace(model=model, m=m, d=d, joint_names=("j",), actuator_ids=(0,))
        from wuji_sim.platforms import mujoco_warp as warp_platform

        warp_platform.set_mit_gains(loaded, [3.0], [0.7])
        d.qpos.assign([[0.1]])
        d.qvel.assign([[0.1]])
        qd_des = 0.0
        d.ctrl.assign([[3.0 * q_des + 0.7 * qd_des + tau_ff]])
        mjw.step(m, d)

        # The joint's actuatorfrcrange caps qfrc_actuator too; only actuator_force shows the actuator's own clamp.
        assert d.actuator_force.numpy()[0, 0] == pytest.approx(force, abs=1e-5)
        assert d.qfrc_actuator.numpy()[0, 0] == pytest.approx(force, abs=1e-5)
        assert m.actuator_biasprm.numpy()[0, 0, :3].tolist() == pytest.approx([0.0, -3.0, -0.7])



def test_implicit_stability_uses_kp_ratio_and_records_explicit_diagnostic():
    loaded = SimpleNamespace(
        joint_names=("j",),
        joint_inertia=(1.0,),
        model=SimpleNamespace(opt=SimpleNamespace(timestep=0.005, integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST)),
        control_path=mit.INTEGRATED_CONTROL_PATH,
    )
    report = platform.mit_stability(loaded, [1.0], [500.0])
    assert report["control_path"] == mit.INTEGRATED_CONTROL_PATH
    assert report["criterion"] == _mjcf.MIT_STABILITY_CRITERION
    assert report["joints"]["j"]["ratio"] == pytest.approx(0.000025)
    assert report["joints"]["j"]["kd_dt_over_I"] == pytest.approx(2.5)

    with pytest.raises(mit.MitStabilityError, match=r"integrated actuator.*kp\*dt\^2/I"):
        platform.mit_stability(loaded, [4.01 / 0.005**2], [0.0])


@pytest.mark.parametrize("integrator", list(_mjcf.INTEGRATORS))
def test_stability_accepts_only_integrators_that_treat_actuator_damping_implicitly(integrator):
    _, model, _ = compile_replacement(0.005)
    model.opt.integrator = _mjcf.INTEGRATORS[integrator]
    loaded = SimpleNamespace(
        joint_names=("j",), joint_inertia=(1.0,), model=model, control_path=mit.INTEGRATED_CONTROL_PATH
    )
    if integrator in ("implicit", "implicitfast"):
        assert platform.mit_stability(loaded, [1.0], [500.0])["max_ratio"] == pytest.approx(0.000025)
        return
    with pytest.raises(mit.MitStabilityError, match=f"implicit and implicitfast integrators, got {integrator}:"):
        platform.mit_stability(loaded, [1.0], [500.0])


def test_implicit_step_matches_half_millisecond_reference():
    def step(timestep):
        _, model, data = compile_replacement(timestep)
        q_des, qd_des, tau_ff = 0.2, 0.0, 0.0
        for _ in range(round(0.5 / timestep)):
            data.ctrl[0] = 2.0 * q_des + 0.3 * qd_des + tau_ff
            mujoco.mj_step(model, data)
        return float(data.qpos[0])

    assert abs(step(0.005) - step(0.0005)) < 0.001


def test_public_mujoco_profile_selects_five_millisecond_implicit_path():
    profile = platform.load_profile()
    assert profile["params"]["option"]["timestep"]["value"] == 0.005
