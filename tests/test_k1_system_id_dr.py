import pytest
import torch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import K1_LEG_FRICTION
from supermimic.data.motion_convert import K1_JOINT_NAMES
from supermimic.robot import NJ, XI_DIM
from supermimic.envs.sim_context import SimContext
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA MuJoCo Warp')
def test_motor_delay_without_joint_friction_randomization():
    e = make_env(SuperMimicEnvCfgArgs(npz_files=['data/npz/k1_taeguek/taeguek_1.npz'], num_envs=32, replay_capacity=16, k1_system_id_dr=True, motor_delay_dr=True, imu_roll_offset_deg=3, imu_pitch_offset_deg=5), device='cuda:0')
    try:
        e.reset()
        lag = e.motor_delay_lag.clone()
        assert lag.min() >= 0 and lag.max() <= 4 and torch.unique(lag).numel() > 1
        for actuator in e.scene['robot'].actuators:
            torch.testing.assert_close(actuator._delay_buffer.current_lags, lag)
        assert e.recorder_manager.get_term('replay').ctx.read().shape == (32, XI_DIM)
        dof = e.sim.model.dof_frictionloss[:]
        nominal = e.sim.get_default_field('dof_frictionloss')
        torch.testing.assert_close(dof, nominal.expand_as(dof))
        e.step(torch.zeros((32, NJ), device='cuda:0'))
        keep = ~e.reset_buf
        torch.testing.assert_close(e.motor_delay_lag[keep], lag[keep])
    finally:
        e.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA MuJoCo Warp')
@pytest.mark.parametrize('extended', [False, True])
def test_system_id_randomization_reaches_model_and_stays_fixed(extended):
    e = make_env(SuperMimicEnvCfgArgs(npz_files=['data/npz/k1_taeguek/taeguek_1.npz'], num_envs=32, replay_capacity=16, k1_system_id_dr=True, k1_legacy_friction_delay_dr=True, extended_dr_context=extended, obs_noise=False, action_noise_std=0), device='cuda:0')
    try:
        e.reset()
        c = e.recorder_manager.get_term('replay').ctx
        recorded = c.read().clone()
        assert recorded.shape == (32, XI_DIM + 12 + 4 * NJ + 1 if extended else XI_DIM)
        diagnostic = c if extended else SimContext(e, FOOT_GEOM_REGEX, extended=True)
        xi = diagnostic.read().clone()
        torch.testing.assert_close(recorded[:, :XI_DIM], xi[:, :XI_DIM])
        assert xi.shape == (32, XI_DIM + 12 + 4 * NJ + 1)
        assert xi[:, 1:13].abs().max() <= .070001
        assert xi[:, 13:1 + NJ].abs().max() <= .010001
        start = XI_DIM
        for i, name in enumerate(K1_JOINT_NAMES[:12]):
            lo, hi = K1_LEG_FRICTION[name]
            assert xi[:, start + i].min() >= lo - 1e-5
            assert xi[:, start + i].max() <= hi + 1e-5
        start += 12
        kp, kd, strength = xi[:, start:start+NJ], xi[:, start+NJ:start+2*NJ], xi[:, start+2*NJ:start+3*NJ]
        assert strength.min() >= .89999 and strength.max() <= 1.00001
        for gain in (kp / strength, kd / strength):
            assert gain.min() >= .89999 and gain.max() <= 1.10001
        lag = xi[:, start+3*NJ]
        assert lag.min() >= 0 and lag.max() <= 4 and torch.all(lag == lag.round())
        for a in e.scene['robot'].actuators:
            torch.testing.assert_close(a._delay_buffer.current_lags.float(), lag)
        e.step(torch.zeros((32, NJ), device='cuda:0'))
        keep = ~e.reset_buf
        after = diagnostic.read()
        torch.testing.assert_close(after[keep, :start+3*NJ+1], xi[keep, :start+3*NJ+1])
        for a in e.scene['robot'].actuators:
            torch.testing.assert_close(a._delay_buffer.current_lags[keep].float(), lag[keep])
        # A four-physics-step delay must deliver the old target for four writes.
        robot = e.scene['robot']
        for actuator in robot.actuators:
            actuator.set_lags(torch.full((32,), 4, device='cuda:0', dtype=torch.long))
        robot.set_joint_position_target(torch.zeros((32, NJ), device='cuda:0'))
        for _ in range(6):
            e.scene.write_data_to_sim()
        ctrl_id = robot.actuators[0].global_ctrl_ids[0]
        robot.set_joint_position_target(torch.ones((32, NJ), device='cuda:0'))
        for _ in range(4):
            e.scene.write_data_to_sim()
            torch.testing.assert_close(e.sim.data.ctrl[:, ctrl_id], torch.zeros(32, device='cuda:0'))
        e.scene.write_data_to_sim()
        torch.testing.assert_close(e.sim.data.ctrl[:, ctrl_id], torch.ones(32, device='cuda:0'))
        e.reset()
        new_lag = diagnostic.read()[:, start+3*NJ]
        assert torch.any(new_lag != lag)
        for actuator in robot.actuators:
            torch.testing.assert_close(actuator._delay_buffer.current_lags.float(), new_lag)
    finally:
        e.close()
