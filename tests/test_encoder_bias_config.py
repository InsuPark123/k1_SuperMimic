import torch
import pytest
from supermimic.config import TrainConfig
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.training.loop import configure_encoder_bias
from supermimic.world_model.state_spec import StateSpec
from supermimic.robot import NJ


def test_k1_bias_range_and_world_model_scaling():
    cfg = TrainConfig()
    assert cfg.env.encoder_bias_range == 0.07
    assert cfg.env.imu_pitch_offset_deg == 0.0
    configure_encoder_bias(cfg)
    assert cfg.world_model.state_spec.xi_scale[1:1 + NJ] == (0.07,) * 12 + (0.01,) * (NJ - 12)
    spec = StateSpec(cfg.world_model.state_spec, device='cpu')
    xi = torch.tensor(list(cfg.world_model.state_spec.xi_offset))
    xi[1:13] = 0.07
    xi[13:1 + NJ] = 0.01
    torch.testing.assert_close(spec.normalize_xi(xi)[1:1 + NJ], torch.ones(NJ))
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], encoder_bias_range=0.07))
    assert env_cfg.events['encoder_bias_legs'].params['bias_range'] == (-0.07, 0.07)
    assert env_cfg.events['encoder_bias_other'].params['bias_range'] == (-0.01, 0.01)
    assert 'imu_pitch_offset' not in env_cfg.events


def test_reject_invalid_bias():
    cfg = TrainConfig()
    cfg.env.encoder_bias_range = -0.1
    with pytest.raises(ValueError):
        configure_encoder_bias(cfg)


def test_k1_system_id_profile_configures_all_requested_events():
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], k1_system_id_dr=True, k1_legacy_friction_delay_dr=True))
    assert env_cfg.events['encoder_bias_legs'].params['bias_range'] == (-0.07, 0.07)
    assert env_cfg.events['encoder_bias_other'].params['bias_range'] == (-0.01, 0.01)
    assert env_cfg.events['pd_gains'].params['kp_range'] == (0.9, 1.1)
    assert env_cfg.events['pd_gains'].params['kd_range'] == (0.9, 1.1)
    assert env_cfg.events['motor_output'].params['effort_limit_range'] == (0.9, 1.0)
    assert len([key for key in env_cfg.events if key.startswith('leg_friction_')]) == 12
    assert all(a.delay_max_lag == 4 and a.delay_hold_prob == 1 for a in env_cfg.scene.entities['robot'].articulation.actuators)


def test_world_model_updates_previous_action_in_context_each_step():
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.heads import ReferenceFrame
    cfg = TrainConfig()
    cfg.env.extended_dr_context = True
    cfg.env.k1_legacy_friction_delay_dr = True
    configure_encoder_bias(cfg)
    model = WorldModel(WorldModelCfg(hidden=(4, 4), state_spec=cfg.world_model.state_spec), None, 'cpu')
    seen = []
    def step(state, action, xi, target, done, pos, quat):
        seen.append(xi.clone())
        return state, torch.zeros(2, 1), torch.zeros(2), torch.zeros(2)
    model._loss_step_fn = step
    actions = torch.stack([torch.full((2, NJ), float(i + 1)) for i in range(3)])
    xi = torch.zeros(2, len(cfg.world_model.state_spec.xi_scale))
    xi[:, -NJ:] = -1
    refs = [ReferenceFrame(None, None, None, None, torch.zeros(2, 3), torch.zeros(2, 4)) for _ in range(3)]
    model.window_loss(torch.zeros(4, 2, 1), actions, xi, torch.zeros(3, 2), refs)
    torch.testing.assert_close(seen[0][:, -NJ:], torch.full((2, NJ), -1.))
    torch.testing.assert_close(seen[1][:, -NJ:], actions[0])
    torch.testing.assert_close(seen[2][:, -NJ:], actions[1])


def test_system_id_dr_with_27_value_world_model_context():
    cfg = TrainConfig()
    assert cfg.env.k1_system_id_dr
    cfg.env.extended_dr_context = False
    configure_encoder_bias(cfg)
    assert len(cfg.world_model.state_spec.xi_scale) == 27
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], k1_system_id_dr=True, extended_dr_context=False))
    assert env_cfg.recorders['replay'].params['extended_dr'] is False
    assert not any(k.startswith('leg_friction_') for k in env_cfg.events)
    assert 'pd_gains' in env_cfg.events and 'motor_output' in env_cfg.events and 'motor_delay' not in env_cfg.events
    assert all(a.delay_max_lag == 0 for a in env_cfg.scene.entities['robot'].articulation.actuators)


def test_extended_context_requires_system_id_dr():
    cfg = TrainConfig()
    cfg.env.k1_system_id_dr = False
    cfg.env.extended_dr_context = True
    with pytest.raises(ValueError):
        configure_encoder_bias(cfg)


def test_k1_training_disables_friction_and_delay_randomization():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], k1_system_id_dr=True))
    assert not any(name.startswith('leg_friction_') for name in cfg.events)
    assert 'motor_delay' not in cfg.events
    assert all(actuator.delay_max_lag == 0 for actuator in cfg.scene.entities['robot'].articulation.actuators)
    assert {'base_com', 'encoder_bias_legs', 'encoder_bias_other', 'foot_friction', 'pd_gains', 'motor_output'} <= cfg.events.keys()


def test_motor_delay_can_be_enabled_without_joint_friction():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], k1_system_id_dr=True, motor_delay_dr=True, imu_pitch_offset_deg=5, imu_roll_offset_deg=3))
    assert cfg.events['motor_delay'].mode == 'reset'
    assert not any(name.startswith('leg_friction_') for name in cfg.events)
    assert all(a.delay_min_lag == 0 and a.delay_max_lag == 4 and a.delay_hold_prob == 1 for a in cfg.scene.entities['robot'].articulation.actuators)
    assert cfg.events['imu_pitch_offset'].params == {'pitch_half_range_deg': 5, 'roll_half_range_deg': 3}
