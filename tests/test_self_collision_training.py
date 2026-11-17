import torch
import pytest
from types import SimpleNamespace
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost
from supermimic.data.replay_buffer import ReplayBuffer

from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.robot import ROBOT, STATE_DIM
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


@pytest.mark.skipif(ROBOT != 'k1', reason='K1 self-collision training')
def test_self_collision_sensor_and_differentiable_prediction():
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], self_collision_weight=-1.0))
    sensor = env_cfg.scene.sensors[0]
    assert sensor.name == 'self_collision' and sensor.history_length == 4
    assert env_cfg.rewards['self_collision'].params['force_threshold'] == 10.0
    wm = WorldModel(WorldModelCfg(hidden=(8,), term_hidden=(8,)), None, 'cpu', self_collision_weight=-1.0)
    state = torch.zeros(2, STATE_DIM)
    state[:, 3] = 1.0
    state.requires_grad_()
    cost = wm.collision_cost(state)
    assert cost.shape == (2,) and ((cost > 0) & (cost < 4)).all()
    cost.sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0


def test_collision_predictor_can_load_previous_world_model():
    cfg = WorldModelCfg(hidden=(8,), term_hidden=(8,))
    old = WorldModel(cfg, None, 'cpu')
    new = WorldModel(cfg, None, 'cpu', self_collision_weight=-1.0)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(name.startswith('collision_head.') for name in missing)
    assert not unexpected


def test_mjlab_self_contact_counts_substeps_above_force_threshold():
    forces = torch.zeros(2, 1, 4, 3)
    forces[0, 0, 1, 0] = 11.0
    forces[0, 0, 3, 0] = 15.0
    forces[1, 0, 2, 0] = 9.0
    sensor = SimpleNamespace(data=SimpleNamespace(force_history=forces))
    env = SimpleNamespace(scene={'self_collision': sensor})
    torch.testing.assert_close(self_collision_cost(env, 'self_collision', 10.0), torch.tensor([2.0, 0.0]))


def test_collision_label_is_aligned_with_transition():
    buf = ReplayBuffer(1, 8, 2, 1, 1, 'cpu')
    for i, label in enumerate((2.0, 0.0)):
        buf.add(torch.tensor([0]), s=torch.tensor([[float(i), 0.0]]), a=torch.zeros(1, 1),
                a_exec=torch.zeros(1, 1), xi=torch.zeros(1, 1), r=torch.zeros(1),
                done=torch.zeros(1), frame=torch.tensor([i]), clip=torch.zeros(1, dtype=torch.long),
                terminal=torch.zeros(1, dtype=torch.bool), ep_start=torch.tensor([i == 0]),
                self_collision=torch.tensor([label]))
    window = buf.sample_windows(1, 1)
    assert window.states[0, 0, 0] == 0 and window.states[1, 0, 0] == 1
    assert window.self_collision[0, 0] == 2
