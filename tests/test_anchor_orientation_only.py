import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import K1_JOINT_NAMES
from supermimic.deployment import prepare_k1_params, export_k1_policy
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg


def test_orientation_only_goal_keeps_orientation_without_position():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused.npz'], anchor_ori_obs=True))
    terms = cfg.observations['goal'].terms
    assert list(terms) == ['future_frames', 'anchor_ori']


def test_orientation_only_params_accepts_566_goal_values(tmp_path):
    motion = tmp_path / 'test.npz'
    np.savez(motion, fps=np.array([50]), joint_names=np.array(K1_JOINT_NAMES),
             body_names=np.array(['pelvis']), body_pos_w=np.zeros((2, 1, 3)),
             body_quat_w=np.array([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]]),
             joint_pos=np.zeros((2, 23)))
    cfg = {'env': {'future_frames': 10, 'anchor_ori_obs': True}, 'policy': {'history': 10, 'goal_dim': 566}}
    assert (prepare_k1_params(tmp_path / 'run', cfg, [motion]) / 'sim2real.yaml').exists()


def test_incompatible_anchor_modes_are_rejected():
    with pytest.raises(ValueError, match='anchor'):
        make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused.npz'], anchor_obs=True, anchor_ori_obs=True))


def test_orientation_only_actor_can_be_exported(tmp_path):
    class Actor(torch.nn.Module):
        def forward(self, observations, goal):
            return observations[:, :23] + goal[:, :23]

    ckpt = tmp_path / 'checkpoint.pt'
    ckpt.write_bytes(b'test checkpoint')
    cfg = {'env': {'future_frames': 10, 'anchor_ori_obs': True}, 'policy': {'history': 10, 'goal_dim': 566}}
    assert export_k1_policy(tmp_path / 'run', ckpt, Actor(), cfg).is_file()
