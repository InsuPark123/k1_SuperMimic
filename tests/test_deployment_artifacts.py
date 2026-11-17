import json

import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import K1_JOINT_NAMES
from supermimic.deployment import export_k1_policy, prepare_k1_params


def test_params_are_created_once_and_do_not_overwrite_changed_motion(tmp_path):
    motion = tmp_path / 'test.npz'
    np.savez(motion, fps=np.array([50]), joint_names=np.array(K1_JOINT_NAMES),
             body_names=np.array(['pelvis']), body_pos_w=np.zeros((2, 1, 3)),
             body_quat_w=np.array([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]]),
             joint_pos=np.zeros((2, 23)))
    cfg = {'env': {'future_frames': 10, 'anchor_obs': True}, 'policy': {'history': 10, 'goal_dim': 569}}
    run = tmp_path / 'run'
    prepare_k1_params(run, cfg, [motion])
    assert (run / 'params/sim2real.yaml').is_file()
    assert (run / 'params/test.csv').is_file()
    assert json.loads((run / 'params/manifest.json').read_text())['status'].startswith('NOT_DEPLOYABLE')
    prepare_k1_params(run, cfg, [motion])
    motion.write_bytes(b'changed')
    with pytest.raises(ValueError, match='existing params'):
        prepare_k1_params(run, cfg, [motion])


def test_export_versions_do_not_overwrite_previous_checkpoints(tmp_path):
    class Actor(torch.nn.Module):
        def forward(self, observations, goal):
            return observations[:, :23] + goal[:, :23]

    cfg = {'env': {'future_frames': 10, 'anchor_obs': True}, 'policy': {'history': 10, 'goal_dim': 569}}
    ckpt = tmp_path / 'ckpt_latest.pt'
    ckpt.write_bytes(b'first')
    run = tmp_path / 'run'
    first = export_k1_policy(run, ckpt, Actor(), cfg)
    assert (run / 'exported/policy.onnx').read_bytes() == first.read_bytes()
    ckpt.write_bytes(b'second')
    second = export_k1_policy(run, ckpt, Actor(), cfg)
    assert first != second and first.is_file() and second.is_file()
    assert (run / 'exported/policy.onnx').read_bytes() == second.read_bytes()
    assert export_k1_policy(run, ckpt, Actor(), cfg) == second
