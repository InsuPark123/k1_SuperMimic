import numpy as np
import torch

from scripts.export_k1 import SingleInputActor, motion_csv_rows


def test_single_input_actor_splits_history_and_goal():
    class Actor(torch.nn.Module):
        def forward(self, obs, goal):
            return torch.cat((obs[:, :1], goal[:, :1]), dim=-1)

    model = SingleInputActor(Actor(), 3)
    actual = model(torch.tensor([[1., 2., 3., 4., 5.]]))
    torch.testing.assert_close(actual, torch.tensor([[1., 4.]]))


def test_motion_csv_preserves_root_and_converts_quaternion_order():
    body_pos = np.array([[[1., 2., 3.]]])
    body_quat = np.array([[[0.5, 0.1, 0.2, 0.3]]])
    joints = np.array([[0.6, 0.7]])
    rows = motion_csv_rows(body_pos, body_quat, joints, 0)
    np.testing.assert_allclose(rows, [[1., 2., 3., 0.1, 0.2, 0.3, 0.5, 0.6, 0.7]])
