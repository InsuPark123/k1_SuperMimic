import math
import os
from pathlib import Path

import pytest
import torch

from supermimic.envs.mdp.rewards import base_tracking_rewards
from supermimic.kinematics.rotations import quat_from_yaw


def test_xy_and_yaw_reward_and_angle_wrap():
    ref_pos = torch.zeros(3, 3)
    pos = torch.tensor([[0., 0., 2.], [0.3, 0., 0.], [0., 0., 0.]], requires_grad=True)
    ref_yaw = torch.tensor([0., 0., math.pi - 0.01])
    yaw = torch.tensor([0., 0.4, -math.pi + 0.01], requires_grad=True)
    xy, heading = base_tracking_rewards(pos, quat_from_yaw(yaw), ref_pos, quat_from_yaw(ref_yaw), 0.3, 0.4)
    torch.testing.assert_close(xy, torch.tensor([1., math.exp(-1), 1.]))
    torch.testing.assert_close(heading, torch.tensor([1., math.exp(-1), math.exp(-0.02**2 / 0.4**2)]))
    (xy.sum() + heading.sum()).backward()
    assert torch.isfinite(pos.grad).all() and torch.isfinite(yaw.grad).all()


@pytest.mark.skipif(os.environ.get('SUPERMIMIC_ROBOT') != 'k1' or not torch.cuda.is_available(), reason='K1 CUDA integration test')
def test_k1_environment_and_imagined_rewards_agree():
    from supermimic.data.motion_convert import load_robot_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.rewards import TrackingRewardCfg
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.wm_trainer import reference_at

    motion = Path('data/npz/k1_taeguek/taeguek_1.npz')
    if not motion.exists():
        pytest.skip('K1 motion unavailable')
    args = SuperMimicEnvCfgArgs(npz_files=[str(motion)], num_envs=2, obs_noise=False,
        reset_noise=False, domain_randomization=False, record_transitions=False,
        anchor_obs=True, base_xy_weight=1.0, base_yaw_weight=1.0)
    env = make_env(args, device='cuda:0')
    try:
        obs, _ = env.reset()
        cmd = env.command_manager.get_term('motion')
        robot = env.scene['robot']
        heads = AnalyticHeads(G1Kinematics(load_robot_model(), device=env.device), TRACKED_BODIES,
            ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0],
            reward_cfg=TrackingRewardCfg(w_base_xy=1.0, w_base_yaw=1.0))
        assert obs['goal'].shape[-1] == 10 * 56 + 9
        for _ in range(3):
            env.step(torch.randn(2, 23, device=env.device) * 0.1)
            terms = heads.reward(StateIO(env).read(), env.action_manager.action,
                env.action_manager.prev_action, reference_at(cmd.library, cmd.frame))
            for name in ('base_xy', 'base_yaw'):
                actual = env.reward_manager.get_term_cfg(name).func(env, command_name='motion', std=0.3 if name == 'base_xy' else 0.4)
                torch.testing.assert_close(terms[name] / env.step_dt, actual, atol=1e-5, rtol=1e-5)
    finally:
        env.close()
