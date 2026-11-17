"""Show the K1 training environment with its foot collision meshes highlighted."""
import os
from pathlib import Path

os.environ['SUPERMIMIC_ROBOT'] = 'k1'

import mujoco
import torch
import viser
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import ViserPlayViewer
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint

CHECKPOINT = Path('logs/k1_taeguek_27_imu_motor_delay/ckpt_latest.pt')
MOTION = 'data/npz/k1_taeguek/taeguek_1.npz'


class Policy:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs['policy'], obs['goal'])


def main():
    policy, cfg = load_policy_from_checkpoint(CHECKPOINT, 'cuda:0')
    args = SuperMimicEnvCfgArgs(
        npz_files=[MOTION], num_envs=1, future_frames=cfg['env']['future_frames'],
        episode_length_s=1e5, action_noise_std=0, obs_noise=False,
        reset_noise=False, domain_randomization=False, sampling_mode='start',
        record_transitions=False, seed=0,
        anchor_obs=cfg['env'].get('anchor_obs', False),
        anchor_ori_obs=cfg['env'].get('anchor_ori_obs', False), debug_vis=False,
    )
    env = RslRlVecEnvWrapper(make_env(args, device='cuda:0'))
    model = env.unwrapped.sim.mj_model
    for side in ('left', 'right'):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'robot/{side}_ankle_roll_link')
        collision = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'robot/{side}_ankle_roll_link_collision_mesh')
        if body < 0 or collision < 0:
            raise RuntimeError(f'Missing {side} foot mesh')
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == body and geom != collision and model.geom_contype[geom] == 0:
                model.geom_rgba[geom, 3] = 0.0
        model.geom_group[collision] = 2  # Viewer appearance only; physics is unchanged.
        model.geom_rgba[collision] = (1.0, 0.48, 0.02, 1.0)
    server = viser.ViserServer(host='127.0.0.1', port=8080, label='K1 foot mesh')
    print('[INFO] Training K1 mesh feet highlighted in orange; Taeguek 1, motor-delay checkpoint.', flush=True)
    try:
        ViserPlayViewer(env, Policy(policy), viser_server=server).run()
    finally:
        env.close()
        server.stop()


if __name__ == '__main__':
    main()
