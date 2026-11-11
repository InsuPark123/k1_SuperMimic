"""Compare a G1 checkpoint under controlled disturbances on motion prefixes.

Example: uv run python scripts/eval_robustness.py --ckpt logs/g1_lafan1/ckpt_latest.pt
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import ConstantNoiseCfg, UniformNoiseCfg
from mjlab.utils.torch import configure_torch_backends

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.eval.benchmark import load_policy_from_checkpoint


def make_scenario_cfg(args, scenario):
    cfg = make_env_cfg(args)
    if scenario.startswith('friction_'):
        friction = float(scenario.removeprefix('friction_'))
        cfg.events['foot_friction'].params['ranges'] = (friction, friction)
    elif scenario == 'gyro_noise':
        cfg.observations['policy'].enable_corruption = True
        cfg.observations['policy'].terms['base_ang_vel'].noise = UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    elif scenario == 'gyro_bias':
        cfg.observations['policy'].enable_corruption = True
        cfg.observations['policy'].terms['base_ang_vel'].noise = ConstantNoiseCfg(bias=(0.2, 0.0, 0.0))
    elif scenario.startswith('density_'):
        scale = float(scenario.removeprefix('density_'))
        alpha = math.log(scale) / 2
        cfg.events['density'] = EventTermCfg(mode='startup', func=dr.pseudo_inertia, params={
            'asset_cfg': SceneEntityCfg('robot', body_names=('.*',)), 'alpha_range': (alpha, alpha),
        })
    elif scenario != 'baseline':
        raise ValueError(f'Unknown scenario: {scenario}')
    return cfg


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path = Path('data/npz/lafan1'),
         clips: list[str] = ['walk1_subject1', 'dance1_subject1', 'jumps1_subject1'],
         seconds: float = 20.0, seed: int = 0, device: str = 'cuda:0',
         out: Path = Path('logs/g1_lafan1/robustness_prefix.json')) -> None:
    if seconds <= 0:
        raise ValueError('seconds must be positive')
    files = discover_npz(npz_dir, names=clips)
    if len(files) != len(set(clips)):
        raise ValueError(f'Missing clips: {set(clips) - {p.stem for p in files}}')
    policy, config = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=len(files),
        future_frames=config['env']['future_frames'], episode_length_s=1e5,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode='start', record_transitions=False, seed=seed,
        anchor_obs=config['env'].get('anchor_obs', False))
    scenarios = ['baseline', 'friction_0.3', 'friction_0.1', 'gyro_noise', 'gyro_bias', 'density_0.8', 'density_1.2']
    results = []
    configure_torch_backends()
    for scenario in scenarios:
        env = ManagerBasedRlEnv(make_scenario_cfg(args, scenario), device=device)
        try:
            cmd = env.command_manager.get_term('motion')
            if any(c.length < round(seconds / env.step_dt) + 1 for c in cmd.library.clips):
                raise ValueError('A selected clip is shorter than the requested evaluation horizon')
            cmd.forced_frame[:] = cmd.library.clip_start
            obs, _ = env.reset()
            alive = torch.ones(len(files), dtype=torch.bool, device=device)
            reward = torch.zeros(len(files), device=device)
            steps = torch.zeros(len(files), dtype=torch.long, device=device)
            for _ in range(round(seconds / env.step_dt)):
                obs, rew, term, timeout, _ = env.step(policy(obs['policy'], obs['goal']))
                reward += rew * alive
                steps += alive.long()
                alive &= ~(term | timeout)
                if not alive.any():
                    break
            for i, clip in enumerate(cmd.library.clips):
                results.append({'scenario': scenario, 'clip': clip.name, 'steps': int(steps[i]),
                    'seconds': float(steps[i]) * env.step_dt, 'survived': bool(alive[i]),
                    'norm_return': float(reward[i] / (0.1 * steps[i].clamp(min=1)))})
            recent = results[-len(files):]
            print(f"RESULT {scenario}: survival {sum(x['survived'] for x in recent)}/{len(files)}, "
                  f"mean return {sum(x['norm_return'] for x in recent)/len(files):.3f}", flush=True)
        finally:
            env.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'checkpoint': str(ckpt), 'seed': seed, 'horizon_seconds': seconds,
                               'results': results}, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    tyro.cli(main)
