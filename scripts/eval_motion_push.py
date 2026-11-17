"""Test random horizontal velocity pushes during motion tracking in simulation."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.kinematics.rotations import heading_yaw, wrap_to_pi
from supermimic.robot import ROBOT


def make_trials(clips, speeds, samples, push_window, seed):
    rng = random.Random(seed)
    trials = []
    for clip in clips:
        perturbations = [(rng.uniform(*push_window), rng.uniform(0, 2 * math.pi)) for _ in range(samples)]
        trials.append(dict(clip=clip, speed=0.0, push_time=None, angle=None))
        for speed in speeds:
            for time, angle in perturbations:
                trials.append(dict(clip=clip, speed=speed, push_time=time, angle=angle))
    return trials


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path = Path('data/npz/k1_taeguek'),
         clips: list[str] = ['taeguek_1', 'taeguek_4', 'taeguek_6'],
         speeds: tuple[float, ...] = (0.2, 0.5, 1.0), samples: int = 6,
         push_window: tuple[float, float] = (3.0, 12.0), seconds: float = 20.0,
         seed: int = 0, device: str = 'cuda:0',
         out: Path = Path('logs/k1_taeguek_xy_yaw/motion_push_seed0.json')) -> None:
    if not clips or len(set(clips)) != len(clips) or samples < 1 or any(s <= 0 for s in speeds):
        raise ValueError('Use distinct clips, positive speeds, and at least one sample')
    if push_window[0] < 0 or push_window[0] >= push_window[1] or push_window[1] + 2 > seconds:
        raise ValueError('Push window must fit at least two seconds before the end of the horizon')
    files = discover_npz(npz_dir, names=clips)
    if len(files) != len(clips):
        raise ValueError(f'Missing clips: {set(clips) - {p.stem for p in files}}')
    names = [p.stem for p in files]
    trials = make_trials(names, speeds, samples, push_window, seed)
    policy, config = load_policy_from_checkpoint(ckpt, device)
    settings = config['env']
    reward_args = {k: settings[k] for k in ('base_xy_weight', 'base_yaw_weight', 'base_xy_std', 'base_yaw_std') if k in settings}
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=len(trials),
        future_frames=settings['future_frames'], episode_length_s=1e5, action_noise_std=0.0,
        obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode='start',
        record_transitions=False, seed=seed, anchor_obs=settings.get('anchor_obs', False), anchor_ori_obs=settings.get('anchor_ori_obs', False), **reward_args)
    env = make_env(args, device)
    try:
        cmd = env.command_manager.get_term('motion')
        if any(c.length < round(seconds / env.step_dt) + 1 for c in cmd.library.clips):
            raise ValueError('A selected motion is shorter than the evaluation horizon')
        indices = torch.tensor([names.index(t['clip']) for t in trials], device=device)
        cmd.forced_frame[:] = cmd.library.clip_start[indices]
        obs, _ = env.reset()
        n = len(trials)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        pushed = torch.zeros(n, dtype=torch.bool, device=device)
        speed = torch.tensor([t['speed'] for t in trials], device=device)
        angle = torch.tensor([t['angle'] or 0.0 for t in trials], device=device)
        push_steps = torch.tensor([round(t['push_time'] / env.step_dt) if t['push_time'] is not None else -1 for t in trials], device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)
        rewards = torch.zeros(n, device=device)
        err_xy = torch.full((n,), float('nan'), device=device)
        err_yaw = torch.full((n,), float('nan'), device=device)
        terminated = torch.zeros(n, dtype=torch.bool, device=device)
        robot = env.scene['robot']
        for step in range(round(seconds / env.step_dt)):
            ids = ((push_steps == step) & alive).nonzero().squeeze(-1)
            if ids.numel():
                velocity = robot.data.root_link_vel_w[ids].clone()
                velocity[:, 0] += speed[ids] * torch.cos(angle[ids])
                velocity[:, 1] += speed[ids] * torch.sin(angle[ids])
                robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
                pushed[ids] = True
            obs, reward, term, timeout, _ = env.step(policy(obs['policy'], obs['goal']))
            rewards += reward * alive
            steps += alive.long()
            terminated |= term & alive
            alive &= ~(term | timeout)
            check = (step + 1 == push_steps + round(2.0 / env.step_dt)) & alive & pushed
            if check.any():
                delta = cmd.robot_root_pos_w[check, :2] - cmd.root_pos_w[check, :2]
                err_xy[check] = torch.linalg.vector_norm(delta, dim=-1)
                err_yaw[check] = wrap_to_pi(heading_yaw(cmd.robot_root_quat_w[check]) - heading_yaw(cmd.root_quat_w[check])).abs()
        max_reward = env.step_dt * (5.0 + settings.get('base_xy_weight', 0.0) + settings.get('base_yaw_weight', 0.0))
        for i, trial in enumerate(trials):
            def number(value):
                v = float(value)
                return v if math.isfinite(v) else None
            trial.update(push_applied=bool(pushed[i]), terminated=bool(terminated[i]),
                completed=bool(alive[i]), elapsed_s=float(steps[i]) * env.step_dt,
                normalized_return=float(rewards[i] / (max_reward * steps[i].clamp(min=1))),
                xy_error_2s_m=number(err_xy[i]), yaw_error_2s_deg=number(torch.rad2deg(err_yaw[i])))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'checkpoint': str(ckpt), 'robot': ROBOT, 'seed': seed,
            'horizon_seconds': seconds, 'push_window_seconds': list(push_window),
            'push_type': 'instantaneous horizontal root velocity change, m/s',
            'domain_randomization': False, 'observation_noise': False, 'trials': trials}, indent=2))
        for s in (0.0, *speeds):
            group = [t for t in trials if t['speed'] == s]
            print(f"RESULT delta_v={s:.2f} m/s: completed {sum(t['completed'] for t in group)}/{len(group)}; push applied {sum(t['push_applied'] for t in group)}", flush=True)
        print(f'Saved {out}')
    finally:
        env.close()


if __name__ == '__main__':
    tyro.cli(main)
