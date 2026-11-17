"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
"""

from __future__ import annotations

from pathlib import Path

import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize
from supermimic.robot import ROBOT


def main(ckpt: Path, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, num_envs: int = 1024, device: str = "cuda:0", out: Path | None = None) -> None:
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    if ROBOT == 'k1':
        from supermimic.deployment import export_k1_policy, prepare_k1_params
        data = cfg['data']
        if data.get('library'):
            raise ValueError('K1 automatic export requires individual training NPZ files')
        training_files = discover_npz(data['npz_dir'], limit=data.get('limit'), names=data.get('clips'), manifest=data.get('manifest'))
        prepare_k1_params(ckpt.parent, cfg, training_files)
        onnx_path = export_k1_policy(ckpt.parent, ckpt, policy, cfg)
        print(f'[INFO] Exported research-draft actor: {onnx_path} (not robot-deployable)', flush=True)
    env_cfg = cfg['env']
    reward_args = {k: env_cfg[k] for k in ('base_xy_weight', 'base_yaw_weight', 'base_xy_std', 'base_yaw_std') if k in env_cfg}
    results = run_benchmark(policy, files, device=device, num_envs=num_envs, future_frames=env_cfg["future_frames"], anchor_obs=env_cfg.get("anchor_obs", False), anchor_ori_obs=env_cfg.get("anchor_ori_obs", False), **reward_args)
    s = summarize(results)
    print(f"\n{s['clips']} clips | failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f}")
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    save_results(results, out)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
