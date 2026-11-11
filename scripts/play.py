"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): [ / ] previous / next motion, Enter reset, space pause,
right-arrow single step, +/- speed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


class MotionViewer(NativeMujocoViewer):
    """Switch clips on the simulation thread when the viewer receives a key."""

    def __init__(self, env, policy):
        super().__init__(env, policy, key_callback=self.on_key)
        self.motion = env.unwrapped.command_manager.get_term("motion")
        self.clip_index = 0
        self.select_clip(0)
        print("[INFO] Use [ and ] to switch motions; Enter restarts the current motion.")

    def on_key(self, key: int) -> None:
        if key == KEY_LEFT_BRACKET:
            self.request_action("CUSTOM", -1)
        elif key == KEY_RIGHT_BRACKET:
            self.request_action("CUSTOM", 1)

    def select_clip(self, index: int) -> None:
        clips = self.motion.library.clips
        self.clip_index = index % len(clips)
        clip = clips[self.clip_index]
        self.motion.forced_frame[:] = clip.start
        self.reset_environment()
        print(f"[INFO] Motion {self.clip_index + 1}/{len(clips)}: {clip.name}", flush=True)

    def _handle_custom_action(self, action, payload):
        if action == ViewerAction.CUSTOM and payload in (-1, 1):
            self.select_clip(self.clip_index + payload)
            return True
        return super()._handle_custom_action(action, payload)


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=ghost,
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    pol = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] {len(files)} clip(s), {num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            MotionViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
