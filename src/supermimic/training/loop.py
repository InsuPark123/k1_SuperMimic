"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_robot_model, K1_JOINT_NAMES
from supermimic.robot import GOAL_FEAT_DIM, NJ, ROBOT
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES, K1_LEG_FRICTION
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


def configure_encoder_bias(cfg: TrainConfig) -> None:
    half_range = cfg.env.encoder_bias_range
    other = cfg.env.encoder_bias_other_range
    if not 0 < half_range < float('inf') or not 0 < other < float('inf'):
        raise ValueError('encoder bias half-range must be positive and finite')
    spec = cfg.world_model.state_spec
    scales = spec.xi_scale
    bias_scales = (half_range,) * 12 + (other,) * (NJ - 12) if ROBOT == 'k1' else (half_range,) * NJ
    spec.xi_scale = scales[:1] + bias_scales + scales[1 + NJ:]
    if cfg.env.k1_system_id_dr:
        if ROBOT != 'k1' or not cfg.env.domain_randomization:
            raise ValueError('K1 system-ID randomization requires K1 and domain randomization')
    if cfg.env.extended_dr_context and not cfg.env.k1_system_id_dr:
        raise ValueError('Extended DR context requires K1 system-ID randomization')
    if cfg.env.k1_legacy_friction_delay_dr and not cfg.env.k1_system_id_dr:
        raise ValueError('Legacy friction and delay randomization requires K1 system-ID randomization')
    if cfg.env.motor_delay_dr and not cfg.env.k1_system_id_dr:
        raise ValueError('Motor delay randomization requires K1 system-ID randomization')
    if cfg.env.extended_dr_context and not cfg.env.k1_legacy_friction_delay_dr:
        raise ValueError('Extended DR context requires legacy friction and delay randomization')
    if cfg.env.extended_dr_context:
        if len(spec.xi_scale) != NJ + 4:
            raise ValueError('Unexpected base world-model context dimensions')
        bounds = [K1_LEG_FRICTION[name] for name in K1_JOINT_NAMES[:12]]
        spec.xi_offset += tuple((lo + hi) / 2 for lo, hi in bounds) + (1.,) * (2 * NJ) + (0.95,) * NJ + (2.,) + (0.,) * NJ
        spec.xi_scale += tuple((hi - lo) / 2 for lo, hi in bounds) + (0.1,) * (2 * NJ) + (0.05,) * NJ + (2.,) + (1.,) * NJ
    if cfg.resume is not None:
        ck = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        old = ck['config']['world_model']['state_spec']['xi_scale'][1:1 + NJ]
        if any(abs(value - target) > 1e-9 for value, target in zip(old, bias_scales)):
            raise ValueError('Encoder bias range differs from the checkpoint world-model normalization; start a new run or use the checkpoint range.')
        saved_spec = ck['config']['world_model']['state_spec']
        if (len(saved_spec['xi_scale']) != len(spec.xi_scale) or len(saved_spec['xi_offset']) != len(spec.xi_offset) or
            any(abs(a - b) > 1e-9 for a, b in zip(saved_spec['xi_scale'], spec.xi_scale)) or
            any(abs(a - b) > 1e-9 for a, b in zip(saved_spec['xi_offset'], spec.xi_offset))):
            raise ValueError('Checkpoint world-model context differs from the requested domain randomization; start a new run.')


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        configure_encoder_bias(cfg)
        if cfg.env.anchor_obs and cfg.env.anchor_ori_obs:
            raise ValueError('anchor_obs and anchor_ori_obs are mutually exclusive')
        cfg.policy.goal_dim = cfg.env.future_frames * GOAL_FEAT_DIM + (9 if cfg.env.anchor_obs else 6 if cfg.env.anchor_ori_obs else 0)
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.anchor_ori_obs = cfg.env.anchor_ori_obs
        cfg.policy_trainer.imu_pitch_offset_deg = cfg.env.imu_pitch_offset_deg
        cfg.policy_trainer.imu_roll_offset_deg = cfg.env.imu_roll_offset_deg
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1 and cfg.env.self_collision_weight:
            raise ValueError('Self-collision predictor currently supports one GPU only')
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        if self.is_main and ROBOT == 'k1':
            from supermimic.deployment import prepare_k1_params
            params_dir = prepare_k1_params(self.log_dir, cfg, files)
            print(f'[INFO] K1 research-draft parameters: {params_dir} (not robot-deployable)')
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                encoder_bias_range=cfg.env.encoder_bias_range,
                encoder_bias_other_range=cfg.env.encoder_bias_other_range,
                k1_system_id_dr=cfg.env.k1_system_id_dr,
                k1_legacy_friction_delay_dr=cfg.env.k1_legacy_friction_delay_dr,
                motor_delay_dr=cfg.env.motor_delay_dr,
                self_collision_weight=cfg.env.self_collision_weight,
                extended_dr_context=cfg.env.extended_dr_context,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                anchor_ori_obs=cfg.env.anchor_ori_obs,
                imu_pitch_offset_deg=cfg.env.imu_pitch_offset_deg,
                imu_roll_offset_deg=cfg.env.imu_roll_offset_deg,
                base_xy_weight=cfg.env.base_xy_weight,
                base_yaw_weight=cfg.env.base_yaw_weight,
                base_xy_std=cfg.env.base_xy_std,
                base_yaw_std=cfg.env.base_yaw_std,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_robot_model(), device=self.device)
        reward_cfg = TrackingRewardCfg(w_base_xy=cfg.env.base_xy_weight, w_base_yaw=cfg.env.base_yaw_weight,
                                       std_base_xy=cfg.env.base_xy_std, std_base_yaw=cfg.env.base_yaw_std)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(), reward_cfg=reward_cfg)
        self.wm = WorldModel(cfg.world_model, heads, self.device, self_collision_weight=cfg.env.self_collision_weight)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, xi_dim=len(cfg.world_model.state_spec.xi_scale), pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.start_iteration = 0  # warm-up (buffer refill) counts from here, also after a resume
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "robot": ROBOT,
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "collision_opt": self.wm_trainer.collision_opt.state_dict() if self.wm_trainer.collision_opt is not None else None,
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("robot", "g1") != ROBOT:
            raise ValueError(f"Checkpoint robot {ck.get('robot', 'g1')!r} does not match {ROBOT!r}")
        missing, unexpected = self.wm.load_state_dict(ck["wm"], strict=False)
        if unexpected or any(not k.startswith('collision_head.') for k in missing):
            raise ValueError(f'Incompatible world model checkpoint: missing={missing}, unexpected={unexpected}')
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        if self.wm_trainer.collision_opt is not None and ck.get('collision_opt') is not None:
            self.wm_trainer.collision_opt.load_state_dict(ck['collision_opt'])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = ck["iteration"]
        self.start_iteration = self.iteration
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()
        collision_ready = self.wm.collision_head is None

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - self.start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    m_pi = {}
                    # Learn the new sensor predictor before its gradients affect the actor.
                    collision_ready = (collision_ready or self.wm.collision_head is None or (
                        self.iteration - self.start_iteration >= 1000
                        and m_wm.get('collision_positive_count', 0) >= 10
                        and m_wm.get('collision_recall', 0) >= 0.5))
                    if collision_ready:
                        for _ in range(sch.policy_updates_per_wm_update):
                            m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    if self.wm.collision_head is not None:
                        msg += (f" | collision target/pred {agg.get('wm/collision_target', 0)/max(agg_n,1):.4f}/"
                                f"{agg.get('wm/collision_prediction', 0)/max(agg_n,1):.4f}"
                                f" recall {agg.get('wm/collision_recall', 0)/max(agg_n,1):.3f}"
                                f" policy_ready {collision_ready}")
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if self.wm.collision_head is not None and self.iteration - self.start_iteration >= 3000 and not collision_ready:
                print('[ERROR] Contact predictor did not reach the readiness threshold; stopping before policy updates.', flush=True)
                break
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
