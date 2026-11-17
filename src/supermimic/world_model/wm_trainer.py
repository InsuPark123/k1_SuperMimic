"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import ReferenceFrame
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam((p for n, p in wm.named_parameters() if not n.startswith('collision_head.')), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.collision_opt = torch.optim.Adam(wm.collision_head.parameters(), lr=cfg.lr) if wm.collision_head is not None else None
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        if self.collision_opt is not None:
            for g in self.collision_opt.param_groups:
                g['lr'] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            self.opt.step()
            if self.collision_opt is not None:
                # Sensor labels are measured over four physics substeps; prediction is a
                # differentiable approximation conditioned on the observed next state.
                target = w.self_collision.reshape(-1) / 4.0
                predicted = self.wm.collision_cost(w.states[1:].reshape(-1, w.states.shape[-1])) / 4.0
                collision_loss = torch.nn.functional.binary_cross_entropy(predicted, target)
                self.collision_opt.zero_grad(set_to_none=True)
                collision_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wm.collision_head.parameters(), self.cfg.grad_clip)
                self.collision_opt.step()
                self.collision_opt.zero_grad(set_to_none=True)
                m['collision_bce'] = collision_loss.detach()
                m['collision_target'] = target.mean().detach()
                m['collision_prediction'] = predicted.mean().detach()
                positive = target > 0
                m['collision_positive_count'] = positive.sum().detach()
                m['collision_recall'] = ((predicted > 0.1) & positive).sum().detach() / positive.sum().clamp(min=1)
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
