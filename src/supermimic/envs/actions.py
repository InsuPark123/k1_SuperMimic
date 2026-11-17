"""Joint-position action with execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the executed set-point. ``executed_raw`` exposes
``a + eps`` for the replay buffer (the world model is fit on executed actions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)


def reset_motor_delay(env, env_ids):
    """Draw a shared 0–4 physics-step delay per environment and hold it for the episode."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, 'motor_delay_lag'):
        env.motor_delay_lag = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    lags = torch.randint(0, 5, (len(env_ids),), device=env.device)
    env.motor_delay_lag[env_ids] = lags
    for actuator in env.scene['robot'].actuators:
        actuator.set_lags(lags, env_ids)


@requires_model_fields('actuator_forcerange', 'jnt_actfrcrange', 'tendon_actfrcrange', 'actuator_gainprm', 'actuator_biasprm')
def randomize_motor_output(env, env_ids, effort_limit_range, asset_cfg):
    """Scale the entire PD output and its torque cap by the same per-motor factor."""
    dr.effort_limits(env, env_ids, effort_limit_range, asset_cfg=asset_cfg, operation='scale')
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    default_force = env.sim.get_default_field('actuator_forcerange')
    for actuator in env.scene[asset_cfg.name].actuators:
        ids = actuator.global_ctrl_ids
        factor = env.sim.model.actuator_forcerange[env_ids[:, None], ids, 1] / default_force[ids, 1]
        env.sim.model.actuator_gainprm[env_ids[:, None], ids, 0] *= factor
        env.sim.model.actuator_biasprm[env_ids[:, None], ids, 1] *= factor
        env.sim.model.actuator_biasprm[env_ids[:, None], ids, 2] *= factor
