"""Per-environment simulation context xi (plan: 33-d = foot friction 1 + encoder bias 29 + torso COM offset 3)."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.robot import XI_DIM, NJ


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str = "torso_link", entity_name: str = "robot", extended: bool = False):
        self.env = env
        self.robot = env.scene[entity_name]
        self.extended = extended
        self.dim = XI_DIM + (12 + 3 * NJ + 1 + NJ if extended else 0)
        if extended:
            joint_ids = self.robot.indexing.joint_ids.cpu().numpy()
            self.leg_dof_ids = torch.as_tensor(env.sim.mj_model.jnt_dofadr[joint_ids[:12]], device=env.device, dtype=torch.long)
            self.ctrl_ids = torch.empty(NJ, device=env.device, dtype=torch.long)
            for actuator in self.robot.actuators:
                self.ctrl_ids[actuator.target_ids] = actuator.global_ctrl_ids
            default_model = env.sim.mj_model
            ctrl = self.ctrl_ids.cpu().numpy()
            self.nominal_kp = torch.as_tensor(default_model.actuator_gainprm[ctrl, 0], device=env.device, dtype=torch.float32)
            self.nominal_kd = torch.as_tensor(-default_model.actuator_biasprm[ctrl, 2], device=env.device, dtype=torch.float32)
            self.nominal_force = torch.as_tensor(default_model.actuator_forcerange[ctrl, 1], device=env.device, dtype=torch.float32)
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)
        if self.extended:
            motor = self.env.action_manager.get_term('joint_pos')
            extra = [model.dof_frictionloss[:, self.leg_dof_ids],
                     model.actuator_gainprm[:, self.ctrl_ids, 0] / self.nominal_kp,
                     -model.actuator_biasprm[:, self.ctrl_ids, 2] / self.nominal_kd,
                     model.actuator_forcerange[:, self.ctrl_ids, 1] / self.nominal_force,
                     self.env.motor_delay_lag[:, None].float(), motor.executed_raw]
            xi = torch.cat([xi, *extra], dim=-1)
        assert xi.shape == (n, self.dim)
        return xi.clone()
