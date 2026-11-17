from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import math

from mjlab.envs import mdp as base_mdp
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul, quat_inv, quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def _rotate_pitch_inverse(v: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(pitch), torch.sin(pitch)
    x, y, z = v.unbind(-1)
    return torch.stack((c * x - s * z, y, s * x + c * z), dim=-1)


def pitch_biased_gravity(gravity: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    return _rotate_pitch_inverse(gravity, pitch)


def pitch_biased_torso_quat(root_quat: torch.Tensor, torso_quat: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Apply pelvis-frame IMU bias before the pelvis-to-torso rotation."""
    bias = quat_from_rotvec(torch.stack((torch.zeros_like(pitch), pitch, torch.zeros_like(pitch)), dim=-1))
    return quat_mul(quat_mul(quat_mul(root_quat, bias), quat_inv(root_quat)), torso_quat)


def tilt_bias_quat(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    return quat_from_rotvec(torch.stack((roll, pitch, torch.zeros_like(roll)), dim=-1))


def tilt_biased_gravity(gravity: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(tilt_bias_quat(roll, pitch), gravity)


def tilt_biased_torso_quat(torso_quat: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    return quat_mul(torso_quat, tilt_bias_quat(roll, pitch))


def _imu_roll_offset(env):
    if not hasattr(env, 'imu_roll_offset'):
        env.imu_roll_offset = torch.zeros(env.num_envs, device=env.device)
    return env.imu_roll_offset


def reset_imu_tilt(env, env_ids, pitch_half_range_deg: float, roll_half_range_deg: float) -> None:
    reset_imu_pitch(env, env_ids, pitch_half_range_deg)
    _imu_roll_offset(env)
    if env_ids is None:
        env_ids = slice(None)
    values = env.imu_roll_offset[env_ids]
    env.imu_roll_offset[env_ids] = (torch.rand_like(values) * 2 - 1) * math.radians(roll_half_range_deg)


def _imu_pitch_offset(env):
    if not hasattr(env, 'imu_pitch_offset'):
        env.imu_pitch_offset = torch.zeros(env.num_envs, device=env.device)
    return env.imu_pitch_offset


def reset_imu_pitch(env, env_ids, half_range_deg: float) -> None:
    _imu_pitch_offset(env)
    if env_ids is None:
        env_ids = slice(None)
    values = env.imu_pitch_offset[env_ids]
    env.imu_pitch_offset[env_ids] = (torch.rand_like(values) * 2 - 1) * math.radians(half_range_deg)


def imu_pitch_gravity(env):
    return tilt_biased_gravity(base_mdp.projected_gravity(env), _imu_roll_offset(env), _imu_pitch_offset(env))


def imu_pitch_anchor(env, command_name: str):
    from supermimic.world_model.heads import anchor_relative_obs
    command = env.command_manager.get_term(command_name)
    torso = tilt_biased_torso_quat(command.robot_anchor_quat_w, _imu_roll_offset(env), _imu_pitch_offset(env))
    return anchor_relative_obs(command.robot_anchor_pos_w, torso, command.anchor_pos_w, command.anchor_quat_w)


def imu_pitch_anchor_pos(env, command_name: str):
    return imu_pitch_anchor(env, command_name)[..., :3]


def imu_pitch_anchor_ori(env, command_name: str):
    return imu_pitch_anchor(env, command_name)[..., 3:]
