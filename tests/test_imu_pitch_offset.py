import math
from types import SimpleNamespace
import torch
from supermimic.envs.mdp.observations import pitch_biased_gravity, pitch_biased_torso_quat, reset_imu_pitch
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul, projected_gravity
from supermimic.world_model.heads import anchor_relative_obs


def test_roll_pitch_bias_matches_review_local_quaternions():
    from supermimic.envs.mdp.observations import tilt_biased_gravity, tilt_biased_torso_quat
    root = quat_from_rotvec(torch.tensor([[0.2, -0.1, 0.3]]))
    torso = quat_mul(root, quat_from_rotvec(torch.tensor([[0., 0., 0.8]])))
    roll, pitch = torch.tensor([math.radians(3)]), torch.tensor([math.radians(-5)])
    bias = quat_from_rotvec(torch.stack((roll, pitch, torch.zeros_like(roll)), dim=-1))
    torch.testing.assert_close(tilt_biased_gravity(projected_gravity(root), roll, pitch), projected_gravity(quat_mul(root, bias)), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tilt_biased_torso_quat(torso, roll, pitch), quat_mul(torso, bias), atol=1e-6, rtol=1e-6)


def test_roll_pitch_offsets_reset_independently_and_stay_fixed():
    from supermimic.envs.mdp.observations import reset_imu_tilt
    env = SimpleNamespace(num_envs=20, device='cpu')
    reset_imu_tilt(env, None, pitch_half_range_deg=5, roll_half_range_deg=3)
    roll, pitch = env.imu_roll_offset.clone(), env.imu_pitch_offset.clone()
    assert torch.all(roll.abs() <= math.radians(3))
    assert torch.all(pitch.abs() <= math.radians(5))
    reset_imu_tilt(env, torch.tensor([0]), pitch_half_range_deg=0, roll_half_range_deg=0)
    assert env.imu_roll_offset[0] == env.imu_pitch_offset[0] == 0
    torch.testing.assert_close(env.imu_roll_offset[1:], roll[1:])
    torch.testing.assert_close(env.imu_pitch_offset[1:], pitch[1:])


def test_environment_enables_roll_and_pitch_observations():
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.observations import reset_imu_tilt, imu_pitch_gravity, imu_pitch_anchor_ori
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused'], anchor_ori_obs=True, imu_pitch_offset_deg=5, imu_roll_offset_deg=3))
    event = cfg.events['imu_pitch_offset']
    assert event.func is reset_imu_tilt and event.mode == 'reset'
    assert event.params == {'pitch_half_range_deg': 5, 'roll_half_range_deg': 3}
    assert cfg.observations['policy'].terms['projected_gravity'].func is imu_pitch_gravity
    assert cfg.observations['goal'].terms['anchor_ori'].func is imu_pitch_anchor_ori


def test_orientation_bias_matches_body_frame_quaternion_experiment():
    q = quat_from_rotvec(torch.tensor([[0.2, -0.1, 0.3], [0., 0., 0.]]))
    pitch = torch.tensor([math.radians(6), math.radians(-3)])
    v = torch.stack((pitch * 0, pitch, pitch * 0), -1)
    biased = quat_mul(q, quat_from_rotvec(v))
    p = torch.tensor([[1., 2., 3.], [0., 0., 0.]])
    ref_q = quat_from_rotvec(torch.tensor([[0., 0., 0.4], [0., 0., 0.]]))
    ref_p = torch.tensor([[2., 4., 6.], [1., 0., 0.]])
    gravity = projected_gravity(q)
    torch.testing.assert_close(pitch_biased_gravity(gravity, pitch), projected_gravity(biased))
    biased_torso = pitch_biased_torso_quat(q, q, pitch)
    torch.testing.assert_close(anchor_relative_obs(p, biased_torso, ref_p, ref_q), anchor_relative_obs(p, biased, ref_p, ref_q), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pitch_biased_torso_quat(q, q, torch.zeros_like(pitch)), q, atol=1e-6, rtol=1e-6)


def test_torso_offset_follows_pelvis_before_waist_yaw():
    root = quat_from_rotvec(torch.tensor([[0.1, -0.2, 0.3]]))
    waist = quat_from_rotvec(torch.tensor([[0., 0., 0.8]]))
    bias = quat_from_rotvec(torch.tensor([[0., 0.1, 0.]]))
    torso = quat_mul(root, waist)
    expected = quat_mul(quat_mul(root, bias), waist)
    torch.testing.assert_close(pitch_biased_torso_quat(root, torso, torch.tensor([0.1])), expected, atol=1e-6, rtol=1e-6)


def test_offset_resampled_only_for_reset_environments():
    env = SimpleNamespace(num_envs=3, device='cpu')
    reset_imu_pitch(env, torch.tensor([0, 1, 2]), half_range_deg=6.)
    first = env.imu_pitch_offset.clone()
    assert torch.all(first.abs() <= math.radians(6))
    reset_imu_pitch(env, torch.tensor([1]), half_range_deg=0.)
    assert env.imu_pitch_offset[1] == 0
    torch.testing.assert_close(env.imu_pitch_offset[[0, 2]], first[[0, 2]])
