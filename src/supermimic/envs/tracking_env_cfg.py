"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push; friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.robot import ROBOT
from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg, reset_motor_delay, randomize_motor_output
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp import rewards as sm_rewards
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg

K1_LEG_FRICTION = {
    'left_hip_pitch_joint': (0.774, 2.322), 'right_hip_pitch_joint': (0.0, 0.320),
    'left_hip_roll_joint': (0.0, 0.712), 'right_hip_roll_joint': (0.291, 0.872),
    'left_hip_yaw_joint': (0.0, 0.406), 'right_hip_yaw_joint': (0.0, 0.399),
    'left_knee_joint': (0.285, 0.855), 'right_knee_joint': (0.214, 0.642),
    'left_ankle_pitch_joint': (0.0, 1.848), 'right_ankle_pitch_joint': (0.0, 1.884),
    'left_ankle_roll_joint': (0.0, 0.320), 'right_ankle_roll_joint': (0.0, 0.167),
}

class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg that disables MuJoCo Warp's per-step overflow printf (a kernel build-time constant,
    so it must be set before the CUDA graphs are captured)."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
if ROBOT == "k1":
    from supermimic.envs.k1_config import get_k1_rev1_cfg, K1_REV1_ACTION_SCALE, K1_REV1_FOOT_COLLISION_GEOM_PATTERN
    TRACKED_BODIES = tuple(n.replace("wrist_yaw_link", "wrist_roll_rubber_hand") for n in TRACKED_BODIES)

ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_GEOM_REGEX = K1_REV1_FOOT_COLLISION_GEOM_PATTERN if ROBOT == "k1" else r"^(left|right)_foot[1-7]_collision$"


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    encoder_bias_range: float = 0.07 if ROBOT == "k1" else 0.01
    encoder_bias_other_range: float = 0.01
    k1_system_id_dr: bool = False
    k1_legacy_friction_delay_dr: bool = False
    motor_delay_dr: bool = False
    self_collision_weight: float = 0.0
    extended_dr_context: bool = False
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    anchor_ori_obs: bool = False  # 6D relative orientation, no relative position.
    imu_pitch_offset_deg: float = 0.0
    imu_roll_offset_deg: float = 0.0
    base_xy_weight: float = 0.0
    base_yaw_weight: float = 0.0
    base_xy_std: float = 0.3
    base_yaw_std: float = 0.4
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    if a.self_collision_weight > 0 or (a.self_collision_weight and ROBOT != 'k1'):
        raise ValueError('Self-collision weight must be nonpositive and requires K1')
    if a.k1_system_id_dr and (ROBOT != 'k1' or not a.domain_randomization):
        raise ValueError('K1 system-ID randomization requires K1 and domain randomization')
    if a.extended_dr_context and not a.k1_system_id_dr:
        raise ValueError('Extended DR context requires K1 system-ID randomization')
    if a.k1_legacy_friction_delay_dr and not a.k1_system_id_dr:
        raise ValueError('Legacy friction and delay randomization requires K1 system-ID randomization')
    if a.motor_delay_dr and not a.k1_system_id_dr:
        raise ValueError('Motor delay randomization requires K1 system-ID randomization')
    if a.extended_dr_context and not a.k1_legacy_friction_delay_dr:
        raise ValueError('Extended DR context requires legacy friction and delay randomization')
    if not 0 < a.encoder_bias_range < float('inf') or not 0 < a.encoder_bias_other_range < float('inf'):
        raise ValueError('encoder bias half-range must be positive and finite')
    if not 0 <= a.imu_pitch_offset_deg <= 180 or not 0 <= a.imu_roll_offset_deg <= 180:
        raise ValueError('IMU roll and pitch offset half-ranges must be between 0 and 180 degrees')
    if a.anchor_obs and a.anchor_ori_obs:
        raise ValueError('anchor_obs and anchor_ori_obs are mutually exclusive')
    if a.base_xy_weight < 0 or a.base_yaw_weight < 0 or a.base_xy_std <= 0 or a.base_yaw_std <= 0:
        raise ValueError('Base reward weights must be nonnegative and standard deviations positive')
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=sm_obs.imu_pitch_gravity if a.imu_pitch_offset_deg or a.imu_roll_offset_deg else base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **({"anchor_pos": ObservationTermCfg(func=sm_obs.imu_pitch_anchor_pos if a.imu_pitch_offset_deg or a.imu_roll_offset_deg else track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25))} if a.anchor_obs else {}),
                **({"anchor_ori": ObservationTermCfg(func=sm_obs.imu_pitch_anchor_ori if a.imu_pitch_offset_deg or a.imu_roll_offset_deg else track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05))} if a.anchor_obs or a.anchor_ori_obs else {}),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=K1_REV1_ACTION_SCALE if ROBOT == "k1" else G1_ACTION_SCALE,
            use_default_offset=True,
            noise_std=a.action_noise_std,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            debug_vis=a.debug_vis,
            **{'g_max': (5.0 + a.base_xy_weight + a.base_yaw_weight) * 2.0, **a.extra_command_kwargs},
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.imu_pitch_offset_deg or a.imu_roll_offset_deg:
        events["imu_pitch_offset"] = EventTermCfg(mode="reset", func=sm_obs.reset_imu_tilt, params={"pitch_half_range_deg": a.imu_pitch_offset_deg, "roll_half_range_deg": a.imu_roll_offset_deg})
    if a.domain_randomization:
        if a.k1_system_id_dr:
            if a.k1_legacy_friction_delay_dr or a.motor_delay_dr:
                events['motor_delay'] = EventTermCfg(mode='reset', func=reset_motor_delay)
            if a.k1_legacy_friction_delay_dr:
                for joint, bounds in K1_LEG_FRICTION.items():
                    events[f'leg_friction_{joint}'] = EventTermCfg(mode='reset', func=dr.joint_friction, params={'asset_cfg': SceneEntityCfg('robot', joint_names=(joint,)), 'ranges': bounds, 'operation': 'abs'})
            events['pd_gains'] = EventTermCfg(mode='reset', func=dr.pd_gains, params={'asset_cfg': SceneEntityCfg('robot'), 'kp_range': (0.9, 1.1), 'kd_range': (0.9, 1.1), 'operation': 'scale'})
            events['motor_output'] = EventTermCfg(mode='reset', func=randomize_motor_output, params={'asset_cfg': SceneEntityCfg('robot'), 'effort_limit_range': (0.9, 1.0)})
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        if ROBOT == 'k1':
            legs = r'^(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint$'
            others = r'^(?!((left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint)$).*'
            events["encoder_bias_legs"] = EventTermCfg(mode="reset", func=dr.encoder_bias, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(legs,)), "bias_range": (-a.encoder_bias_range, a.encoder_bias_range)})
            events["encoder_bias_other"] = EventTermCfg(mode="reset", func=dr.encoder_bias, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(others,)), "bias_range": (-a.encoder_bias_other_range, a.encoder_bias_other_range)})
        else:
            events["encoder_bias"] = EventTermCfg(mode="reset", func=dr.encoder_bias, params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-a.encoder_bias_range, a.encoder_bias_range)})
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    }
    if a.base_xy_weight:
        rewards['base_xy'] = RewardTermCfg(func=sm_rewards.base_xy_reward, weight=a.base_xy_weight, params={'command_name': 'motion', 'std': a.base_xy_std})
    if a.base_yaw_weight:
        rewards['base_yaw'] = RewardTermCfg(func=sm_rewards.base_yaw_reward, weight=a.base_yaw_weight, params={'command_name': 'motion', 'std': a.base_yaw_std})
    sensors = ()
    if a.self_collision_weight:
        sensors = (ContactSensorCfg(name='self_collision',
            primary=ContactMatch(mode='subtree', pattern='pelvis', entity='robot'),
            secondary=ContactMatch(mode='subtree', pattern='pelvis', entity='robot'),
            fields=('found', 'force'), reduce='none', num_slots=1, history_length=4),)
        rewards['self_collision'] = RewardTermCfg(func=self_collision_cost,
            weight=a.self_collision_weight, params={'sensor_name': 'self_collision', 'force_threshold': 10.0})

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": "torso_link", 'extended_dr': a.extended_dr_context})

    robot_cfg = get_k1_rev1_cfg() if ROBOT == 'k1' else get_g1_robot_cfg()
    if a.k1_legacy_friction_delay_dr or a.motor_delay_dr:
        for actuator in robot_cfg.articulation.actuators:
            actuator.delay_min_lag = 0
            actuator.delay_max_lag = 4
            actuator.delay_hold_prob = 1.0
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": robot_cfg}, sensors=sensors),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link", distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
