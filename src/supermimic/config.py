"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg
from supermimic.robot import ROBOT


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    encoder_bias_range: float = 0.07 if ROBOT == "k1" else 0.01
    encoder_bias_other_range: float = 0.01
    k1_system_id_dr: bool = ROBOT == 'k1'
    k1_legacy_friction_delay_dr: bool = False
    motor_delay_dr: bool = False
    self_collision_weight: float = 0.0  # Negative enables K1 self-contact penalty.
    extended_dr_context: bool = False  # Include system-ID parameters and previous action in xi.
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False
    anchor_ori_obs: bool = False  # Reference-relative torso orientation without position.
    imu_pitch_offset_deg: float = 0.0  # Symmetric, fixed per episode; 0 disables.
    imu_roll_offset_deg: float = 0.0
    base_xy_weight: float = 0.0
    base_yaw_weight: float = 0.0
    base_xy_std: float = 0.3
    base_yaw_std: float = 0.4


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
