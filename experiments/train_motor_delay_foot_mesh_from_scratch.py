"""Train K1 with mesh feet and motor-delay randomization from scratch."""
import dataclasses
import json
import os
from pathlib import Path
import sys

os.environ['SUPERMIMIC_ROBOT'] = 'k1'

from supermimic.config import TrainConfig
from experiments.train_motor_delay_self_collision import restore

CONFIG = Path('logs/k1_taeguek_27_imu_motor_delay/config.json')
RUN_NAME = 'k1_taeguek_motor_delay_foot_mesh_scratch_retry_20261117'


def main():
    cfg = restore(TrainConfig(), json.loads(CONFIG.read_text()))
    cfg.run_name = RUN_NAME
    cfg.log_root = Path('experiments/foot_mesh_scratch_20261117/runs')
    cfg.resume = None
    cfg.env.self_collision_weight = 0.0
    cfg.schedule.max_hours = None
    if not cfg.env.motor_delay_dr:
        raise ValueError('Motor delay randomization must be enabled')
    output = Path(cfg.log_root) / cfg.run_name
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str), flush=True)
    if '--validate' in sys.argv:
        return
    from supermimic.training.loop import Trainer
    Trainer(cfg).run()


if __name__ == '__main__':
    main()
