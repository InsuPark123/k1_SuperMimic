"""Prepare a motor-delay K1 run with mesh feet and no self-collision penalty.

Run from the repository root with:
  SUPERMIMIC_ROBOT=k1 python -m experiments.train_motor_delay_foot_mesh --validate
Omit --validate to start training.
"""
import dataclasses
import json
import os
from pathlib import Path
import sys

os.environ['SUPERMIMIC_ROBOT'] = 'k1'

import torch
from supermimic.config import TrainConfig
from experiments.train_motor_delay_self_collision import restore

CHECKPOINT = Path('logs/k1_taeguek_27_imu_motor_delay/ckpt_latest.pt')


def main():
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if ck['iteration'] != 167050 or not ck['config']['env']['motor_delay_dr']:
        raise ValueError('Expected the motor-delay 167050 checkpoint')
    cfg = restore(TrainConfig(), ck['config'])
    cfg.resume = CHECKPOINT
    cfg.run_name = 'k1_taeguek_motor_delay_foot_mesh_no_self_collision'
    cfg.env.self_collision_weight = 0.0
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
