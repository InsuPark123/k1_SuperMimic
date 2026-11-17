"""Resume the final motor-delay run with an MJLab self-contact penalty."""
import dataclasses
import json
import os
from pathlib import Path

os.environ['SUPERMIMIC_ROBOT'] = 'k1'

import torch
from supermimic.config import TrainConfig

CHECKPOINT = Path('logs/k1_taeguek_27_imu_motor_delay/ckpt_latest.pt')


def restore(obj, values):
    fields = {f.name for f in dataclasses.fields(obj)}
    if set(values) - fields:
        raise ValueError(f'Unknown checkpoint fields: {set(values) - fields}')
    for name, value in values.items():
        current = getattr(obj, name)
        if dataclasses.is_dataclass(current):
            restore(current, value)
        elif isinstance(current, Path) or (name in {'resume', 'library', 'manifest'} and value is not None):
            setattr(obj, name, Path(value))
        elif isinstance(current, tuple):
            setattr(obj, name, tuple(value))
        else:
            setattr(obj, name, value)
    return obj


def main():
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if ck['iteration'] != 167050 or not ck['config']['env']['motor_delay_dr']:
        raise ValueError('Unexpected checkpoint')
    cfg = restore(TrainConfig(), ck['config'])
    cfg.resume = CHECKPOINT
    cfg.run_name = 'k1_taeguek_motor_delay_self_collision_167050'
    cfg.env.self_collision_weight = -1.0
    output = Path(cfg.log_root) / cfg.run_name
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str), flush=True)
    if '--validate' in __import__('sys').argv:
        return
    from supermimic.training.loop import Trainer
    Trainer(cfg).run()


if __name__ == '__main__':
    main()
