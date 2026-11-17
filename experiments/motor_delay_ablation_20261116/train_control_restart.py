"""Continue from the delay run's checkpoint with identical settings except delay.

Stopping at iteration 167050 must not shorten the original LR decay horizon.
"""
import dataclasses,json,os,sys
from pathlib import Path

os.environ.setdefault('SUPERMIMIC_ROBOT','k1')
from supermimic.config import TrainConfig
import supermimic.training.loop as loop

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'logs/k1_taeguek_27_imu_motor_delay/config.json'
NAME='k1_taeguek_27_imu_no_motor_delay_control_167050_restart'
TARGET=167050

def restore(obj, raw):
    names={f.name for f in dataclasses.fields(obj)}
    if names != set(raw): raise ValueError(f'Config mismatch: missing={names-set(raw)}, extra={set(raw)-names}')
    for key,value in raw.items():
        old=getattr(obj,key)
        if dataclasses.is_dataclass(old): restore(old,value)
        elif isinstance(old,Path) or key in {'resume','library','manifest'} and value is not None: setattr(obj,key,Path(value))
        elif isinstance(old,tuple): setattr(obj,key,tuple(value))
        else: setattr(obj,key,value)
    return obj

def jsonlike(obj): return json.loads(json.dumps(dataclasses.asdict(obj),default=str))

def build():
    original=json.loads(REFERENCE.read_text())
    cfg=restore(TrainConfig(),original)
    if jsonlike(cfg)!=original: raise ValueError('Failed to reconstruct the reference configuration exactly')
    if not cfg.env.motor_delay_dr or cfg.env.k1_legacy_friction_delay_dr: raise ValueError('Reference delay flags not as expected')
    if str(cfg.resume) != 'logs/k1_taeguek_27_imu_roll3_pitch5/ckpt_0128000.pt': raise ValueError('Unexpected reference checkpoint')
    if cfg.schedule.total_iterations != 320000: raise ValueError('Unexpected learning-rate schedule')
    cfg.run_name=NAME
    cfg.env.motor_delay_dr=False
    cfg.schedule.total_iterations=TARGET
    cfg.schedule.max_hours=None
    return cfg

def main():
    cfg=build();print(json.dumps(jsonlike(cfg),indent=2),flush=True)
    if '--validate' in sys.argv: return
    output=ROOT/cfg.log_root/cfg.run_name
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    orig=loop.cosine_lr
    def reference_lr(step,total_steps,base_lr,final_ratio,decay_fraction):
        if total_steps != TARGET: raise ValueError('Unexpected stop target')
        return orig(step,320000,base_lr,final_ratio,decay_fraction)
    loop.cosine_lr=reference_lr
    print('[INFO] Stop at 167050; learning-rate schedule remains based on 320000',flush=True)
    loop.Trainer(cfg).run()

if __name__=='__main__':main()
