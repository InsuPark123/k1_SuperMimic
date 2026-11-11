import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import K1_JOINT_NAMES, convert_csv_to_npz, load_k1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

out = Path(sys.argv[1])
raw = np.zeros((40, 30)); raw[:, 2] = .78; raw[:, 6] = 1
np.savetxt(out / 'motion.csv', raw, delimiter=',')
convert_csv_to_npz(out / 'motion.csv', out / 'motion.npz', 50)
a=SuperMimicEnvCfgArgs(npz_files=[str(out / 'motion.npz')], num_envs=2,obs_noise=False,reset_noise=False,domain_randomization=True,replay_capacity=50)
env=make_env(a,device='cpu')
assert env.sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
robot=env.scene['robot']
assert tuple(robot.joint_names)==K1_JOINT_NAMES, (robot.joint_names,K1_JOINT_NAMES)
obs,_=env.reset()
for i in range(20):
 obs,r,term,tout,extra=env.step(torch.zeros(2,23))
cmd=env.command_manager.get_term('motion')
buffer=env.recorder_manager.get_term('replay').buffer
print('BUFFER',buffer.size)
fk=G1Kinematics(load_k1_model(),device='cpu')
heads=AnalyticHeads(fk,TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
states = env.recorder_manager.get_term('replay').state_io.read()
predicted_bodies = heads.robot_bodies(states)
body_ids = torch.tensor(robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0])
actual_positions = robot.data.body_link_pos_w[:, body_ids] - env.scene.env_origins[:, None, :]
assert torch.allclose(predicted_bodies['pos'], actual_positions, atol=2e-5)
assert torch.allclose(predicted_bodies['lin_vel'], robot.data.body_link_lin_vel_w[:, body_ids], atol=2e-5)
wm=WorldModel(WorldModelCfg(hidden=(16,16)),heads,'cpu')
wmt=WorldModelTrainer(wm,cmd.library,WMTrainerCfg(batch_size=2,batches_per_update=1,window=2,compile=None))
metrics = wmt.update(buffer)
assert np.isfinite(metrics['loss'])
print('WORLD_MODEL_UPDATE', metrics['loss'])
pc=PolicyCfg(encoder_hidden=(16,),decoder_hidden=(16,))
policy=Policy(pc)
critic=Critic(CriticCfg(hidden=(16,)),pcfg=pc)
pt=PolicyTrainer(wm,policy,critic,cmd.library,PolicyTrainerCfg(batch_size=2,horizon=2,critic_epochs=1,critic_minibatches=1,compile=None,obs_noise=False))
metrics = pt.update(buffer)
assert np.isfinite(metrics['critic_loss']) and np.isfinite(metrics['policy_grad_norm'])
print('POLICY_UPDATE', metrics['critic_loss'])
env.close()

# Exercise the real coordinator as well. CUDA synchronization is intentionally
# disabled only in this CPU-only test; GPU execution remains to be verified.
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

cfg = TrainConfig()
cfg.device = 'cpu'
cfg.data.npz_dir = out
cfg.log_root = out / 'logs'
cfg.env.num_envs = 2
cfg.env.replay_capacity = 50
cfg.schedule.total_iterations = 13
cfg.schedule.warmup_iterations = 9
cfg.schedule.updates_per_env_step = 1.0
cfg.schedule.log_interval = 100
cfg.schedule.checkpoint_interval = 100
cfg.world_model.hidden = (16, 16)
cfg.wm_trainer.window = 2
cfg.wm_trainer.batch_size = 2
cfg.wm_trainer.batches_per_update = 1
cfg.wm_trainer.compile = None
cfg.policy.encoder_hidden = (16,)
cfg.policy.decoder_hidden = (16,)
cfg.critic.hidden = (16,)
cfg.policy_trainer.batch_size = 2
cfg.policy_trainer.horizon = 2
cfg.policy_trainer.critic_epochs = 1
cfg.policy_trainer.critic_minibatches = 1
cfg.policy_trainer.compile = None
torch.cuda.synchronize = lambda: None
trainer = Trainer(cfg)
trainer.run()
checkpoint = torch.load(out / 'logs/debug/ckpt_latest.pt', map_location='cpu', weights_only=False)
assert checkpoint['robot'] == 'k1' and checkpoint['num_updates'] > 0
print('FULL_TRAINER_LOOP', checkpoint['num_updates'])
