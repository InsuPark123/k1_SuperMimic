"""K1 inputs must agree from motion conversion through imagined dynamics."""

import os
import subprocess
import sys
from pathlib import Path


def test_k1_components_in_fresh_process(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = '''
import torch
from supermimic.data.motion_convert import load_k1_model, convert_csv_to_npz
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
from supermimic.policy.nets import Policy, PolicyCfg
import numpy as np
import mujoco
from pathlib import Path
from supermimic.kinematics.g1_fk import G1Kinematics

model = load_k1_model()
assert (model.nq, model.nv, model.nu) == (30, 29, 23)
data = mujoco.MjData(model)
data.qpos[:] = model.qpos0
data.qpos[7:] = np.linspace(-.2, .2, 23)
data.qvel[:] = np.linspace(-.1, .1, 29)
mujoco.mj_forward(model, data)
fk = G1Kinematics(model, device='cpu', dtype=torch.float64)
fk_result = fk(torch.tensor(data.qpos.copy()), torch.tensor(data.qvel.copy()))
np.testing.assert_allclose(fk_result['pos'].numpy(), data.xpos[1:], atol=1e-9)
vel = np.zeros(6)
for i in range(1, model.nbody):
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, i, vel, 0)
    np.testing.assert_allclose(fk_result['ang_vel'][i-1].numpy(), vel[:3], atol=1e-9)
    np.testing.assert_allclose(fk_result['lin_vel'][i-1].numpy(), vel[3:], atol=1e-9)
raw = np.zeros((12, 30)); raw[:, 2] = .78; raw[:, 6] = 1
np.savetxt(Path(OUT) / 'motion.csv', raw, delimiter=',')
convert_csv_to_npz(Path(OUT) / 'motion.csv', Path(OUT) / 'motion.npz', 50, model=model)
with np.load(Path(OUT) / 'motion.npz') as d:
    assert d['joint_pos'].shape == (12, 23)
    names = tuple(d['body_names'].tolist())
lib = MotionLibrary([Path(OUT) / 'motion.npz'], names, torch.zeros(23), device='cpu')
assert lib.goal_window(torch.tensor([0]), 10).shape == (1, 560)
state = torch.zeros(2, 59); state[:, 3] = 1
spec = StateSpec(StateSpecCfg(), 'cpu')
assert spec.canonical(state).shape == (2, 59)
assert spec.integrate(state, torch.zeros(2, 58)).shape == (2, 59)
obs = analytic_observation(state, torch.zeros(2, 23), torch.zeros(2, 27), torch.zeros(23))
assert obs.shape == (2, 75)
hist = flatten_history_term_major(obs[None].expand(10, -1, -1))
policy = Policy(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,)))
assert policy(hist, torch.zeros(2, 560)).shape == (2, 23)
'''.replace('OUT', repr(str(tmp_path)))
    env = dict(os.environ, SUPERMIMIC_ROBOT='k1', PYTHONPATH=str(repo / 'src'), CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-c', script], cwd=repo, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_k1_simulator_world_model_and_policy_update(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, SUPERMIMIC_ROBOT="k1", PYTHONPATH=str(repo / "src"), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, str(repo / "tests/k1_pipeline_smoke.py"), str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WORLD_MODEL_UPDATE" in result.stdout and "POLICY_UPDATE" in result.stdout
    assert "FULL_TRAINER_LOOP" in result.stdout
