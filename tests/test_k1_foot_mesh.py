import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.entity import Entity
from supermimic.robot import ROBOT
from supermimic.envs.k1_config import (
    K1_REV1_FOOT_COLLISION_GEOM_NAMES,
    K1_REV1_FOOT_COLLISION_GEOM_PATTERN,
    get_k1_rev1_cfg,
)
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg


@pytest.mark.skipif(ROBOT != 'k1', reason='K1 foot geometry')
def test_k1_mesh_feet_and_default_disabled_self_collision():
    env = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=['unused']))
    assert 'self_collision' not in env.rewards
    assert all(sensor.name != 'self_collision' for sensor in env.scene.sensors)
    model = Entity(get_k1_rev1_cfg()).spec.compile()
    assert len(K1_REV1_FOOT_COLLISION_GEOM_NAMES) == 2
    for name in K1_REV1_FOOT_COLLISION_GEOM_NAMES:
        assert re.fullmatch(K1_REV1_FOOT_COLLISION_GEOM_PATTERN, name)
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert i >= 0 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_contype[i] == 1 and model.geom_conaffinity[i] == 1
        assert model.geom_condim[i] == 3 and model.geom_priority[i] == 1
        np.testing.assert_allclose(model.geom_friction[i], [0.6, 0.005, 0.0001])
    for side in ('left', 'right'):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{side}_ankle_roll_link_collision_0') == -1
        knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{side}_knee_link_collision_0')
        assert model.geom_contype[knee] == 1 and model.geom_conaffinity[knee] == 1
