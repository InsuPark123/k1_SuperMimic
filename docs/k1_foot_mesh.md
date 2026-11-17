# K1 foot mesh collision

The training entity now replaces the nine collision spheres on each K1 foot
with one mesh geom (`left_ankle_roll_link_collision_mesh` and
`right_ankle_roll_link_collision_mesh`). It uses the same ankle-roll STL as the
visual mesh; MuJoCo uses its convex collision representation. Self-contact
physics remains enabled. The default self-collision reward weight is zero.
The separately named `experiments.train_motor_delay_self_collision` still
explicitly enables the old penalty and is retained for reproducibility.

For a new motor-delay run without the penalty, validate the explicitly prepared
configuration first:

```bash
SUPERMIMIC_ROBOT=k1 .venv/bin/python -m experiments.train_motor_delay_foot_mesh --validate
```

Remove `--validate` to start the new run. It resumes the motor-delay 167050
checkpoint, keeps its environment count, schedule and max-hours settings, and
writes a separate run. Changing contacts invalidates prior claim of identical
physics; old checkpoints and old evaluation results stay as they were.

The matching Robotis MuJoCo model is a separate file so historical tests still
use their original sphere model:

```bash
cd /home/hc/robotis_mujoco
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robotis_mujoco supermimic.launch.py \
  model_xml_path:=/home/hc/robotis_mujoco/assets/ai_sapiens_description/mujoco/k1_rev1/k1_mjlab_foot_mesh.xml \
  policy_path:=/home/hc/robotis_mujoco/policy/Mimic/supermimic_taeguek_imu_motor_delay_167050
```

The previous standalone robustness evaluators compute reference foot clearance
from sphere geoms; do not use their contact/footwork metrics on the mesh model
until the evaluators are adapted and revalidated. A short smoke test does not
establish motion quality or real-world safety.

## Deployment contact compatibility

Keep each `*_knee_link_collision_0` as the training cylinder
(`pos="0 0 -0.1"`, `size="0.04 0.09"`) when using these mesh feet.
The deployment model's older full shank meshes overlap the mesh feet by about
37 mm at the Taeguek 1 initial pose and produce large self-contact forces.
The Robotis mesh-foot XML now uses the training shank cylinders; visual meshes,
self-collision masks and foot friction remain unchanged. Check geometry after
regenerating a deployment model with `robotis_mujoco/scripts/check_foot_mesh_contacts.py
--reference <motion.npz>`.
