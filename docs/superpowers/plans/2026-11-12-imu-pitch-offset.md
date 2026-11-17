# IMU Pitch Offset Implementation Plan

**Goal:** Train with an optional per-episode orientation-only pitch offset, defaulting to zero.

**Architecture:** Apply the body-frame quaternion offset to projected gravity and reference-relative orientation in simulator collection and imagined policy rollouts. Keep physical state, gyro, dynamics, and reward unchanged. Sample independently for each imagined trajectory and keep it fixed across history and rollout.

**Tech Stack:** PyTorch, mjlab, MuJoCo.

1. Add focused tests for quaternion sign, gravity, anchor rotation, zero offset, reset sampling, and rollout consistency; run red.
2. Implement opt-in configuration and observation transforms for collection and imagination; document the command.
3. Run focused tests and a short actual training smoke test if GPU availability permits; inspect the diff.
