"""Robot selection is fixed for each Python process; G1 remains the default."""
import os

ROBOT = os.environ.get("SUPERMIMIC_ROBOT", "g1").lower()
if ROBOT not in ("g1", "k1"):
    raise ValueError(f"Unknown SUPERMIMIC_ROBOT={ROBOT!r}; choose g1 or k1")
NJ = 23 if ROBOT == "k1" else 29
NQ = 7 + NJ
NV = 6 + NJ
STATE_DIM = NQ + NV
GOAL_FEAT_DIM = 2 * NJ + 10
XI_DIM = NJ + 4
