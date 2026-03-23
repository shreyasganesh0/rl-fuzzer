"""Model M1_2_SKIP — same as M1_2 but trained every 4 steps."""

from .m1_2 import *  # noqa: F401,F403

MODEL_PATH_DEFAULT = "rl_m1_2_skip.pt"
LABEL = "M1_2_SKIP"
