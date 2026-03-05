"""Model M1_0_SKIP — same as M1_0 but trained every 4 steps."""

from .m1_0 import *  # noqa: F401,F403

MODEL_PATH_DEFAULT = "rl_m1_0_skip.pt"
LABEL = "M1_0_SKIP"
