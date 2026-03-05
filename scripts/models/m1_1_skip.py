"""Model M1_1_SKIP — same as M1_1 but trained every 4 steps."""

from .m1_1 import *  # noqa: F401,F403

MODEL_PATH_DEFAULT = "rl_m1_1_skip.pt"
LABEL = "M1_1_SKIP"
