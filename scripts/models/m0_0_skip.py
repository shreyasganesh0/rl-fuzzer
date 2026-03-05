"""Model M0_0_SKIP — same as M0_0 but trained every 4 steps."""

from .m0_0 import *  # noqa: F401,F403

MODEL_PATH_DEFAULT = "rl_m0_0_skip.pt"
LABEL = "M0_0_SKIP"
