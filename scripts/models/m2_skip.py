"""Model M2_SKIP — same as M2 but trained every 4 steps."""

from .m2 import *  # noqa: F401,F403

MODEL_PATH_DEFAULT = "rl_m2_skip.pt"
LABEL = "M2_SKIP"
