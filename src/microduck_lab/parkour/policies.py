"""Optional bilateral walking wrapper, using upstream's 61-D symmetry contract.

Permutation/sign convention adapted from pollen-robotics/microduck_rl
`tasks/symmetry.py` (Apache-2.0). Physics is never mirrored or overwritten;
only policy observations and resulting joint action offsets are transformed.
Each direction must still pass its own physical validation battery.
"""

import numpy as np
from ..sim.runtime import PolicyBank

JOINT_PERM = np.array([9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4])
JOINT_SIGN = np.array(
    [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32
)
OBS_PERM = np.concatenate(
    [np.arange(6), 6 + JOINT_PERM, 20 + JOINT_PERM, 34 + JOINT_PERM, np.arange(48, 61)]
)
OBS_SIGN = np.concatenate(
    [
        [-1, 1, -1],
        [1, -1, 1],
        JOINT_SIGN,
        JOINT_SIGN,
        JOINT_SIGN,
        [1, -1, -1],
        [1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1],
    ]
).astype(np.float32)


class ParkourPolicyBank(PolicyBank):
    mirror_run = False

    def infer(self, name, obs):
        if name == "run" and self.mirror_run:
            mirrored = super().infer(name, obs[OBS_PERM] * OBS_SIGN)
            return mirrored[JOINT_PERM] * JOINT_SIGN
        return super().infer(name, obs)
