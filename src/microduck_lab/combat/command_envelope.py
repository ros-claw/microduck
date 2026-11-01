"""Select measured walking inputs; never filter the 14 joint outputs."""

import math


def facing_command(distance, angle):
    if distance < 0.055:
        return (0.0, 0.0, 0.0)
    if abs(angle) > 0.25:
        side = 1.0 if angle > 0 else -1.0
        return (0.28 if distance > 0.36 else -0.15, side * 0.30, side)
    return (0.28, 0.0, max(-0.6, min(0.6, 2.8 * angle)))


def flank_ready(me, target, partner):
    import numpy as np

    a, b = np.array(me.xy) - target.xy, np.array(partner.xy) - target.xy
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    angle = math.acos(float(np.clip(np.dot(a, b) / max(1e-9, da * db), -1, 1)))
    return bool(da < 0.42 and db < 0.60 and 0.6 < angle < 2.4)
