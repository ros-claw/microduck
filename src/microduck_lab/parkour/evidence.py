"""Evidence gates shared by audit and film publication."""

import hashlib
import math
from pathlib import Path


def verify_capture(source, manifest):
    source = Path(source)
    required = {"scene.mjb", "inputs.npz", "trajectory.npz"}
    if not required.issubset(manifest):
        raise ValueError("Incomplete capture manifest")
    for name, expected in manifest.items():
        if Path(name).name != name:
            raise ValueError("Capture names must be local filenames")
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Capture checksum mismatch: " + name)


def contact_gate(categories, warnings):
    """Conservative publication gate: no ignored native-floor/self contacts."""
    failures = []
    for name in ("duck_floor", "duck_hazard"):
        if name not in categories or categories[name]["samples"] == 0:
            failures.append(name + ": missing force-bearing contact evidence")
    for name, row in categories.items():
        if not all(
            math.isfinite(row[k]) and row[k] >= 0
            for k in ("max_penetration_m", "p99_penetration_m")
        ):
            failures.append(name + ": invalid penetration metrics")
            continue
        if row["max_penetration_m"] >= 0.0015:
            failures.append(name + ": max penetration >= 1.5 mm")
        if row["p99_penetration_m"] >= 0.001:
            failures.append(name + ": P99 penetration >= 1 mm")
    if any(warnings):
        failures.append("MuJoCo solver warnings")
    return dict(
        passed=not failures,
        failures=failures,
        scope="Every force-bearing contact category, including native ground and self-contact",
    )
