"""Regenerate local rules, motor histories, physics, contacts and referee."""

from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import zip_longest
import gzip
import json
import mujoco
import numpy as np
from .world import digest, ASSETS, ROBOT, POLICIES
from .episode import run, source_hashes
from .config import Config, Physics


def verify(directory):
    directory = Path(directory)
    expected = json.loads((directory / "manifest.json").read_text())
    if expected["schema"] != "duckfu.run.v1" or not expected["capture"]:
        raise ValueError("Replay needs an intact full capture")
    if (
        expected["source_sha256"] != source_hashes()
        or expected["mujoco_version"] != mujoco.__version__
    ):
        raise ValueError("Replay source or MuJoCo version differs")
    if expected["robot_sha256"] != digest(ASSETS / ROBOT) or expected[
        "policies_sha256"
    ] != {n: digest(ASSETS / "microduck/policies" / p) for n, p in POLICIES.items()}:
        raise ValueError("Replay model or policy differs")
    for name, sha in expected["hashes"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Evidence digest differs: {name}")
    with TemporaryDirectory(prefix="duckfu-replay-") as temp:
        dest = Path(temp) / "regenerated"
        actual = run(
            dest,
            Config(**expected["config"]),
            Physics(**expected["physics"]),
            expected["seed"],
            True,
            expected["probe"],
        )
        for key in expected:
            if key not in ("performance", "hashes") and expected[key] != actual[key]:
                raise ValueError(f"Regenerated manifest differs: {key}")
        largest = 0.0
        for filename in ("states.npy",):
            x = np.load(directory / filename, mmap_mode="r")
            y = np.load(dest / filename, mmap_mode="r")
            if x.shape != y.shape or not np.array_equal(x, y):
                raise ValueError(f"Physical replay differs: {filename}")
            largest = max(largest, float(np.max(np.abs(x - y))))
        for filename in ("controls.npz",):
            with np.load(directory / filename) as x, np.load(dest / filename) as y:
                for key in x.files:
                    if not np.array_equal(x[key], y[key]):
                        raise ValueError(f"Motor/state history differs: {key}")
        for filename in ("contacts_full.jsonl.gz", "tactical_decisions.jsonl.gz"):
            with (
                gzip.open(directory / filename, "rt") as x,
                gzip.open(dest / filename, "rt") as y,
            ):
                if any(a != b for a, b in zip_longest(x, y)):
                    raise ValueError(f"Regenerated semantic log differs: {filename}")
        for filename in (
            "contacts_summary.json",
            "referee.json",
            "runtime_events.jsonl",
            "metrics.json",
            "model_inventory.json",
        ):
            if (directory / filename).read_bytes() != (dest / filename).read_bytes():
                raise ValueError(f"Regenerated result differs: {filename}")
    return dict(
        verified=True,
        closed_loop_rules_regenerated=True,
        motor_histories_regenerated=True,
        contact_log_identical=True,
        referee_identical=True,
        max_integration_state_error=largest,
        evidence_domain="simulation",
        steps=expected["steps"],
    )
