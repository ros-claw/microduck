"""Full closed-loop regeneration, including public decisions and physical contacts.

This proves deterministic simulation replay, not independent hardware validation.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import zip_longest
import gzip
import json
import numpy as np
import mujoco
from .episode import sha
from .multiplayer import GameConfig, run_game, ASSETS


def verify_game(directory):
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    if audit.get("schema") != "microduck.arena.multiplayer.v1" or not audit.get(
        "capture"
    ):
        raise ValueError("A full-rate multiplayer capture is required")
    required = {
        "scene.mjb",
        "trajectory.npz",
        "contacts.jsonl.gz",
        "decisions.jsonl.gz",
    }
    if set(audit.get("hashes", {})) != required:
        raise ValueError("Missing or unexpected evidence hash")
    if audit["mujoco_version"] != mujoco.__version__:
        raise ValueError("Replay requires the captured MuJoCo version")
    for name, digest in audit["hashes"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"Evidence digest mismatch: {name}")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in audit["source_sha256"].items():
        if sha(repo / name) != digest:
            raise ValueError(f"Replay source differs: {name}")
    for name, filename in (
        ("stand", "alpha_stand.onnx"),
        ("walk", "alpha_walking.onnx"),
    ):
        if (
            sha(ASSETS / "microduck/policies" / filename)
            != audit["policy_sha256"][name]
        ):
            raise ValueError(f"Policy differs: {name}")
    config = GameConfig(**audit["config"])
    if audit["steps"] < 1 or audit["duration"] != audit["steps"] * config.dt:
        raise ValueError("Invalid duration/step evidence")
    with TemporaryDirectory(prefix="duckverse-replay-") as temp:
        regenerated = Path(temp) / "run"
        actual = run_game(regenerated, audit["seed"], config, capture=True)
        ignore = {"hashes", "performance"}
        for key, value in audit.items():
            if key not in ignore and actual.get(key) != value:
                raise ValueError(f"Regenerated audit differs: {key}")
        if sha(regenerated / "scene.mjb") != audit["hashes"]["scene.mjb"]:
            raise ValueError("Regenerated model differs")
        max_error = 0.0
        with (
            np.load(directory / "trajectory.npz") as expected,
            np.load(regenerated / "trajectory.npz") as replay,
        ):
            if set(expected.files) != {"states", "ctrl"} or set(replay.files) != set(
                expected.files
            ):
                raise ValueError("Unexpected trajectory fields")
            for key in expected.files:
                x, y = expected[key], replay[key]
                if (
                    x.shape != y.shape
                    or not np.isfinite(x).all()
                    or not np.array_equal(x, y)
                ):
                    raise ValueError(f"Physical trajectory differs: {key}")
                max_error = max(max_error, float(np.max(np.abs(x - y))))
        for filename in ("contacts.jsonl.gz", "decisions.jsonl.gz"):
            with (
                gzip.open(directory / filename, "rt") as expected,
                gzip.open(regenerated / filename, "rt") as replay,
            ):
                for index, (x, y) in enumerate(zip_longest(expected, replay)):
                    if x is None or y is None or json.loads(x) != json.loads(y):
                        raise ValueError(
                            f"Regenerated semantic log differs: {filename}:{index}"
                        )
    return dict(
        verified=True,
        steps=audit["steps"],
        max_state_control_error=max_error,
        decisions_regenerated=True,
        contacts_regenerated=True,
        outcome=audit["outcome"],
        evidence_domain="simulation",
        independent_physical_validation=False,
    )
