"""Numerically verify the mandatory normalized ONNX export against its actor."""

import argparse, json, hashlib
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from torch import nn


def check(checkpoint, policy, output):
    checkpoint, policy = Path(checkpoint), Path(policy)
    a = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "actor_state_dict"
    ]
    net = nn.Sequential(
        nn.Linear(61, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 14),
    )
    net.load_state_dict(
        {k.removeprefix("mlp."): v for k, v in a.items() if k.startswith("mlp.")}
    )
    rng = np.random.default_rng(27)
    x = rng.normal(size=(128, 61)).astype(np.float32)
    with torch.no_grad():
        expected = net(
            (torch.from_numpy(x) - a["obs_normalizer._mean"])
            / (a["obs_normalizer._std"] + 0.01)
        ).numpy()
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    actual = np.concatenate(
        [session.run(None, {session.get_inputs()[0].name: z[None]})[0] for z in x]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    report = dict(
        passed=True,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        samples=len(x),
        max_abs_error=float(abs(actual - expected).max()),
        method="CPU Torch actor + checkpoint empirical normalization (epsilon .01) versus standard scripts/export.py ONNX; random 61-D numerical inputs, not behavioral validation.",
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    check(a.checkpoint, a.policy, a.output)
