"""Separate weight/statistic output changes on frozen physical observations."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load(path):
    with gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb") as f:
        return torch.load(f, weights_only=False, map_location="cpu")["actor_state_dict"]


def network(state):
    model = nn.Sequential(
        nn.Linear(61, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 14),
    )
    model.load_state_dict(
        {k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")}
    )
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-checkpoint", required=True)
    p.add_argument("--candidate-checkpoint", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    old, new = load(a.seed_checkpoint), load(a.candidate_checkpoint)
    x = torch.tensor(np.load(a.inputs)["obs"])

    def normalize(s):
        return (x - s["obs_normalizer._mean"]) / (s["obs_normalizer._std"] + 0.01)

    original, candidate = network(old), network(new)
    with torch.inference_mode():
        baseline = original(normalize(old))
        variants = {
            "both_updated": candidate(normalize(new)),
            "weights_only": candidate(normalize(old)),
            "normalizer_only": original(normalize(new)),
        }
        result = {
            k: dict(
                rms=float((v - baseline).square().mean().sqrt()),
                max=float((v - baseline).abs().max()),
            )
            for k, v in variants.items()
        }
    result["observations"] = len(x)
    result["scope"] = (
        "Output diagnostic on reconstructed native observations; no ablated actor is exported, deployed or claimed to prove the cause of a specific fall."
    )
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
