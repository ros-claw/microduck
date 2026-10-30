"""Frozen categorical tactical policy; NumPy inference, explicit export schema.

This selects already defined maneuvers, never joints. CPU inference has no
network call and cannot observe opponent phases or an episode's future seed.
"""

from pathlib import Path
import numpy as np

from .world import digest
from .tactics import ACTIONS, FEATURES

SCHEMA = "duckfu.tactics.categorical.v1"


def forward(weights, obs):
    x = np.asarray(obs, dtype=np.float32)
    for layer in (0, 2):
        x = np.tanh(x @ weights[f"trunk.{layer}.weight"].T + weights[f"trunk.{layer}.bias"])
    logits = x @ weights["policy.weight"].T + weights["policy.bias"]
    value = x @ weights["value.weight"].T + weights["value.bias"]
    return logits, value[..., 0]


class FrozenSelector:
    def __init__(self, path, sha, seed=0, stochastic=False):
        self.path = Path(path)
        if digest(path) != sha:
            raise ValueError("Tactical weight digest differs")
        with np.load(path, allow_pickle=False) as bundle:
            if bundle["schema"].item() != SCHEMA or bundle["actions"].tolist() != list(ACTIONS):
                raise ValueError("Unsupported tactical observation/action contract")
            self.weights = {k:bundle[k].copy() for k in bundle.files if k not in ("schema", "actions")}
        if self.weights["trunk.0.weight"].shape != (64, FEATURES):
            raise ValueError("Invalid tactical input width")
        logits, value = forward(self.weights, np.zeros(FEATURES, dtype=np.float32))
        if logits.shape != (len(ACTIONS),) or not np.isfinite(logits).all() or not np.isfinite(value):
            raise ValueError("Invalid tactical export")
        self.sha, self.rng, self.stochastic = sha, np.random.default_rng(seed), stochastic

    def select(self, obs, mask, context):
        logits, value = forward(self.weights, obs)
        logits = np.where(mask, logits, -1e9)
        probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        action = int(self.rng.choice(len(ACTIONS), p=probabilities)) if self.stochastic else int(logits.argmax())
        return action, dict(selector="learned:"+self.sha, log_probability=float(np.log(max(1e-12, probabilities[action]))),
                            value=float(value), probabilities=probabilities.tolist())


def export_tactics(model, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    weights = {k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    obs = np.random.default_rng(1030).normal(size=(64, FEATURES)).astype(np.float32)
    import torch
    with torch.no_grad():
        expected_logits, expected_value = model(torch.from_numpy(obs))
    logits, value = forward(weights, obs)
    np.testing.assert_allclose(logits, expected_logits.numpy(), rtol=1e-4, atol=2e-6)
    np.testing.assert_allclose(value, expected_value.numpy(), rtol=1e-4, atol=2e-6)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=np.array(SCHEMA), actions=np.array(ACTIONS), **weights)
    return dict(sha256=digest(path), max_export_error=float(np.max(np.abs(logits-expected_logits.numpy()))),
                observation_features=FEATURES, action_count=len(ACTIONS), normalization="fixed physical feature scales; no running normalizer")
