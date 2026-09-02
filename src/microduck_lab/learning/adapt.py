"""Adaptation (ROSClaw 'Auto') + evaluation (ROSClaw 'Darwin') + promotion.

L1 procedural adaptation: the tunable surface is declared by the skill, not
arbitrary code. Search = coordinate descent over the declared bounds.
Darwin = evaluation on *holdout* seeds the search never saw.
Promotion gate = candidate must beat baseline on holdout, with no regression
on safety metrics (upright fractions), and results must be reproducible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

from ..skills.skip import SkipConfig, SkipMetrics


# The declared tunable surface for microduck.rope_skip
TUNABLE = {
    "trigger_lead_s": (0.28, 0.44),
    "jump_crouch": (0.55, 0.95),
    "frequency": (1.1, 1.6),
    "jump_land_time": (0.20, 0.45),
}


@dataclass
class Candidate:
    params: dict
    parent: str | None = None
    tag: str = ""


@dataclass
class EvalResult:
    episodes: list[SkipMetrics]
    success_rate: float
    consecutive_best: int
    turner_upright: float
    jumper_upright: float
    turner_max_down_s: float
    jumper_max_down_s: float

    @classmethod
    def of(cls, eps: list[SkipMetrics]):
        return cls(
            episodes=eps,
            success_rate=float(np.mean([e.success_rate for e in eps])),
            consecutive_best=max(e.consecutive_best for e in eps),
            turner_upright=float(np.mean([e.turner_upright_frac for e in eps])),
            jumper_upright=float(np.mean([e.jumper_upright_frac for e in eps])),
            turner_max_down_s=max(e.turner_max_down_s for e in eps),
            jumper_max_down_s=max(e.jumper_max_down_s for e in eps),
        )


def generate_candidates(base: SkipConfig, n_per_axis: int = 3) -> list[Candidate]:
    """Coordinate-wise candidates around the current champion config."""
    cands = []
    for key, (lo, hi) in TUNABLE.items():
        cur = getattr(base, key)
        span = (hi - lo) / (n_per_axis + 1)
        for k in range(1, n_per_axis + 1):
            v = float(np.clip(cur + (k - (n_per_axis + 1) / 2) * span, lo, hi))
            if abs(v - cur) < 1e-9:
                continue
            cands.append(Candidate(params={key: round(v, 4)}, tag=f"{key}={v:.3f}"))
    return cands


@dataclass
class PromotionVerdict:
    promoted: bool
    reason: str
    champion: Candidate | None
    baseline_rate: float
    candidate_rate: float


def promotion_gate(base: EvalResult, cand: EvalResult,
                   min_delta: float = 0.08,
                   max_down_s: float = 3.0) -> PromotionVerdict:
    """Darwin promotion gate: beat baseline on holdout, no safety regression.

    Safety = recovery-aware: a duck may trip (that's skipping), but it must
    always get back up — max continuous down-time bounded.
    """
    if cand.turner_max_down_s > max_down_s or cand.jumper_max_down_s > max_down_s:
        return PromotionVerdict(False, "SAFETY: unrecovered fall", None,
                                base.success_rate, cand.success_rate)
    if cand.turner_upright < 0.85:
        return PromotionVerdict(False, "SAFETY: turner instability", None,
                                base.success_rate, cand.success_rate)
    if cand.success_rate < base.success_rate + min_delta:
        return PromotionVerdict(False, "no significant improvement", None,
                                base.success_rate, cand.success_rate)
    return PromotionVerdict(True, "CHAMPION", None,
                            base.success_rate, cand.success_rate)
