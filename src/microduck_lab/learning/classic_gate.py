"""Classic rope-skip benchmark gate (CR-00) — the semantic invariants that the
snake floor-sweep can NOT pass, so Darwin can't substitute the easy variant.

The 0904 doc's §26: the benchmark's invariants must not be changeable by the
search — the snake won v3 because the benchmark let it. `microduck.rope_snake`
keeps its 93% champion; `microduck.classic_rope_skip` is the HARD game:

  FULL ROTATION REQUIRED   — the rope completes full 360° revolutions
  no floor-sweep subst.    — a snake (belly stays low, sweeps laterally) fails
  no rope park             — continuous same-direction rotation, no parking
  top pass:  rope near jumper > head + 2 cm (the loop goes OVER the head)
  bottom pass: rope near jumper < 2 cm (the belly grazes the floor)
  final tier: no mocap, no coach xfrc (the ducks really turn it)

The verifier consumes a rope-morphology history (per-frame θ/ω/R/top/bottom
near the jumper) and returns a structured verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DUCK_HEAD_Z = 0.25          # jumper head height (m)
TOP_MARGIN = 0.02           # top must clear head by this
BOTTOM_MAX = 0.02           # bottom must graze below this


@dataclass
class ClassicGateResult:
    full_rotation: bool = False        # continuous same-direction revolutions
    sustained_revs: float = 0.0
    no_parking: bool = False           # phase keeps advancing (never holds)
    top_clears_head: bool = False      # top_z (near jumper) > head + margin
    bottom_grazes: bool = False        # bottom_z (near jumper) < 2 cm
    autonomous: bool = False           # no mocap carriers, no coach xfrc
    passed: bool = False
    reasons: list[str] = field(default_factory=list)


def verify_classic(
    morph_history: np.ndarray,
    *,
    used_mocap: bool,
    used_coach_force: bool,
    final_tier: bool = False,
    min_revs: float = 3.0,
) -> ClassicGateResult:
    """Check the classic-rope invariants against a morphology history.

    morph_history columns (per frame): [theta, omega, belly_radius, top_z,
    bottom_z, plane_error] — measured NEAR the jumper's x (not a global max).

    A floor-sweep snake fails: it never does a full 360° revolution and its
    top_z never clears the head. A park-and-release fails no_parking.
    """
    r = ClassicGateResult()
    if len(morph_history) < 10:
        r.reasons.append("no morphology history")
        return r
    th = morph_history[:, 0]
    om = morph_history[:, 1]
    top = morph_history[:, 3]
    bot = morph_history[:, 4]

    # full rotation: net unwrapped angle covers multiple revolutions, mostly
    # one-directional (|net| ≈ gross)
    net = abs(th[-1] - th[0]) / (2 * math.pi)
    gross = float(np.sum(np.abs(np.diff(th)))) / (2 * math.pi)
    r.sustained_revs = net
    one_directional = gross > 0 and (net / gross) > 0.8
    r.full_rotation = net >= min_revs and one_directional
    if not r.full_rotation:
        r.reasons.append(f"no full rotation (net {net:.1f} revs, "
                         f"directionality {net/max(gross,1e-9):.0%})")

    # no parking: the rotation rate stays positive (never holds ~0 for >0.5 s)
    om_s = np.abs(om)
    dt_est = 1.0 / 50.0
    longest_stall = 0.0
    stall = 0.0
    for v in om_s:
        if v < 0.5:
            stall += dt_est
            longest_stall = max(longest_stall, stall)
        else:
            stall = 0.0
    r.no_parking = longest_stall < 0.5 and r.full_rotation
    if not r.no_parking:
        r.reasons.append(f"rope parks/stalls (longest {longest_stall:.2f} s)")

    # top clears the jumper's head, bottom grazes the floor — measured NEAR the
    # jumper's x (a whip to 30 cm at one end is not the loop clearing the head).
    # Per revolution the loop sweeps bottom→top, so use the ENVELOPE of the
    # sweep: the upper quantile of top_z (revolution tops) and the LOWER
    # quantile of bottom_z (the belly's reliable lowest reach — the floor-graze
    # the jumper must hop over). p95/p5, not the median (which sits mid-sweep).
    top_q = float(np.quantile(top, 0.95))
    bot_q = float(np.quantile(bot, 0.05))
    r.top_clears_head = top_q > DUCK_HEAD_Z + TOP_MARGIN
    r.bottom_grazes = bot_q < BOTTOM_MAX
    if not r.top_clears_head:
        r.reasons.append(f"top {top_q*100:.0f} cm does not clear head "
                         f"({(DUCK_HEAD_Z+TOP_MARGIN)*100:.0f} cm)")
    if not r.bottom_grazes:
        r.reasons.append(f"bottom {bot_q*100:.0f} cm does not graze the floor")

    # autonomy tiers
    r.autonomous = (not used_mocap) and (not used_coach_force)
    if final_tier and not r.autonomous:
        r.reasons.append("final tier requires no mocap carriers and no coach xfrc")

    core = r.full_rotation and r.no_parking and r.top_clears_head and r.bottom_grazes
    r.passed = core and (r.autonomous if final_tier else True)
    if r.passed:
        r.reasons.append("CLASSIC GATE: PASS")
    return r
