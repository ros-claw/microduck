# V2 first-round results / 首轮结果

[English methods](README.md) · [中文方法](README.zh-CN.md) · [Full summary](../../../artifacts/combat/v2/final-evaluation.json)

Frozen actor: `1e5f3a6`. V1 is preserved at `1c45c12` in its original worktree. This report distinguishes numerical validity, body behavior, observed responses, qualified team intent and human acceptance.

## Protocol and numerical gates

| Battery | Scheduled | Distinct base seeds | Numerical pass |
| --- | ---: | ---: | ---: |
| Duels: V1/full-brake/dynamic × pressure/counter × both sides | 48 | 4 | 48/48 |
| Standing-target probe/drive × both sides | 16 | 4 | 16/16 |
| Free team intentions × both sides | 8 | 4 | 8/8 |
| Team pressure course × both sides | 8 | 4 | 8/8 |
| Total | **80** | **16** | **80/80** |

Each group rotates arena orientation. Repeated sides, candidates and opponents share seeds. These are correlated observations; no 80-independent-seed confidence claim is made. Development runs use separate seeds and remain development data. Selected captures are demonstrations, not new independent trials.

Physics remains MuJoCo 3.12/libccd, 8 kHz, 50 Hz motor control, 80 Newton iterations, native mass/inertia and ±0.6405 Nm torque caps. Maximum observed penetration across these 80 runs was **1.313 mm**, below the unchanged 3 mm maximum and 1.5 mm penetrating-subset P99 proposal. Friendly contact remains enabled. No root edits, artificial knockback or predefined winner is added.

## Body distinction

| Standing-target course | Probe | Committed drive |
| --- | ---: | ---: |
| Runs | 8 | 8 |
| Median complete maneuver duration | 1.10 s | 2.06 s |
| Median complete maneuver path | 0.1735 m | 0.3819 m |
| Legacy ACTIVE response runs | 5/8 | 8/8 |
| Offline response-audit runs | 6/8 | 8/8 |
| Worst upright during attack | 0.9974 | 0.9960 |

These figures cover the complete maneuver, including windup/braking, not only ACTIVE travel. The same initial course state does not mean the two attacks enter at the same distance: their declared entry ranges differ. This measures distinct commitments and trajectories, not equal-effect attacks or anonymous human recognition.

## Duels and counter windows

| Opponent | Controller | Games | Legacy counter sequences | Offline clean counters | DOWN events |
| --- | --- | ---: | ---: | ---: | ---: |
| Pressure | V1 rule | 8 | 0 | 0 | 0 |
| Pressure | V2 full-brake | 8 | 1 | 2 | 4 |
| Pressure | V2 dynamic | 8 | 4 | 4 | 4 |
| Counter | V1 rule | 8 | 0 | 0 | 1 |
| Counter | V2 full-brake | 8 | 0 | 0 | 0 |
| Counter | V2 dynamic | 8 | 0 | 0 | 0 |

Counts are sequences, not percentages of games or independently qualified opportunities. A counter requires actual source attack, completed no-contact enemy attack and later impulse/response evidence after evasion. Unfinished source attacks are not labelled clean misses. Within-run censored windows and failures remain in the raw reports.

In the paired example (seed 91003, swapped side, enemy attack `cream:1`), both defenders evade at 2.10 s. Full-brake evasion ends at 3.62 s and response is observed at 6.36 s; dynamic evasion ends at 3.26 s and response at 5.60 s. Post-evasion latency is **2.74 → 2.34 s**, a **14.6%** reduction. The proposed 30% paired improvement is **not demonstrated**. Conditional medians over different successful sequences do not repair that gap.

V2 does not establish stable superiority over V1, nor demonstrate a 50% reduction in nonprogress contact. Original 45/60-second V1 cross-play statistics are not mixed with these 15-second development comparisons. Four DOWN events in the pressure-condition V2 variants are a real limitation, even though all numerical gates passed.

## Team evidence and stricter attribution

The initial contact-only audit found four retarget/contact candidates in the pressure course. Post-hoc intent auditing retained **one**: the actor must move at least 3 cm toward its initial requested target, contact at least 0.12 s after retargeting, and still hold the PEEL role, actual target and relevant plan at contact. The original contact-only report and raw episodes remain available.

The selected course (seed 93004, swapped side) includes ROOK switching **BOLT → ANCHOR** at **9.76 s**, followed by contact at **12.3725 s**, after **0.398 m** of directed own travel with retained intent. This supports a planned intervention. A later contact in that run fails the retained-plan check and is not a second qualified intervention.

Across eight course runs, this is **one qualified intervention in one run**. Across eight free team runs, there are **zero**. The apparent pressure decrease after a near-instant contact in another run does not qualify as intent-driven relief. **No qualified pressure-relief or saved-win causality is demonstrated.**

Same-target 45°–135° angular positions total 80.38 duck-pair seconds in the eight free runs, but geometric placement alone does not prove useful coordinated pressure. [Strict derived team report](../../../artifacts/combat/v2/team-intervention-v2.json).

## Replay, timestep and media

The selected duel regenerates **120,000 integration states**; the selected team course regenerates **160,000**, both with zero state error and identical motor histories, contacts and referee logs. [Duel proof](../../../artifacts/combat/v2/v2-film-dynamic-91003-1-replay.json) · [Team proof](../../../artifacts/combat/v2/v2-film-team_course-93004-1-replay.json).

The selected 0.4 s fixed-input window at dt/dt2 changes maximum final root position by **1.399 mm** and duck-contact impulse by approximately **5.54%**. It is a local component check, not a resumed game. Both 15-second closed-loop variants draw, but controls differ by more than 0.01 rad at **0.84 s**, and phases first diverge at **2.00 s**; maximum position-component difference later reaches **0.380 m**. Equal outcomes here do not establish timestep-invariant behavior. [Full comparison](../../../artifacts/combat/v2/selected-physics-check.json).

All eight source films pass decoded-frame and physical-source mapping checks. Composite films retain one source frame per panel per output frame, 50 fps, silence and explicit separate-run labels. Slow replay uses actual states without synthesized motion. Human comprehension and watchability remain **PENDING_HUMAN_REVIEW**.

## Scope still open

- No new motor training or tactical PPO was performed; this is rule composition on V1 learned motors.
- Dynamic recovery, held-out opponent breadth, stronger team execution and learned observation/reward ablations remain open.
- Short kick, feint deception and qualified roll escape are not implemented.
- Official ROSClaw authorization receipts and hardware transfer remain unverified.
- No Release, tag, merge or tactical Champion promotion is made.
