#!/usr/bin/env python3
"""Fixed seed batteries; persist every outcome, including quality failures/draws."""

import argparse
import collections
import json
from pathlib import Path
from microduck_lab.arena.multiplayer import GameConfig, run_game

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
p.add_argument("--suite", choices=("two", "four", "heldout"), required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
seeds = (
    range(501, 511)
    if a.suite == "two"
    else range(601, 613)
    if a.suite == "four"
    else range(20000, 20032)
)
results = []
for j, seed in enumerate(seeds):
    config = GameConfig(
        players=2 if a.suite == "two" else 4,
        grid=4 if a.suite == "two" else 5,
        layout=("square", "ring", "cross", "terraces")[j // 8]
        if a.suite == "heldout"
        else "square",
        cadence=("steady", "rapid")[(j // 4) % 2] if a.suite == "heldout" else "steady",
    )
    result = run_game(out / str(seed), seed, config)
    quality = (
        result["finite"]
        and not result["time_reset"]
        and not any(result["solver_warnings"])
        and result["external_forces_zero"]
        and result["penetration_max_m"] <= 0.003
        and result["penetrating_contacts_p99_m"] <= 0.0015
        and result["motor_force_max_Nm"] <= 0.640501
    )
    row = dict(
        seed=seed,
        config=result["config"],
        outcome=result["outcome"],
        quality_pass=bool(quality),
        duration=result["duration"],
        penetration_max_m=result["penetration_max_m"],
        penetrating_contacts_p99_m=result["penetrating_contacts_p99_m"],
        body_contact_steps=result["body_contact_steps"],
        performance=result["performance"],
        eliminations=[e for e in result["events"] if e["state"] == "ELIMINATED"],
    )
    results.append(row)
    print(json.dumps(row), flush=True)
report = dict(
    suite=a.suite,
    seed_count=len(results),
    results=results,
    quality_pass_count=sum(r["quality_pass"] for r in results),
    outcomes=dict(collections.Counter(r["outcome"]["status"] for r in results)),
    wins=dict(
        collections.Counter(
            r["outcome"]["winner"] for r in results if r["outcome"]["winner"]
        )
    ),
    source_sha256=result["source_sha256"],
    policy_sha256=result["policy_sha256"],
    heldout=a.suite == "heldout",
    new_training=False,
    jev_used=False,
)
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
