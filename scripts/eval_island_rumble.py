#!/usr/bin/env python3
"""Frozen-rule qualification; gameplay and physical quality gates are separate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from microduck_lab.arena.island_rumble import RumbleConfig, run_rumble


def evaluate(args):
    root, seed, players, size, final, radius, hold, mode = args
    run = Path(root) / str(seed)
    r = run_rumble(
        run,
        seed,
        RumbleConfig(
            players=players,
            tile_size=size,
            final_at_s=final,
            claim_radius_m=radius,
            claim_s=hold,
            score_mode=mode,
        ),
    )
    quality = (
        r["finite"]
        and not r["time_reset"]
        and r["external_forces_zero"]
        and not any(r["solver_warnings"])
        and r["penetration_max_m"] <= 0.003
        and r["motor_force_max_Nm"] <= 0.640501
    )
    row = dict(
        seed=seed,
        players=players,
        quality=quality,
        outcome=r["outcome"],
        duration=r["duration"],
        max_penetration_m=r["penetration_max_m"],
        body_contact_steps=r["body_contact_steps"],
        stationary_fraction=r["stationary_fraction"],
        no_safe_option_fraction=r["no_safe_option_fraction"],
        first_contact_s=min(
            (e["time"] for e in r["events"] if e["state"] == "BODY_CONTACT"),
            default=None,
        ),
        first_elimination_s=min(
            (e["time"] for e in r["events"] if e["state"] == "ELIMINATED"), default=None
        ),
        released_tiles=sum(e["state"] == "RELEASED" for e in r["events"]),
    )
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=41000)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--players", type=int, choices=(2, 4), default=4)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--tile-size", type=float, default=0.36)
    p.add_argument("--final-at", type=float, default=6.0)
    p.add_argument("--claim-radius", type=float, default=0.0)
    p.add_argument("--claim-time", type=float, default=1.5)
    p.add_argument(
        "--score-mode", choices=("continuous", "cumulative"), default="continuous"
    )
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(
            pool.map(
                evaluate,
                [
                    (
                        a.out,
                        s,
                        a.players,
                        a.tile_size,
                        a.final_at,
                        a.claim_radius,
                        a.claim_time,
                        a.score_mode,
                    )
                    for s in range(a.start, a.start + a.count)
                ],
            )
        )
    wins = sum(r["outcome"]["status"] == "WINNER" for r in rows)
    summary = dict(
        schema="microduck.island-rumble.qualification.v1",
        players=a.players,
        seeds=[a.start, a.start + a.count - 1],
        rules_frozen_before_test=True,
        winner_count=wins,
        matches=len(rows),
        winner_fraction=wins / len(rows),
        outcome_gate_passed=len(rows) >= 32 and wins / len(rows) >= 0.75,
        physical_quality_passed=sum(r["quality"] for r in rows),
        max_penetration_m=max(r["max_penetration_m"] for r in rows),
        rows=rows,
        entertainment_panel="not performed; no independent human reviewers recruited",
    )
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
