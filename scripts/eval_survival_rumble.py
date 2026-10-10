#!/usr/bin/env python3
"""Frozen local gameplay/physics validation; this script never publishes anything."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from microduck_lab.arena.survival_rumble import SurvivalConfig, run_survival


def evaluate(job):
    path, seed, config = job
    a = run_survival(path, seed, config)
    contacts = [e for e in a["events"] if e["state"] == "BODY_CONTACT"]
    captures = [e for e in a["events"] if e["state"] == "BEACON_CAPTURE"]
    physical = (
        a["finite"]
        and not a["time_reset"]
        and a["external_forces_zero"]
        and not any(a["solver_warnings"])
        and a["penetration_max_m"] <= 0.003
        and a["motor_force_max_Nm"] <= 0.640501
        and a["arena_motor_force_max_Nm"] <= 4.000001
    )
    return dict(
        seed=seed,
        sole_survivor=(
            a["outcome"]["status"] == "WINNER"
            and a["outcome"]["alive"] == [a["outcome"]["winner"]]
            and a["final_alive"] == [a["outcome"]["winner"]]
            and len([e for e in a["events"] if e["state"] == "ELIMINATED"]) == 3
            and a["final_observations"][a["outcome"]["winner"]]["upright"]
            and 12
            in a["final_observations"][a["outcome"]["winner"]]["supporting_tiles"]
        ),
        final_alive=a["final_alive"],
        arena_motor_force_max_Nm=a["arena_motor_force_max_Nm"],
        physical_pass=physical,
        outcome=a["outcome"],
        duration_s=a["duration"],
        penetration_max_m=a["penetration_max_m"],
        first_contact_s=contacts[0]["time"] if contacts else None,
        body_contact_events=len(contacts),
        captures=len(captures),
        distinct_scorers=len({e["body_id"] for e in captures}),
        stationary_fraction=a["stationary_fraction"],
        no_safe_option_fraction=a["no_safe_option_fraction"],
        contested_s=a["contested_s"],
        travelled_m=a["travelled_m"],
        source_sha256=a["source_sha256"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=62000)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = []
        for row in pool.map(
            evaluate,
            [
                (root / str(s), s, SurvivalConfig())
                for s in range(a.start, a.start + a.count)
            ],
        ):
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "seed",
                            "physical_pass",
                            "outcome",
                            "captures",
                            "distinct_scorers",
                        )
                    }
                ),
                flush=True,
            )
    winners = sum(r["sole_survivor"] for r in rows)
    result = dict(
        status="unaccepted development evaluation",
        count=len(rows),
        physical_passes=sum(r["physical_pass"] for r in rows),
        winners=winners,
        physical_gate=all(r["physical_pass"] for r in rows),
        outcome_gate=len(rows) >= 32 and winners / len(rows) >= 0.75,
        penetration_max_m=max(r["penetration_max_m"] for r in rows),
        mean_stationary_fraction=sum(r["stationary_fraction"] for r in rows)
        / len(rows),
        source_frozen=len(
            {json.dumps(r["source_sha256"], sort_keys=True) for r in rows}
        )
        == 1,
        qualification_scope="one 5x5 layout; cardinal role/spawn rotation and +/-0.005rad joint perturbations; no hardware or blind audience test",
        rows=rows,
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
