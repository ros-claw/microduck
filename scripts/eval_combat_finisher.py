"""Frozen-source finishing trials with paired starts and explicit motor pools.

The seed range is declared before execution. Swaps and controller/motor pairs
are correlated comparisons, not independent samples of a population win rate.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
from pathlib import Path

from eval_combat_tactics import summarize
from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.episode import run, source_hashes, write_json
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.world import digest


def trial(job):
    dest, cfg, physics, seed, group = job
    run(dest, cfg, physics, seed)
    result = summarize(dest)
    result["group"] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/combat/v2/team_finish.yaml")
    parser.add_argument("--motor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["baseline", "candidate", "paired", "crossplay", "styles"],
        default="candidate",
    )
    parser.add_argument(
        "--split", choices=["development", "heldout"], default="development"
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1:
        parser.error("Positive seed and worker counts required")
    if args.split == "heldout" and not 220001 <= args.seed_base <= 220004:
        parser.error("Heldout base seeds were predeclared as 220001..220004")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    path = Path(args.config)
    old = strict_load(path, V2Config)
    physics = strict_load(path.parent / old.physics_profile, Physics)
    motor_sha = digest(args.motor)
    new = replace(
        old, candidate_motor_path=args.motor, candidate_motor_sha256=motor_sha
    )
    groups = {"new": new}
    if args.mode == "baseline":
        groups = {"old": old}
    elif args.mode == "paired":
        groups["old"] = old
    elif args.mode == "crossplay":
        groups = {
            "new-red": replace(
                new,
                blue_motor_path=old.candidate_motor_path,
                blue_motor_sha256=old.candidate_motor_sha256,
            ),
            "new-blue": replace(
                old, blue_motor_path=args.motor, blue_motor_sha256=motor_sha
            ),
        }
    elif args.mode == "styles":
        groups = {
            "counter-counter": replace(
                new, red_v2_style="counter", blue_v2_style="counter"
            ),
            "pressure-pressure": replace(
                new, red_v2_style="pressure", blue_v2_style="pressure"
            ),
            "mixed-mixed-large": replace(
                new, red_v2_style="mixed", blue_v2_style="mixed", radius_m=1.0
            ),
        }
    jobs = [
        (
            out / group / f"{seed}-{int(swap)}",
            replace(cfg, swap=swap, heading_rad=i % 4 * math.pi / 2),
            physics,
            seed,
            group,
        )
        for i, seed in enumerate(range(args.seed_base, args.seed_base + args.seeds))
        for swap in (False, True)
        for group, cfg in groups.items()
    ]
    protocol = dict(
        schema="duckfu.finisher-check.v1",
        split=args.split,
        mode=args.mode,
        source_sha256=source_hashes(),
        motor_sha256=motor_sha,
        jobs=[
            dict(path=str(d), config=c.dict(), physics=p.__dict__, seed=s, group=g)
            for d, c, p, s, g in jobs
        ],
        note="All four robots use physical contacts and the same torque cap. Paired starts, side swaps and style groups are correlated. No scripted opponent course.",
    )
    write_json(out / "schedule.json", protocol)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(trial, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "directory",
                            "group",
                            "physical_pass",
                            "result",
                            "counts",
                            "flank_responses",
                        )
                    }
                ),
                flush=True,
            )
    summary = {}
    for group in groups:
        rs = [r for r in rows if r["group"] == group]
        summary[group] = dict(
            runs=len(rs),
            physical_pass=sum(r["physical_pass"] for r in rs),
            wins=sum(r["result"]["kind"] == "WIN" for r in rs),
            outcomes={
                k: sum(r["result"]["kind"] == k for r in rs)
                for k in sorted({r["result"]["kind"] for r in rs})
            },
            counts={k: sum(r["counts"][k] for r in rs) for k in rs[0]["counts"]},
            flank_responses=sum(r["flank_responses"] for r in rs),
            stable_counters=sum(len(r["stable_counter_sequences"]) for r in rs),
        )
    write_json(
        out / "summary.json",
        dict(
            protocol=protocol,
            groups=summary,
            rows=sorted(rows, key=lambda r: (r["group"], r["seed"], r["swap"])),
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
