"""Single-action standing-target course: probe and drive are not free bouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import math
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.config import strict_load, Physics
from microduck_lab.combat.episode import run, write_json
from probe_combat_transition import summarize


def trial(job):
    dest, cfg, physics, seed = job
    run(dest, cfg, physics, seed)
    row = summarize(dest)
    row["family"] = cfg.action_course
    row["red_valid_responses"] = sum(
        e["state"] == "VALID_STRIKE" and e.get("actor") == "lavender"
        for e in [
            json.loads(line)
            for line in (dest / "runtime_events.jsonl").read_text().splitlines()
        ]
    )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--seed-base", type=int, default=84001)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = strict_load(a.config, V2Config)
    physics = strict_load(Path(a.config).parent / cfg.physics_profile, Physics)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        config=cfg.dict(),
        physics=physics.__dict__,
        seed_base=a.seed_base,
        base_seeds=a.seeds,
        split="development",
        paired_sides=True,
        label="STANDING_TARGET_ACTION_COURSE",
    )
    write_json(out / "protocol.json", protocol)
    jobs = [
        (
            out / family / f"{a.seed_base + i}-{side}",
            replace(
                cfg,
                opponent_course="stand",
                action_course=family,
                heading_rad=i * math.pi / 2,
                swap=bool(side),
            ),
            physics,
            a.seed_base + i,
        )
        for family in ["probe_bump", "committed_drive"]
        for i in range(a.seeds)
        for side in (0, 1)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for future in as_completed([pool.submit(trial, j) for j in jobs]):
            rows.append(future.result())
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(jobs),
                        family=rows[-1]["family"],
                        seed=rows[-1]["seed"],
                        physical=rows[-1]["physical_pass"],
                        responses=rows[-1]["red_valid_responses"],
                    )
                ),
                flush=True,
            )
    write_json(out / "report.json", dict(protocol=protocol, rows=rows))


if __name__ == "__main__":
    main()
