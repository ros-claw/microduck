"""Paired revision checks with fixed sides/headings and unchanged physics gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import math
from pathlib import Path

from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.episode import run, write_json
from microduck_lab.combat.v2_config import V2Config
from probe_combat_transition import summarize
from eval_combat_team_intents import analyze_team


def trial(job):
    dest, config, physics, seed, group = job
    run(dest, config, physics, seed)
    row = dict(group=group, revision=config.controller_revision, **summarize(dest))
    if config.players == 4:
        row["team"] = analyze_team(dest)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--group",
        choices=["pressure", "counter", "free", "course", "open"],
        action="append",
        required=True,
    )
    p.add_argument(
        "--revision", type=int, choices=[1, 2], action="append", required=True
    )
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument(
        "--split", choices=["development", "validation"], default="development"
    )
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for group in a.group:
        filename = (
            "team_intents"
            if group in ("free", "open")
            else "peel_course"
            if group == "course"
            else "mixed"
        )
        path = Path("configs/combat/v2") / (filename + ".yaml")
        base = strict_load(path, V2Config)
        physics = strict_load(path.parent / base.physics_profile, Physics)
        for i in range(a.seeds):
            for side in (False, True):
                for revision in a.revision:
                    changes = dict(
                        controller_revision=revision,
                        swap=side,
                        heading_rad=i % 4 * math.pi / 2,
                    )
                    if group in ("pressure", "counter"):
                        changes["blue_style"] = group
                    if group == "open":
                        changes.update(
                            blue_v2=True, blue_v2_style="pressure", duration_s=60.0
                        )
                    cfg = replace(base, **changes)
                    dest = (
                        out / group / f"r{revision}" / f"{a.seed_base + i}-{int(side)}"
                    )
                    jobs.append((dest, cfg, physics, a.seed_base + i, group))
    write_json(
        out / "protocol.json",
        dict(
            schema="duckfu.revision-check.v1",
            split=a.split,
            seed_base=a.seed_base,
            seeds=a.seeds,
            groups=a.group,
            revisions=a.revision,
            sides=[False, True],
            headings_rad=[i % 4 * math.pi / 2 for i in range(a.seeds)],
            jobs=[
                dict(path=str(d), config=c.dict(), physics=p.__dict__, seed=s, group=g)
                for d, c, p, s, g in jobs
            ],
        ),
    )
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for task in as_completed([pool.submit(trial, j) for j in jobs]):
            row = task.result()
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(jobs),
                        group=row["group"],
                        revision=row["revision"],
                        seed=row["seed"],
                        swap=row["swap"],
                        physical=row["physical_pass"],
                        events=row["event_counts"],
                    )
                ),
                flush=True,
            )
    rows.sort(key=lambda r: (r["group"], r["seed"], r["swap"], r["revision"]))
    write_json(
        out / "report.json",
        dict(protocol=json.loads((out / "protocol.json").read_text()), rows=rows),
    )


if __name__ == "__main__":
    main()
