"""Frozen cross-play: 32 base seeds x paired sides, never 64 independent seeds."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import yaml

from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.episode import run, write_json
from microduck_lab.combat.world import digest, ROOT


def trial(job):
    directory, cfg, seed, candidate, opponent, side = job
    physics = strict_load(ROOT / "configs/combat" / cfg.physics_profile, Physics)
    manifest = run(directory, cfg, physics, seed)
    events = [
        json.loads(s)
        for s in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    contacts = json.loads((directory / "contacts_summary.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    physical = manifest["physical"]
    passed = bool(
        physical["finite"]
        and physical["external_forces_zero"]
        and physical["model_unchanged"]
        and physical["clock_ok"]
        and not any(physical["solver_warnings"])
        and physical["penetration_max_m"] <= 0.003
        and physical["penetrating_p99_m"] <= 0.0015
        and physical["torque_max_Nm"] <= 0.6405001
    )
    return dict(
        candidate=candidate,
        opponent=opponent,
        base_seed=seed,
        side=side,
        run=str(directory),
        result=manifest["result"],
        physical_pass=passed,
        physical=physical,
        metrics=metrics,
        event_counts={
            state: sum(e["state"] == state for e in events)
            for state in (
                "VALID_STRIKE",
                "MISS",
                "COUNTER_SEQUENCE",
                "DOWN",
                "RECOVERED",
                "RING_OUT",
                "DOWN_COUNT",
                "TARGET_CHANGED",
            )
        },
        first_contact_s=min(
            (e["time_s"] for e in events if e["state"] == "CONTACT"), default=None
        ),
        longest_contact_s=max(
            (e["duration_s"] for e in contacts["episodes"]), default=0.0
        ),
        source_sha256=manifest["source_sha256"],
        config=cfg.dict(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    protocol_path = Path(a.protocol)
    protocol = yaml.safe_load(protocol_path.read_text())
    if (
        set(protocol)
        != {"schema", "split", "seed_base", "base_seeds", "candidates", "opponents"}
        or protocol["schema"] != "duckfu.crossplay.v1"
    ):
        raise ValueError("Strict frozen cross-play protocol required")
    if (
        not 1 <= protocol["base_seeds"] <= 64
        or not 1 <= a.jobs <= 8
        or protocol["split"] not in ("development", "validation", "final_holdout")
    ):
        raise ValueError("Invalid evaluation split/budget")
    if protocol["split"] != "development" and protocol["base_seeds"] < 32:
        raise ValueError("Qualification uses at least 32 base seeds, paired sides")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    frozen = digest(protocol_path)
    jobs = []
    for candidate in protocol["candidates"]:
        if set(candidate) != {"id", "config"}:
            raise ValueError("Unknown candidate fields")
        cfg = strict_load(candidate["config"], MatchConfig)
        for opponent in protocol["opponents"]:
            if (
                set(opponent) - {"id", "style", "policy_path", "policy_sha256"}
                or "id" not in opponent
                or "style" not in opponent
            ):
                raise ValueError("Unknown frozen opponent fields")
            for i in range(protocol["base_seeds"]):
                seed = protocol["seed_base"] + i
                for side in (0, 1):
                    common = dict(
                        swap=bool(side),
                        heading_rad=i % 4 * math.pi / 2,
                        blue_style=opponent["style"],
                        opponent_path=opponent.get("policy_path", ""),
                        opponent_sha256=opponent.get("policy_sha256", ""),
                    )
                    instance = replace(cfg, **common)
                    jobs.append(
                        (
                            out / candidate["id"] / opponent["id"] / f"{i:02d}-{side}",
                            instance,
                            seed,
                            candidate["id"],
                            opponent["id"],
                            side,
                        )
                    )
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(trial, jobs))
    if frozen != digest(protocol_path):
        raise RuntimeError("Evaluation protocol changed")
    matrix = {}
    for candidate in protocol["candidates"]:
        for opponent in protocol["opponents"]:
            group = [
                r
                for r in rows
                if r["candidate"] == candidate["id"] and r["opponent"] == opponent["id"]
            ]
            wins = sum(
                r["result"]["kind"] == "WIN" and r["result"]["winner"] == "RED"
                for r in group
            )
            losses = sum(
                r["result"]["kind"] == "WIN" and r["result"]["winner"] == "BLUE"
                for r in group
            )
            no_contest = sum(r["result"]["kind"] == "NO_CONTEST_PHYSICS" for r in group)
            draws = len(group) - wins - losses - no_contest
            matrix[candidate["id"] + " vs " + opponent["id"]] = dict(
                games=len(group),
                base_seeds=protocol["base_seeds"],
                paired_sides=True,
                wins=wins,
                draws=draws,
                losses=losses,
                no_contest=no_contest,
                physical_pass=sum(r["physical_pass"] for r in group),
                valid_strikes=sum(r["event_counts"]["VALID_STRIKE"] for r in group),
                counter_sequences=sum(
                    r["event_counts"]["COUNTER_SEQUENCE"] for r in group
                ),
                maximum_penetration_m=max(
                    r["physical"]["penetration_max_m"] for r in group
                ),
                maximum_contact_duration_s=max(r["longest_contact_s"] for r in group),
            )
    report = dict(
        schema="duckfu.crossplay-report.v1",
        split=protocol["split"],
        protocol_sha256=frozen,
        matrix=matrix,
        rows=rows,
        film_gate="PENDING_HUMAN_REVIEW",
    )
    write_json(out / "report.json", report)
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
