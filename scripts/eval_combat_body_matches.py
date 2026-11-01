"""Paired free matches for legacy, stand ablation and contact-learned recovery."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np

from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.episode import run, write_json
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.world import digest


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [
        json.loads(s)
        for s in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    rows = {n: [] for n in manifest["spawns"]}
    with gzip.open(directory / "tactical_decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            rows[r["actor"]].append(r)
    episodes = []
    for actor, trace in rows.items():
        starts = [
            e["time_s"]
            for e in events
            if e.get("actor") == actor and e["state"] == "RECOVERY_STARTED"
        ]
        eliminated = [
            e["time_s"]
            for e in events
            if e.get("actor") == actor and e["state"] in ("RING_OUT", "DOWN_COUNT")
        ]
        for i, start in enumerate(starts):
            end = min([manifest["duration_s"], *eliminated, *starts[i + 1 : i + 2]])
            stable = complete = returned = None
            for r in trace:
                t = r["time_s"]
                if t < start or t > end:
                    continue
                b = r["observation"]
                good = (
                    b["upright"] > 0.94
                    and b["z"] > 0.105
                    and b["foot_support"]
                    and not b["body_ground"]
                    and math.hypot(*b["velocity"]) < 0.16
                )
                stable = (t if stable is None else stable) if good else None
                if complete is None and stable is not None and t - stable >= 0.3 - 1e-9:
                    complete = t
                if (
                    complete is not None
                    and returned is None
                    and r["command"]["owner"] not in ("recovery", "eliminated_hold")
                    and b["upright"] > 0.94
                    and b["foot_support"]
                ):
                    returned = t
            duration = complete - start if complete is not None else None
            episodes.append(
                dict(
                    actor=actor,
                    start_s=start,
                    observed_until_s=end,
                    common_complete_after_s=duration,
                    within_3s=duration is not None and duration <= 3,
                    returned_to_normal_command_after_s=returned - start
                    if returned is not None
                    else None,
                )
            )
    p = manifest["physical"]
    physical = (
        all(
            p[k]
            for k in ("finite", "clock_ok", "external_forces_zero", "model_unchanged")
        )
        and not any(p["solver_warnings"])
        and p["penetration_max_m"] <= 0.003
        and p["penetrating_p99_m"] <= 0.0015
        and p["torque_max_Nm"] <= 0.6405001
        and manifest["result"]["kind"] != "NO_CONTEST_PHYSICS"
    )
    initial = np.load(directory / "initial_state.npz")["state"]
    return dict(
        directory=str(directory.resolve()),
        seed=manifest["seed"],
        swap=manifest["config"]["swap"],
        mode=manifest["config"]["recovery_mode"],
        duration_s=manifest["duration_s"],
        physical_pass=physical,
        physical=p,
        result=manifest["result"],
        recovery_episodes=episodes,
        learned_motor_s=sum(
            r["command"]["policy"] == "recovery_candidate"
            for trace in rows.values()
            for r in trace
        )
        / 50,
        event_counts={
            name: sum(e["state"] == name for e in events)
            for name in ("DOWN", "RECOVERY_STARTED", "RECOVERED", "RING_OUT")
        },
        initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        source_manifest_sha256=digest(directory / "manifest.json"),
    )


def trial(job):
    directory, cfg, physics, seed, capture = job
    run(directory, cfg, physics, seed, capture=capture)
    return summarize(directory)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/combat/v2/team_open_coordinated.yaml")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--mode",
        action="append",
        choices=["legacy", "stand_first", "contact_learned", "learned_stabilize"],
        required=True,
    )
    p.add_argument("--motor")
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--capture-first", action="store_true")
    p.add_argument("--split", choices=["development", "validation"], required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if (
        any(m in a.mode for m in ("contact_learned", "learned_stabilize"))
        and not a.motor
    ):
        p.error("Explicit learned recovery motor required")
    base = strict_load(a.config, V2Config)
    physics = strict_load(Path(a.config).parent / base.physics_profile, Physics)
    jobs = []
    for i in range(a.seeds):
        for side in (False, True):
            for mode in a.mode:
                changes = dict(
                    recovery_mode=mode,
                    swap=side,
                    heading_rad=i % 4 * math.pi / 2,
                    duration_s=a.duration,
                )
                if mode in ("contact_learned", "learned_stabilize"):
                    changes.update(
                        recovery_motor_path=a.motor,
                        recovery_motor_sha256=digest(a.motor),
                    )
                cfg = replace(base, **changes)
                jobs.append(
                    (
                        out / mode / f"{a.seed_base + i}-{int(side)}",
                        cfg,
                        physics,
                        a.seed_base + i,
                        a.capture_first and i == 0 and not side,
                    )
                )
    write_json(
        out / "protocol.json",
        dict(
            schema="duckfu.body-match-check.v1",
            split=a.split,
            evaluator_sha256=digest(__file__),
            jobs=[
                dict(
                    path=str(d), config=c.dict(), physics=p.__dict__, seed=s, capture=v
                )
                for d, c, p, s, v in jobs
            ],
        ),
    )
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for task in as_completed([pool.submit(trial, j) for j in jobs]):
            r = task.result()
            rows.append(r)
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(jobs),
                        mode=r["mode"],
                        seed=r["seed"],
                        swap=r["swap"],
                        physical_pass=r["physical_pass"],
                        events=r["event_counts"],
                    )
                ),
                flush=True,
            )
    rows.sort(key=lambda r: (r["seed"], r["swap"], r["mode"]))
    summary = []
    for mode in a.mode:
        batch = [r for r in rows if r["mode"] == mode]
        valid = [r for r in batch if r["physical_pass"]]
        episodes = [e for r in valid for e in r["recovery_episodes"]]
        times = [
            e["common_complete_after_s"]
            for e in episodes
            if e["common_complete_after_s"] is not None
        ]
        summary.append(
            dict(
                mode=mode,
                games=len(batch),
                physical_pass=len(valid),
                recovery_starts=len(episodes),
                common_completed=len(times),
                within_3s=sum(e["within_3s"] for e in episodes),
                returned_to_normal_command=sum(
                    e["returned_to_normal_command_after_s"] is not None
                    for e in episodes
                ),
                completed_only_median_s=float(np.median(times)) if times else None,
                down=sum(r["event_counts"]["DOWN"] for r in valid),
                learned_motor_s=sum(r["learned_motor_s"] for r in valid),
            )
        )
    write_json(
        out / "report.json",
        dict(
            protocol=json.loads((out / "protocol.json").read_text()),
            summary=summary,
            rows=rows,
        ),
    )
    print(json.dumps(summary, indent=2))
