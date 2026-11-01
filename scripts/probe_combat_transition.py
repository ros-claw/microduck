"""Paired development ablation. No scripted enemy reactions or winner selection."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import numpy as np
from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.episode import run, write_json


def summarize(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    events = [
        json.loads(line)
        for line in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    contacts = json.loads((directory / "contacts_summary.json").read_text())
    trace = json.loads((directory / "transitions.json").read_text())
    motors = np.load(directory / "controls.npz")
    names = list(m["spawns"])
    red = [n for n in names if m["spawns"][n]["team"] == "RED"]
    evades = [
        e
        for e in events
        if e["state"] == "MANEUVER"
        and e.get("skill") in ("evade_step", "cut_counter")
        and e["actor"] in red
    ]
    sequences, used = [], set()
    for evade in evades:
        launches = [
            e
            for e in events
            if e["state"] == "ATTACK_LAUNCHED"
            and e["actor"] == evade["target"]
            and e.get("target") == evade["actor"]
            and -0.8 <= e["time_s"] - evade["time_s"] <= 1.2
        ]
        if not launches:
            continue
        attack = min(launches, key=lambda e: abs(e["time_s"] - evade["time_s"]))
        if attack["attack_id"] in used:
            continue
        used.add(attack["attack_id"])
        contacted = any(
            attack["attack_id"] in e["attacks"] and evade["actor"] in e["actor_pair"]
            for e in contacts["episodes"]
        )
        hit = next(
            (
                e
                for e in events
                if e["state"] == "VALID_STRIKE"
                and e["actor"] == evade["actor"]
                and e["target"] == evade["target"]
                and max(attack["time_s"], evade["time_s"])
                <= e["time_s"]
                <= evade["time_s"] + 5.0
            ),
            None,
        )
        end = next(
            (
                e["time_s"]
                for e in events
                if e.get("actor") == evade["actor"]
                and e["time_s"] >= evade["time_s"]
                and e["state"] in ("DYNAMIC_HANDOFF", "EVADE_BRAKE")
            ),
            None,
        )
        index = names.index(evade["actor"])
        sample = min(
            len(motors["body"]) - 1,
            round(
                (
                    hit["time_s"]
                    if hit
                    else min(m["duration_s"] - 0.02, evade["time_s"] + 5.0)
                )
                * 50
            ),
        )
        stable = bool(
            motors["body"][sample, index, 3] > 0.94 and motors["body"][sample, index, 6]
        )
        sequences.append(
            dict(
                actor=evade["actor"],
                enemy_attack=attack["attack_id"],
                evade_s=evade["time_s"],
                evade_active_end_s=end,
                enemy_contacted=contacted,
                counter_response_s=hit["time_s"] if hit else None,
                latency_after_evade_active_s=max(0.0, hit["time_s"] - end)
                if hit and end is not None
                else None,
                stable_at_response_or_window_end=stable,
                clean_evade_counter_response=bool(not contacted and hit and stable),
                censored_by_episode_end=evade["time_s"] + 5.0 > m["duration_s"],
            )
        )
    p = m["physical"]
    passed = (
        all(
            p[k]
            for k in ("finite", "clock_ok", "external_forces_zero", "model_unchanged")
        )
        and not any(p["solver_warnings"])
        and p["penetration_max_m"] <= 0.003
        and p["penetrating_p99_m"] <= 0.0015
        and p["torque_max_Nm"] <= 0.6405001
    )
    moves = []
    for n in red:
        index = names.index(n)
        for d in trace[n]:
            if d["action"] not in (3, 9):
                continue
            a, b = (
                round(d["time_s"] * 50),
                min(
                    len(motors["body"]) - 1, round((d["time_s"] + d["duration_s"]) * 50)
                ),
            )
            sample = motors["body"][a : b + 1, index]
            if not len(sample):
                continue
            moves.append(
                dict(
                    actor=n,
                    family="probe" if d["action"] == 9 else "drive",
                    start_s=d["time_s"],
                    duration_s=d["duration_s"],
                    path_m=float(
                        np.linalg.norm(np.diff(sample[:, :2], axis=0), axis=1).sum()
                    ),
                    minimum_upright=float(sample[:, 3].min()),
                    end_reason=d["end_reason"],
                )
            )
    return dict(
        seed=m["seed"],
        swap=m["config"]["swap"],
        source_run=str(directory.resolve()),
        result=m["result"],
        physical_pass=bool(passed),
        physical=p,
        event_counts=dict(Counter(e["state"] for e in events)),
        evade_counter_windows=sequences,
        attack_trajectories=moves,
        longest_contact_s=max(
            (e["duration_s"] for e in contacts["episodes"]), default=0.0
        ),
        red_down_fraction=float(
            np.mean(motors["body"][:, [names.index(n) for n in red], 3] < 0.65)
        ),
        duration_s=m["duration_s"],
        source_sha256=m["source_sha256"],
    )


def trial(job):
    dest, cfg, physics, seed, version = job
    run(dest, cfg, physics, seed)
    return dict(version=version, **summarize(dest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed-base", type=int, default=81011)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--opponent", choices=["pressure", "counter", "orbit"])
    p.add_argument(
        "--split", choices=["development", "validation"], default="development"
    )
    a = p.parse_args()
    cfg = strict_load(a.config, V2Config)
    if a.opponent:
        cfg = replace(cfg, blue_style=a.opponent)
    physics = strict_load(Path(a.config).parent / cfg.physics_profile, Physics)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for i in range(a.seeds):
        for side in (False, True):
            common = dict(swap=side, heading_rad=(i % 4) * math.pi / 2)
            for version in ("v1", "full_brake", "dynamic"):
                current = replace(cfg, **common, dynamic_handoff=version == "dynamic")
                if version == "v1" and cfg.opponent_course != "none":
                    current = replace(current, red_v2=False, blue_v2=False)
                elif version == "v1":
                    values = {
                        f.name: getattr(current, f.name) for f in fields(MatchConfig)
                    }
                    values.update(schema="duckfu.match.v1", red_style="counter")
                    current = MatchConfig(**values)
                dest = out / version / f"{a.seed_base + i}-{int(side)}"
                jobs.append((dest, current, physics, a.seed_base + i, version))
    protocol = dict(
        schema="duckfu.transition-ablation.v1",
        split=a.split,
        config=cfg.dict(),
        physics=physics.__dict__,
        seed_base=a.seed_base,
        base_seeds=a.seeds,
        paired_sides=True,
        versions=["v1", "full_brake", "dynamic"],
        qualification="Measured clean avoidance plus later response within five seconds; not a causal saved-win claim.",
    )
    write_json(out / "protocol.json", protocol)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        tasks = [pool.submit(trial, job) for job in jobs]
        for task in as_completed(tasks):
            row = task.result()
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(jobs),
                        version=row["version"],
                        seed=row["seed"],
                        physical=row["physical_pass"],
                        events=row["event_counts"],
                    )
                ),
                flush=True,
            )
    rows.sort(key=lambda r: (r["seed"], r["swap"], r["version"]))
    write_json(out / "report.json", dict(protocol=protocol, rows=rows))


if __name__ == "__main__":
    main()
