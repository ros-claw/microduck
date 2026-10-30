"""Development paired experiments; these are calibration data, never holdout."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np
from microduck_lab.combat.config import load
from microduck_lab.combat.episode import run, write_json


def trial(job):
    path, config, physics, seed, capture, command = job
    manifest = run(path, config, physics, seed, capture, command)
    contacts = json.loads((path / "contacts_summary.json").read_text())
    events = [
        json.loads(line)
        for line in (path / "runtime_events.jsonl").read_text().splitlines()
    ]
    metrics = json.loads((path / "metrics.json").read_text())
    bodies = np.load(path / "controls.npz")["body"]
    attacker = next(
        n for n, s in manifest["spawns"].items() if s["role"] == "coach_attacker"
    )
    defender = next(n for n in manifest["spawns"] if n != attacker)
    ai = ["lavender", "cream"].index(attacker)
    di = 1 - ai
    attacks = [
        e
        for e in events
        if e["state"] == "MANEUVER"
        and e["actor"] == attacker
        and e["phase"] == "ACTIVE"
        and e["skill"] == "dash_bump"
    ]
    first = attacks[0]["time_s"] if attacks else None
    end = next(
        (
            e["time_s"]
            for e in events
            if e["state"] == "ATTACK_WINDOW_ENDED" and e["actor"] == attacker
        ),
        None,
    )
    window = [
        e
        for e in contacts["episodes"]
        if first is not None
        and e["start_step"] * physics.dt >= first
        and e["start_step"] * physics.dt <= (end or config.duration_s) + 0.25
    ]
    if first is not None:
        k = round(first * 50)
        tail = bodies[k:, di, :2]
        yaw = bodies[k, ai, 5]
        lateral = (tail - bodies[k, di, :2]) @ np.array([-math.sin(yaw), math.cos(yaw)])
        lateral_max = float(np.abs(lateral[:60]).max())
    else:
        lateral_max = 0.0
    return dict(
        run=str(path),
        seed=seed,
        config=config.dict(),
        command=command,
        physical=manifest["physical"],
        performance=manifest["performance"],
        result=manifest["result"],
        metrics=metrics,
        first_attack_s=first,
        first_attack_end_s=end,
        first_exchange_contact=bool(window),
        first_exchange_impulse_Ns=sum(e["summed_normal_impulse_Ns"] for e in window),
        defender_lateral_max_m=lateral_max,
        actor=attacker,
        defender=defender,
        episodes=len(contacts["episodes"]),
        events=events,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--suite", choices=["command_range", "first_exchange"], required=True
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument(
        "--count",
        type=int,
        default=12,
        help="Paired initial states (three defenses each)",
    )
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not 1 <= a.jobs <= 4 or not 1 <= a.count <= 40:
        p.error("Use 1–4 workers and 1–40 development initial states")
    config, physics = load(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    if a.suite == "command_range":
        commands = [
            (0, 0, 0),
            (0.2, 0, 0),
            (0.4, 0, 0),
            (0.55, 0, 0),
            (-0.2, 0, 0),
            (0, 0.15, 0),
            (0, 0.3, 0),
            (0, -0.3, 0),
            (0, 0, 1.0),
            (0, 0, -1.0),
            (0.2, 0.2, 0),
            (0.2, -0.2, 0),
        ]
        for i, command in enumerate(commands):
            cfg = replace(config, duration_s=5.0, defense="absent", separation_m=0.6)
            jobs.append(
                (out / f"command-{i:02d}", cfg, physics, a.seed, a.capture, command)
            )
    else:
        rng = np.random.default_rng(a.seed)
        for i in range(a.count):
            cfg = replace(
                config,
                separation_m=float(rng.uniform(0.43, 0.58)),
                lateral_offset_m=float(rng.uniform(-0.025, 0.025)),
                heading_rad=(i % 4) * math.pi / 2,
                swap=bool(i % 2),
                evade_sign=1 if i % 2 else -1,
                duration_s=10.0,
            )
            for defense in ("stand", "evade", "counter"):
                jobs.append(
                    (
                        out / f"{i:02d}-{defense}",
                        replace(cfg, defense=defense),
                        physics,
                        a.seed + i,
                        a.capture,
                        None,
                    )
                )
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(trial, jobs))
    summary = dict(
        suite=a.suite,
        development_only=True,
        independent_initial_states=a.count if a.suite == "first_exchange" else 1,
        runs=len(rows),
        rows=rows,
    )
    write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
