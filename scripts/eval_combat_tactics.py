"""Predeclared, paired team trials and physical/observable tactical audits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np

from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.episode import run, write_json, source_hashes
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.world import digest


def summarize(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    events = [
        json.loads(s) for s in (root / "runtime_events.jsonl").read_text().splitlines()
    ]
    names = list(m["spawns"])
    teams = {n: s["team"] for n, s in m["spawns"].items()}
    with np.load(root / "controls.npz") as a:
        bodies = a["body"]
    traces = {}
    with gzip.open(root / "tactical_decisions.jsonl.gz", "rt") as f:
        for s in f:
            r = json.loads(s)
            traces[(round(r["time_s"] * 50), r["actor"])] = r
    hits = []
    for e in events:
        if e["state"] != "VALID_STRIKE":
            continue
        i = min(len(bodies) - 1, round(e["time_s"] * 50))
        actor, target = e["actor"], e["target"]
        # Geometry is measured at launch, before contact can rotate the victim.
        launch = next(
            (
                x
                for x in reversed(events)
                if x["state"] == "ATTACK_LAUNCHED"
                and x.get("attack_id") == e["attack_id"]
            ),
            None,
        )
        j = min(i, round(launch["time_s"] * 50)) if launch else i
        a, b = bodies[j, names.index(actor)], bodies[j, names.index(target)]
        delta = a[:2] - b[:2]
        theta = np.arctan2(delta[1], delta[0]) - b[5]
        angle = float(abs(np.arctan2(np.sin(theta), np.cos(theta))))
        # A flank hit is geometry plus a separately solver-qualified response.
        support = []
        for n in names:
            if n == actor or teams[n] != teams[actor]:
                continue
            c = bodies[i, names.index(n)]
            if (
                c[3] > 0.9
                and np.linalg.norm(c[:2] - bodies[i, names.index(target), :2]) < 0.45
            ):
                support.append(n)
        row = traces.get((max(0, i - 1), actor), {})
        hits.append(
            dict(
                **e,
                launch_target_bearing_rad=angle,
                flank_response=angle > np.pi / 3,
                nearby_supported_allies=support,
                executed_role=row.get("role"),
                executed_skill=row.get("skill"),
            )
        )
    episodes = json.loads((root / "contacts_summary.json").read_text())["episodes"]
    # The historical sequence detector permits an intervening fall. Keep its
    # count, but separately require upright uninterrupted execution for a film
    # to describe the whole sequence as a stable dodge and counter.
    stable_counters = []
    for e in events:
        if e["state"] != "COUNTER_SEQUENCE":
            continue
        i, j = round(e["evade_time_s"] * 50), round(e["counter_contact_time_s"] * 50)
        up = bodies[i : j + 1, names.index(e["actor"]), 3]
        interrupted = any(
            x.get("actor") == e["actor"]
            and x["state"] in ("RECOVERY_STARTED", "DOWN", "RING_OUT", "DOWN_COUNT")
            and e["evade_time_s"] <= x["time_s"] <= e["counter_contact_time_s"]
            for x in events
        )
        if len(up) and float(up.min()) >= 0.94 and not interrupted:
            stable_counters.append(dict(**e, minimum_upright=float(up.min())))
    # A long touching episode is a stall proxy, not a diagnosis of an actor's intent.
    long_contacts = [
        e for e in episodes if e["end_step"] - e["start_step"] > 3 / m["physics"]["dt"]
    ]
    p = m["physical"]
    physical = (
        all(
            p[k]
            for k in ("finite", "clock_ok", "external_forces_zero", "model_unchanged")
        )
        and not any(p["solver_warnings"])
        and p["penetration_max_m"] <= 0.003
        and p["penetrating_p99_m"] <= 0.0015
        and p["torque_max_Nm"] <= 0.6405001
        and m["result"]["kind"] != "NO_CONTEST_PHYSICS"
    )
    return dict(
        directory=str(root.resolve()),
        seed=m["seed"],
        swap=m["config"]["swap"],
        revision=m["config"]["controller_revision"],
        result=m["result"],
        physical_pass=physical,
        physical=p,
        duration_s=m["duration_s"],
        counts={
            s: sum(e["state"] == s for e in events)
            for s in (
                "VALID_STRIKE",
                "COUNTER_SEQUENCE",
                "MISS",
                "DOWN",
                "RING_OUT",
                "DOWN_COUNT",
                "COUNTER_ATTEMPT",
            )
        },
        flank_responses=sum(h["flank_response"] for h in hits),
        stable_counter_sequences=stable_counters,
        supported_responses=sum(bool(h["nearby_supported_allies"]) for h in hits),
        long_contact_episodes=len(long_contacts),
        strikes=hits,
        initial_state_sha256=digest(root / "initial_state.npz"),
        source_manifest_sha256=digest(root / "manifest.json"),
    )


def trial(job):
    directory, c, p, seed, capture = job
    run(directory, c, p, seed, capture=capture)
    return summarize(directory)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    c = strict_load(a.config, V2Config)
    physics = strict_load(Path(a.config).parent / c.physics_profile, Physics)
    jobs = [
        (
            out / f"r{rev}" / f"{seed}-{int(swap)}",
            replace(c, controller_revision=rev, swap=swap),
            physics,
            seed,
            False,
        )
        for seed in range(a.seed_base, a.seed_base + a.seeds)
        for swap in (False, True)
        for rev in (2, 3)
    ]
    write_json(
        out / "schedule.json",
        dict(
            source_sha256=source_hashes(),
            jobs=[dict(path=str(j[0]), seed=j[3], config=j[1].dict()) for j in jobs],
            note="Paired initial states; side swaps share base seeds. Not independent trials.",
        ),
    )
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(trial, j): j for j in jobs}
        for f in as_completed(futures):
            row = f.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "directory",
                            "physical_pass",
                            "counts",
                            "flank_responses",
                            "result",
                        )
                    }
                ),
                flush=True,
            )
    groups = {}
    for rev in (2, 3):
        rs = [r for r in rows if r["revision"] == rev]
        groups[str(rev)] = dict(
            runs=len(rs),
            physical_pass=sum(r["physical_pass"] for r in rs),
            counts={s: sum(r["counts"][s] for r in rs) for s in rs[0]["counts"]},
            flank_responses=sum(r["flank_responses"] for r in rs),
            stable_counter_sequences=sum(
                len(r["stable_counter_sequences"]) for r in rs
            ),
            supported_responses=sum(r["supported_responses"] for r in rs),
            long_contact_episodes=sum(r["long_contact_episodes"] for r in rs),
            outcomes={
                k: sum(r["result"]["kind"] == k for r in rs)
                for k in sorted(set(r["result"]["kind"] for r in rs))
            },
        )
    write_json(
        out / "summary.json",
        dict(
            groups=groups,
            rows=sorted(rows, key=lambda r: (r["seed"], r["swap"], r["revision"])),
        ),
    )
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
