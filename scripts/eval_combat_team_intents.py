"""Audit actual target changes/contact intervention; courses are labelled."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import numpy as np
from microduck_lab.combat.config import strict_load, Physics
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.episode import run, write_json
from probe_combat_transition import summarize


def qualifies_intervention(
    delay_s, progress_m, role, current_target, requested_target, plan_matches
):
    return bool(
        delay_s >= 0.12
        and progress_m >= 0.03
        and role == "PEEL"
        and current_target == requested_target
        and plan_matches
    )


def analyze_team(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    events = [
        json.loads(line)
        for line in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    episodes = json.loads((directory / "contacts_summary.json").read_text())["episodes"]
    dt = m["physics"]["dt"]
    frames = {}
    with gzip.open(directory / "tactical_decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            frames.setdefault(row["sim_step"], {})[row["actor"]] = row
    interventions = []
    for change in events:
        if (
            change["state"] != "TARGET_CHANGED"
            or change.get("intent") != "PEEL_COVER"
            or change.get("previous") is None
        ):
            continue
        actor, target = change["actor"], change["target"]
        plans = [
            e
            for e in events
            if e["state"] == "TEAM_PLAN"
            and e["time_s"] <= change["time_s"]
            and e.get("assignments", {}).get(actor, {}).get("role") == "PEEL"
            and e["assignments"][actor]["target"] == target
        ]
        if not plans:
            continue
        protected = plans[-1]["assignments"][actor]["protected"]
        contacts = [
            e
            for e in episodes
            if set(e["actor_pair"]) == {actor, target}
            and change["time_s"] <= e["start_step"] * dt <= change["time_s"] + 3.0
            and e["summed_normal_impulse_Ns"] >= 0.01
        ]
        contact = min(contacts, key=lambda e: e["start_step"]) if contacts else None
        time = contact["start_step"] * dt if contact else change["time_s"]
        progress, role, current_target, plan_matches = 0.0, None, None, False
        if contact:
            first = frames[round(change["time_s"] * 50) * round(0.02 / dt)]
            last = frames[int(time * 50) * round(0.02 / dt)]
            delta = (
                np.array(first[target]["observation"]["xy"])
                - first[actor]["observation"]["xy"]
            )
            delta /= max(1e-9, np.linalg.norm(delta))
            progress = float(
                (
                    np.array(last[actor]["observation"]["xy"])
                    - first[actor]["observation"]["xy"]
                )
                @ delta
            )
            role, current_target = last[actor].get("role"), last[actor].get("target")
            current_plans = [
                e
                for e in events
                if e["state"] == "TEAM_PLAN"
                and e["time_s"] <= time
                and actor in e.get("assignments", {})
            ]
            if current_plans:
                assignment = current_plans[-1]["assignments"][actor]
                plan_matches = (
                    assignment["role"] == "PEEL"
                    and assignment["target"] == target
                    and assignment.get("protected") == protected
                )
        qualified = bool(
            contact
            and qualifies_intervention(
                time - change["time_s"],
                progress,
                role,
                current_target,
                target,
                plan_matches,
            )
        )
        proximity = {}
        for label, start, end in [
            ("before", time - 0.6, time),
            ("after", time + 0.3, time + 1.3),
        ]:
            values = []
            for frame in frames.values():
                t = next(iter(frame.values()))["time_s"]
                if not start <= t < end or protected not in frame:
                    continue
                xy = np.array(frame[protected]["observation"]["xy"])
                values.append(
                    sum(
                        max(
                            0.0, 1 - np.linalg.norm(xy - row["observation"]["xy"]) / 0.5
                        )
                        for n, row in frame.items()
                        if m["spawns"][n]["team"] != m["spawns"][protected]["team"]
                    )
                )
            proximity[label] = float(np.mean(values)) if values else None
            proximity[label + "_seconds"] = len(values) / 50
        full = (
            proximity["before_seconds"] >= 0.58 and proximity["after_seconds"] >= 0.98
        )
        relief = bool(
            qualified and full and proximity["after"] < proximity["before"] - 0.1
        )
        interventions.append(
            dict(
                state="TEAM_INTERVENTION"
                if qualified
                else "UNRESOLVED_PLANNED_CONTACT"
                if contact
                else "RETARGET_WITHOUT_QUALIFIED_CONTACT",
                actor=actor,
                target=target,
                protected=protected,
                previous_target=change["previous"],
                change_s=change["time_s"],
                contact_s=time if contact else None,
                episode_id=contact["event_id"] if contact else None,
                measured_impulse_Ns=contact["summed_normal_impulse_Ns"]
                if contact
                else None,
                pressure_proximity=proximity,
                intervention_schema="duckfu.team-intervention.v2",
                actor_progress_toward_initial_target_m=progress,
                contact_delay_s=time - change["time_s"] if contact else None,
                role_at_contact=role,
                actual_target_at_contact=current_target,
                current_plan_matches=plan_matches,
                pressure_relief_observed=relief,
                note="Intent qualification requires >=0.12s, >=0.03m own directed travel, retained PEEL role/target/plan and measured contact. Proximity changes are observational, not a saved-win counterfactual.",
            )
        )
    angular_seconds, longest, streak = 0.0, 0.0, 0.0
    for frame in frames.values():
        red = [n for n in frame if m["spawns"][n]["team"] == "RED"]
        good = False
        if len(red) == 2:
            a, b = (frame[n] for n in red)
            target = a.get("target")
            if (
                target
                and target == b.get("target")
                and target in frame
                and a.get("role") != b.get("role")
            ):
                xy = np.array(frame[target]["observation"]["xy"])
                vectors = [np.array(r["observation"]["xy"]) - xy for r in (a, b)]
                distances = [np.linalg.norm(v) for v in vectors]
                angle = math.acos(
                    float(
                        np.clip(np.dot(*vectors) / max(1e-9, np.prod(distances)), -1, 1)
                    )
                )
                good = bool(
                    all(d < 0.6 for d in distances)
                    and math.pi / 4 <= angle <= 3 * math.pi / 4
                )
        streak = streak + 0.02 if good else 0.0
        longest = max(longest, streak)
        angular_seconds += 0.02 * good
    summary = summarize(directory)
    return dict(
        **summary,
        layout=m["config"].get("initial_layout", "standard"),
        team_plans=[e for e in events if e["state"] == "TEAM_PLAN"],
        interventions=interventions,
        same_target_different_role_45_135_degree_seconds=angular_seconds,
        longest_angular_position_s=longest,
        qualification_note="Angular positions are not alone proof of effective coordinated pressure.",
    )


def trial(job):
    directory, cfg, physics, seed = job
    run(directory, cfg, physics, seed)
    return analyze_team(directory)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append")
    p.add_argument("--config")
    p.add_argument("--seed-base", type=int, default=92001)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if a.run:
        rows = [analyze_team(d) for d in a.run]
        protocol = dict(
            split="existing_development",
            selection="Explicit existing runs; no holdout claim",
        )
    elif a.config:
        cfg = strict_load(a.config, V2Config)
        physics = strict_load(Path(a.config).parent / cfg.physics_profile, Physics)
        jobs = [
            (
                out / "runs" / f"{a.seed_base + i}-{side}",
                replace(cfg, heading_rad=i * math.pi / 2, swap=bool(side)),
                physics,
                a.seed_base + i,
            )
            for i in range(a.seeds)
            for side in (0, 1)
        ]
        protocol = dict(
            config=cfg.dict(),
            physics=physics.__dict__,
            base_seeds=a.seeds,
            seed_base=a.seed_base,
            paired_sides=True,
            split="validation",
            not_independent_games=True,
        )
        write_json(out / "protocol.json", protocol)
        rows = []
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for future in as_completed([pool.submit(trial, j) for j in jobs]):
                rows.append(future.result())
                print(
                    json.dumps(
                        dict(
                            done=len(rows),
                            total=len(jobs),
                            seed=rows[-1]["seed"],
                            physical=rows[-1]["physical_pass"],
                            interventions=rows[-1]["interventions"],
                        )
                    ),
                    flush=True,
                )
    else:
        p.error("Provide --run or --config")
    write_json(
        out / "report.json",
        dict(schema="duckfu.team-intent-evaluation.v1", protocol=protocol, rows=rows),
    )


if __name__ == "__main__":
    main()
