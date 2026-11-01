"""Offline same-criterion response audit, including 0.30s post-contact lag.

Never changes controls, rewards, referee, or historical VALID_STRIKE labels.
Displacement plus impulse is observational response, not causal damage.
"""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest

POST_CONTACT_WINDOW_S = 0.30


def measure(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    events = [
        json.loads(line)
        for line in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    episodes = json.loads((directory / "contacts_summary.json").read_text())["episodes"]
    frames = {}
    with gzip.open(directory / "tactical_decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            frames.setdefault(round(row["time_s"] * 50), {})[row["actor"]] = row[
                "observation"
            ]
    output = []
    for launch in [e for e in events if e["state"] == "ATTACK_LAUNCHED"]:
        actor, target, attack = launch["actor"], launch["target"], launch["attack_id"]
        linked = sorted(
            [
                e
                for e in episodes
                if attack in e["attacks"] and set(e["actor_pair"]) == {actor, target}
            ],
            key=lambda e: e["start_step"],
        )
        if not linked:
            continue
        first = linked[0]["start_step"] * m["physics"]["dt"]
        last = linked[-1]["end_step"] * m["physics"]["dt"]
        end = min(m["duration_s"] - 0.02, last + POST_CONTACT_WINDOW_S)
        # Another attack's new physical contact invalidates delayed attribution.
        conflicts = [
            e["start_step"] * m["physics"]["dt"]
            for e in episodes
            if target in e["actor_pair"]
            and e not in linked
            and first < e["start_step"] * m["physics"]["dt"] <= end
            and e["summed_normal_impulse_Ns"] >= 0.01
        ]
        if conflicts:
            end = min(end, min(conflicts))
        baseline = frames[int(first * 50)]
        origin = np.array(baseline[target]["xy"])
        direction = origin - baseline[actor]["xy"]
        direction /= max(1e-9, np.linalg.norm(direction))
        impulse = sum(
            float(np.array(e["normal_impulse_on_second_Ns"][:2]) @ direction)
            * (1 if e["actor_pair"][1] == target else -1)
            for e in linked
        )
        if impulse < 0.01:
            continue
        active_end = next(
            (
                e["time_s"]
                for e in events
                if e["state"] == "ATTACK_ACTIVE_ENDED" and e.get("attack_id") == attack
            ),
            m["duration_s"],
        )
        response = None
        for tick in range(int(first * 50), min(len(frames), int(end * 50) + 1)):
            if tick not in frames:
                continue
            a, b = frames[tick][actor], frames[tick][target]
            known_impulse = sum(
                float(np.array(e["normal_impulse_on_second_Ns"][:2]) @ direction)
                * (1 if e["actor_pair"][1] == target else -1)
                for e in linked
                if e["end_step"] * m["physics"]["dt"] <= tick / 50
            )
            displacement = float((np.array(b["xy"]) - origin) @ direction)
            stagger = (
                baseline[target]["upright"] - b["upright"] >= 0.08
                and float(np.array(b["velocity"]) @ direction) > 0.04
            )
            if (
                known_impulse >= 0.01
                and a["upright"] > 0.85
                and (displacement >= 0.01 or stagger)
            ):
                response = dict(
                    time_s=tick / 50,
                    displacement_m=displacement,
                    stagger=bool(stagger),
                    attacker_upright=a["upright"],
                    target_upright=b["upright"],
                    known_forward_impulse_Ns=known_impulse,
                )
                break
        if response:
            output.append(
                dict(
                    state="OBSERVED_CONTACT_RESPONSE_V2",
                    actor=actor,
                    target=target,
                    attack_id=attack,
                    first_contact_s=first,
                    last_contact_s=last,
                    forward_impulse_Ns=response["known_forward_impulse_Ns"],
                    launch_s=launch["time_s"],
                    response=response,
                    after_active=bool(response["time_s"] > active_end),
                    delay_after_active_s=max(0.0, response["time_s"] - active_end),
                    maximum_contact_lag_s=POST_CONTACT_WINDOW_S,
                    conflicts_truncated_window=bool(conflicts),
                )
            )
    evades = [
        e
        for e in events
        if e["state"] == "MANEUVER" and e.get("skill") in ("evade_step", "cut_counter")
    ]
    launches = [e for e in events if e["state"] == "ATTACK_LAUNCHED"]
    counters = []
    used = set()
    for response in output:
        candidates = [
            e
            for e in evades
            if e["actor"] == response["actor"]
            and e["target"] == response["target"]
            and 0 <= response["response"]["time_s"] - e["time_s"] <= 5.0
            and response["launch_s"] >= e["time_s"]
        ]
        for evade in reversed(candidates):
            attacks = [
                e
                for e in launches
                if e["actor"] == response["target"]
                and e["target"] == response["actor"]
                and -0.8 <= e["time_s"] - evade["time_s"] <= 1.2
            ]
            if not attacks:
                continue
            enemy = min(attacks, key=lambda e: abs(e["time_s"] - evade["time_s"]))
            enemy_id = enemy["attack_id"]
            completed = any(
                e["state"] == "ATTACK_WINDOW_ENDED"
                and e.get("attack_id") == enemy_id
                and e.get("reason") == "COMPLETE"
                for e in events
            )
            if (
                not completed
                or enemy_id in used
                or any(
                    enemy_id in e["attacks"] and response["actor"] in e["actor_pair"]
                    for e in episodes
                )
            ):
                continue
            used.add(enemy_id)
            end = next(
                (
                    e["time_s"]
                    for e in events
                    if e.get("actor") == response["actor"]
                    and e["time_s"] >= evade["time_s"]
                    and e["state"] in ("DYNAMIC_HANDOFF", "EVADE_BRAKE")
                ),
                None,
            )
            counters.append(
                dict(
                    actor=response["actor"],
                    enemy_attack_id=enemy_id,
                    attack_id=response["attack_id"],
                    evade_s=evade["time_s"],
                    evade_active_end_s=end,
                    response_s=response["response"]["time_s"],
                    latency_after_evade_s=response["response"]["time_s"] - end
                    if end is not None
                    else None,
                    after_active=response["after_active"],
                )
            )
            break
    return dict(
        schema="duckfu.contact-response-audit.v2",
        source_run=str(directory.resolve()),
        source_manifest_sha256=digest(directory / "manifest.json"),
        measurement_source_sha256=digest(__file__),
        sample_period_s=0.02,
        post_contact_window_s=POST_CONTACT_WINDOW_S,
        observed_responses=output,
        clean_evade_counters=counters,
        limitations=[
            "Only completed contact episodes establish an impulse bound, so onset times are conservative; future impulses cannot qualify earlier displacement.",
            "20ms public samples; no substep-exact displacement onset claim.",
            "Positive impulse plus displacement/stagger is observational, not counterfactual causal damage.",
            "Legacy runtime events, rewards, controls and referee are unchanged.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, action="append")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    write_json(
        out, dict(schema="duckfu.response-batch.v2", runs=[measure(d) for d in a.run])
    )


if __name__ == "__main__":
    main()
