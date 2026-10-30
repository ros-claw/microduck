"""Read-only exchange diagnostics. Offline oracle labels never enter actors."""

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np


def distribution(values):
    return {
        "count": len(values),
        "median_s": float(np.median(values)) if values else None,
        "p90_s": float(np.quantile(values, 0.9)) if values else None,
        "max_s": max(values) if values else None,
    }


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    transitions = json.loads((directory / "transitions.json").read_text())
    events = [
        json.loads(x)
        for x in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    contacts = json.loads((directory / "contacts_summary.json").read_text())
    period = 1 / manifest["physics"]["motor_hz"]
    actors = {
        n: dict(
            phase_seconds=Counter(),
            state_seconds=Counter(),
            decisions=Counter(),
            end_reasons=Counter(),
            mask_available=Counter(),
            missed_reference_opportunity_seconds=Counter(),
            reward={},
            phases=[],
            transitions=[],
        )
        for n in transitions
    }
    rows, previous = defaultdict(list), {}
    with gzip.open(directory / "tactical_decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["sim_step"]].append(row)
    # Oracle reference uses recorded enemy phases solely to find blocking causes.
    # It is not a label of executable opportunity or an actor observation.
    for step_rows in rows.values():
        by_actor = {r["actor"]: r for r in step_rows}
        for row in step_rows:
            n, now, body = row["actor"], row["time_s"], row["observation"]
            a = actors[n]
            phase, skill = row["phase"], row["skill"]
            a["phase_seconds"][skill + ":" + phase] += period
            speed = math.hypot(*body["velocity"])
            if phase == "ELIMINATED":
                state = "eliminated"
            elif body["upright"] < 0.65 or body["body_ground"]:
                state = "down_or_recovering"
            elif skill == "recover_guard":
                state = "recovering"
            elif body["touching_opponent"]:
                state = "standing_contact"  # Not automatically stagnation.
            elif speed < 0.035:
                state = "standing_low_motion"
            else:
                state = "free_motion"
            a["state_seconds"][state] += period
            if previous.get(n) != (skill, phase):
                a["phases"].append(dict(time_s=now, skill=skill, phase=phase))
                previous[n] = skill, phase
            # Identify selected target from the actual decision/target events.
            history = transitions[n]
            current = next((d for d in reversed(history) if d["time_s"] <= now), None)
            if current is None or current.get("target") not in by_actor:
                continue
            enemy = by_actor[current["target"]]
            target = enemy["observation"]
            delta = np.array(target["xy"]) - body["xy"]
            distance = float(np.linalg.norm(delta))
            angle = math.atan2(delta[1], delta[0]) - body["yaw"]
            angle = math.atan2(math.sin(angle), math.cos(angle))
            # Generous offline envelope; report its exclusions rather than
            # asserting all these milliseconds were physically executable.
            if (
                enemy["phase"] == "RECOVERY"
                and enemy["skill"] in ("dash_bump", "probe_bump", "committed_drive")
                and body["upright"] > 0.94
                and body["foot_support"]
                and distance < 0.55
            ):
                if skill in ("evade_step", "cut_counter") and phase != "READY":
                    reason = "locked_evade_or_handoff"
                elif phase == "RECOVERY":
                    reason = "locked_brake_recovery"
                elif abs(angle) > 0.24:
                    reason = "not_facing"
                elif not 0.20 < distance < manifest["config"]["dash_start_distance_m"]:
                    reason = "outside_attack_distance"
                elif current["action"] not in (3, 9):
                    reason = "selector_did_not_attack_or_mask_blocked"
                else:
                    reason = "attack_in_progress"
                a["missed_reference_opportunity_seconds"][reason] += period
    reward_totals, timelines = {}, {}
    for n, a in actors.items():
        trace = transitions[n]
        for decision in trace:
            a["decisions"][str(decision["action"])] += 1
            a["end_reasons"][decision["end_reason"]] += 1
            for k, value in enumerate(decision["mask"]):
                a["mask_available"][str(k)] += int(value)
        valid = [e for e in events if e["state"] == "VALID_STRIKE" and e["actor"] == n]
        total = sum(d["reward"] for d in trace)
        winner = manifest["result"]["winner"]
        terminal = (
            (3.0 if winner == manifest["spawns"][n]["team"] else -3.0)
            if winner
            else 0.0
        )
        contact_reward = (
            0.3 * len(valid)
            if manifest["config"]["schema"] == "duckfu.match.v1"
            else None
        )
        a["reward"] = dict(
            total=total,
            terminal=terminal,
            legacy_contact_reward=contact_reward,
            residual_time_or_new_components=total - terminal - (contact_reward or 0),
            discounted_transition_sum=sum(
                (0.985 ** d["time_s"]) * d["reward"] for d in trace
            ),
            discounted_note="Transition-start approximation; old logs lack within-transition reward timestamps.",
        )
        a["duration_by_action"] = {
            action: distribution(
                [d["duration_s"] for d in trace if str(d["action"]) == action]
            )
            for action in a["decisions"]
        }
        a["transitions"] = [
            dict(
                time_s=d["time_s"],
                duration_s=d["duration_s"],
                action=d["action"],
                target=d["target"],
                reason=d["end_reason"],
            )
            for d in trace
        ]
        reward_totals[n] = a["reward"]
        timelines[n] = a["phases"]
    long_contact = []
    for e in contacts["episodes"]:
        if e["duration_s"] >= 0.5:
            long_contact.append(
                dict(
                    pair=e["actor_pair"],
                    start_s=e["start_step"] * manifest["physics"]["dt"],
                    duration_s=e["duration_s"],
                    impulse_Ns=e["summed_normal_impulse_Ns"],
                )
            )
    return dict(
        schema="duckfu.engagement-diagnosis.v1",
        source_run=str(directory.resolve()),
        seed=manifest["seed"],
        result=manifest["result"],
        duration_s=manifest["duration_s"],
        config=manifest["config"],
        actors=actors,
        long_contacts=long_contact,
        detected_counter_sequences=sum(
            e["state"] == "COUNTER_SEQUENCE" for e in events
        ),
        target_changes=[e for e in events if e["state"] == "TARGET_CHANGED"],
        reward_components=reward_totals,
        timeline=timelines,
        limitations=[
            "Offline recovery-phase windows are diagnostic oracle references, not actor inputs or qualified opportunities.",
            "Standing contact / low motion is not automatically stagnation or deliberate reward exploitation.",
            "Counter detector agreement with human labels remains PENDING_HUMAN_REVIEW.",
        ],
    )


def timeline_svg(report):
    import html

    width, left, plot = 1450, 150, 1270
    duration = report["duration_s"]
    colors = {
        "ACTIVE": "#438cba",
        "WINDUP": "#eeaa42",
        "RECOVERY": "#b681cc",
        "STABILIZE": "#dc6868",
        "READY": "#787e85",
        "ELIMINATED": "#252a33",
        "PIVOT": "#37b9ae",
    }
    height = 95 + 64 * len(report["timeline"])
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101722"/>',
        f'<text x="20" y="27" fill="white" font-family="sans-serif" font-size="18">Seed {report["seed"]} / actual phase timeline / {duration:.2f}s</text>',
    ]
    for i, (n, lane) in enumerate(report["timeline"].items()):
        y = 50 + i * 64
        chunks.append(
            f'<text x="15" y="{y + 22}" fill="white" font-family="sans-serif">{html.escape(n)}</text>'
        )
        for j, seg in enumerate(lane):
            end = lane[j + 1]["time_s"] if j + 1 < len(lane) else duration
            x, w = (
                left + plot * seg["time_s"] / duration,
                plot * (end - seg["time_s"]) / duration,
            )
            title = html.escape(
                f"{seg['time_s']:.2f}–{end:.2f}s {seg['skill']} / {seg['phase']}"
            )
            chunks.append(
                f'<rect x="{x:.2f}" y="{y}" width="{max(0.2, w):.2f}" height="32" fill="{colors.get(seg["phase"], "#66aa77")}"><title>{title}</title></rect>'
            )
    chunks.append(
        f'<text x="20" y="{height - 15}" fill="white" font-family="sans-serif">Blue active / amber windup / purple brake-recovery / red recovery guard / teal pivot. Hover for source time.</text></svg>'
    )
    return "\n".join(chunks)
