"""One complete actual win, with separately marked recorded-state replays.

The storyboard changes cameras and editorial timing only. It cannot choose the
winner, write a body state, alter contacts or omit time from the 1x match.
"""

import argparse
import json
from pathlib import Path

from combat_director import shot
from eval_combat_tactics import summarize
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest


def build(run, storyboard, hold_audit, out):
    run, storyboard, hold_audit, out = map(Path, (run, storyboard, hold_audit, out))
    if out.exists():
        raise FileExistsError(out)
    m = json.loads((run / "manifest.json").read_text())
    s = summarize(run)
    if (
        not m["capture"]
        or not s["physical_pass"]
        or m["result"]["kind"] != "WIN"
        or m["config"]["players"] != 4
        or m["config"]["opponent_course"] != "none"
        or m["config"]["initial_layout"] != "standard"
    ):
        raise ValueError("Need a numerically valid, fully captured free 2v2 win")
    hold = json.loads(hold_audit.read_text())
    if (
        hold["source_manifest_sha256"] != digest(run / "manifest.json")
        or not all(
            hold[k]
            for k in (
                "stable",
                "stopped",
                "finite",
                "external_forces_zero",
                "model_unchanged",
            )
        )
        or any(hold["solver_warnings"])
        or not hold["contacts"]["gate_proposal_pass"]
    ):
        raise ValueError("The separate post-match winner hold did not pass")
    referee = json.loads((run / "referee.json").read_text())
    exits = [e for e in referee["events"] if e["state"] in ("RING_OUT", "DOWN_COUNT")]
    last = exits[-1]
    teams = {n: v["team"] for n, v in m["spawns"].items()}
    if teams[last["actor"]] == m["result"]["winner"]:
        raise ValueError("Last elimination does not end the losing team")
    recent = [
        e
        for e in json.loads((run / "contacts_summary.json").read_text())["episodes"]
        if last["actor"] in e["actor_pair"]
        and any(teams[n] == m["result"]["winner"] for n in e["actor_pair"])
        and last["time_s"] - 1.5 <= e["end_step"] * m["physics"]["dt"] <= last["time_s"]
        and e["summed_normal_impulse_Ns"] > 0.01
    ]
    if not recent:
        raise ValueError("No recent enemy contact to support an edge-pressure ending")
    board = json.loads(storyboard.read_text())
    shots, cues, at, main_end = [], [], 0.0, 0.0
    for entry in board["shots"]:
        entry = dict(entry)
        voice = entry.pop("voice", None)
        offset = entry.pop("voice_offset_s", 0.15)
        kind = entry.pop("kind", "match")
        start, end = entry.pop("start_s"), entry.pop("end_s")
        speed = entry.pop("speed", 1.0)
        view = entry.pop("view", "follow")
        caption = entry.pop("caption")
        if kind == "match":
            if speed != 1.0 or abs(start - main_end) > 0.001:
                raise ValueError(
                    "The complete 1x match must remain chronological and uncut"
                )
            main_end = min(end, m["duration_s"])
            entry["continuous_segment"] = True
        elif kind == "replay":
            if not 0 < speed < 1:
                raise ValueError("Replays must declare a slower recorded-state rate")
        elif kind != "result":
            raise ValueError("Unsupported editorial shot")
        result = shot(run, start, end, speed, view, caption, kind)
        frames = entry.pop(
            "frames",
            round((result["end_s"] - result["start_s"]) / speed * 50) if speed else 0,
        )
        if kind == "result" and speed != 0:
            raise ValueError("Result hold must have zero playback speed")
        if frames < 1:
            raise ValueError("Empty editorial shot")
        result.update(entry, frames=frames, output_start_s=at, bout_label="FINAL BOUT")
        shots.append(result)
        if voice:
            cues.append(dict(at_s=at + offset, text=voice))
        at += frames / 50
    if abs(main_end - m["duration_s"]) > 0.001 or not 0 < at <= 120:
        raise ValueError("Need the whole match and a film within 120 seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        out,
        dict(
            schema="duckfu.film-plan.v1",
            edition="finisher",
            fps=50,
            continuous=False,
            declutter_labels=True,
            tactical_hud=True,
            team_markers=False,
            film_gate="PENDING_HUMAN_REVIEW",
            duration_s=at,
            source_run=str(run.resolve()),
            shots=shots,
            commentary=cues,
            result=m["result"],
            selection="One selected free bout, shown completely at 1x with explicit slow replays and a final frame hold. Selection is not a win-rate estimate. The recent final contact is measured, not a counterfactual proof that a single blow caused the ring-out.",
            evidence=dict(
                final_elimination=last,
                recent_final_contacts=recent,
                stable_counters=s["stable_counter_sequences"],
                winner_hold_sha256=digest(hold_audit),
                storyboard_sha256=digest(storyboard),
            ),
            builder_sha256=digest(__file__),
        ),
    )
    print(
        json.dumps(
            dict(
                duration_s=at,
                shots=len(shots),
                commentary=len(cues),
                result=m["result"],
            )
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--storyboard", required=True)
    p.add_argument("--hold-audit", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    build(a.run, a.storyboard, a.hold_audit, a.out)


if __name__ == "__main__":
    main()
