"""Two clearly separated free bouts, selected for visible tactical exchanges.

Bout A shows an actual enemy miss and uninterrupted upright return contact.
Bout B shows a public-state cover assignment, real side contact and ring-outs.
This is an edited showcase, never an efficacy estimate or a fictional match.
"""

import argparse
import json
from pathlib import Path

from combat_director import shot
from eval_combat_tactics import summarize
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counter-run", required=True)
    p.add_argument("--edge-run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ra, rb = Path(a.counter_run), Path(a.edge_run)
    sa, sb = summarize(ra), summarize(rb)
    if not sa["physical_pass"] or not sb["physical_pass"]:
        raise ValueError("Do not turn a rejected physical run into a film")
    for r in (ra, rb):
        m = json.loads((r / "manifest.json").read_text())
        if (
            not m["capture"]
            or m["config"]["players"] != 4
            or m["config"]["opponent_course"] != "none"
            or m["config"]["initial_layout"] != "standard"
        ):
            raise ValueError("Need full recordings of standard free 2v2 bouts")
    counters = [
        e
        for e in sa["stable_counter_sequences"]
        if 41 < e["evade_time_s"] < 43
        and e["actor"] == "coral"
        and e["target"] == "cyan"
    ]
    if not counters or sa["seed"] != 200003 or sb["seed"] != 190002:
        raise ValueError(
            "This reviewed storyboard requires its exact audited source bouts"
        )
    es = [
        json.loads(line)
        for line in (rb / "runtime_events.jsonl").read_text().splitlines()
    ]
    outs = [e for e in es if e["state"] == "RING_OUT" and 25 < e["time_s"] < 26]
    if {e["actor"] for e in outs} != {"coral", "cyan"}:
        raise ValueError("Missing actual double ring-out")
    if not any(
        h["actor"] == "cream"
        and h["target"] == "coral"
        and h["flank_response"]
        and 8 < h["time_s"] < 8.5
        for h in sb["strikes"]
    ):
        raise ValueError("Missing measured flank response")
    shots = []

    def add(r, start, end, caption, speed=1.0, view="follow", **kw):
        s = shot(
            r, start, end, speed, view, caption, "replay" if speed < 1 else "match"
        )
        s.update(bout_label="BOUT A" if r == ra else "BOUT B", **kw)
        shots.append(s)

    add(
        rb,
        23.4,
        24.3,
        "DUCK FU / TWO TEAMS. ONE UNFORGIVING EDGE.",
        0.75,
        focus_pair=["coral", "cyan", "lavender"],
        camera_distance_m=1.1,
        camera_elevation_deg=-19,
    )
    add(ra, 0, 3.2, "BOUT A / ROOK + SHADE vs BOLT + ANCHOR")
    add(ra, 18.1, 22.4, "DRAW THE CHARGE / STEP ASIDE / HIT BACK")
    add(
        ra,
        18.45,
        19.9,
        "SHADE SIDESTEPS / ANCHOR'S DRIVE MISSES",
        0.35,
        "pair",
        focus_pair=["coral", "cyan"],
        camera_distance_m=0.96,
        camera_azimuth_deg=170,
        focus_labels_only=True,
    )
    add(
        ra,
        20.60,
        21.95,
        "TURN BACK IN / THE RETURN BUMP LANDS",
        0.35,
        "pair",
        focus_pair=["coral", "cyan"],
        camera_distance_m=1.0,
        camera_azimuth_deg=55,
        focus_labels_only=True,
    )
    add(ra, 40.9, 44.6, "LATER IN BOUT A / ROOK SCREENS, SHADE COUNTERS")
    add(
        ra,
        41.7,
        43.5,
        "SLIP THE LINE / TURN / MAKE CONTACT",
        0.30,
        "pair",
        focus_pair=["coral", "cyan"],
        camera_distance_m=1.0,
        camera_azimuth_deg=260,
        camera_elevation_deg=-18,
        focus_labels_only=True,
    )
    add(rb, 0, 6.2, "NEW BOUT / SHADE DRAWS, ROOK CUTS AROUND")
    add(rb, 6.2, 8.85, "BLUE CHANGES TARGETS / BOLT FINDS THE OPEN SIDE")
    add(
        rb,
        7.75,
        8.65,
        "FLANK REPLAY / BOLT ATTACKS FROM THE SIDE",
        0.35,
        "pair",
        focus_pair=["cream", "coral"],
        camera_distance_m=0.93,
        camera_azimuth_deg=275,
        focus_labels_only=True,
    )
    add(rb, 8.85, 13.2, "ANCHOR PRESSES ROOK / THE TEAM FIGHT SHIFTS")
    add(rb, 19.6, 23.35, "ROOK WITHDRAWS / SHADE MOVES ACROSS TO COVER")
    add(
        rb,
        23.35,
        26.3,
        "CONTACT AT THE EDGE / NEITHER SIDE IS SAFE",
        focus_pair=["lavender", "cream", "coral", "cyan"],
    )
    add(
        rb,
        23.25,
        24.75,
        "COVER BECOMES A THREE-ROBOT SCRAMBLE",
        0.35,
        "pair",
        focus_pair=["coral", "cyan", "lavender"],
        camera_distance_m=1.10,
        camera_azimuth_deg=220,
        camera_elevation_deg=-20,
        focus_labels_only=True,
    )
    add(
        rb,
        24.75,
        25.95,
        "DOUBLE RING-OUT / BOTH TEAMS LOSE A ROBOT",
        0.30,
        "pair",
        focus_pair=["coral", "cyan"],
        camera_distance_m=1.1,
        camera_azimuth_deg=112,
        camera_elevation_deg=-18,
        focus_labels_only=True,
    )
    add(rb, 26.3, 28, "ROOK AND BOLT REMAIN")
    add(
        rb,
        38.2,
        40.45,
        "LATER IN BOUT B / ROOK FINDS THE SIDE",
        view="pair",
        focus_pair=["lavender", "cream"],
        camera_distance_m=1.0,
        focus_labels_only=True,
    )
    m = json.loads((rb / "manifest.json").read_text())
    if m["result"]["kind"] != "TIMEOUT_DRAW":
        raise ValueError("Result must match the actual referee")
    end = m["duration_s"] - m["physics"]["dt"]
    shots.append(
        dict(
            run=str(rb.resolve()),
            start_s=end,
            end_s=end,
            speed=0.0,
            frames=140,
            view="follow",
            kind="result",
            bout_label="BOUT B",
            caption="TIMEOUT DRAW / LEARNED MOTORS + PUBLIC-STATE TACTICS",
            source_manifest_sha256=digest(rb / "manifest.json"),
        )
    )
    at = 0.0
    for s in shots:
        s["frames"] = s.get(
            "frames",
            round((s["end_s"] - s["start_s"]) / s["speed"] * 50) if s["speed"] else 140,
        )
        s["output_start_s"] = at
        at += s["frames"] / 50
    if at >= 120:
        raise ValueError("Keep the film under two minutes")
    cues = []

    def cue(index, offset, text):
        cues.append(dict(at_s=shots[index]["output_start_s"] + offset, text=text))

    cue(0, 0.15, "Four robots. Two teams. One unforgiving edge.")
    cue(2, 0.1, "Anchor commits. Shade steps aside, turns, and comes back in!")
    cue(6, 0.1, "Another charge. Another sidestep. Watch Shade turn into the opening!")
    cue(7, 0.15, "New bout. Shade draws Anchor forward. Rook cuts around the side.")
    cue(8, 0.1, "Blue switches targets! Bolt gets around Shade's exposed side.")
    cue(11, 0.1, "Rook is running out of room. Shade moves across to cover!")
    cue(12, 0.4, "Contact at the edge! Anchor loses his footing!")
    cue(14, 0.1, "And Shade follows him out! One robot lost on each team!")
    cue(16, 0.05, "Rook finds the side!")
    cue(17, 0.15, "Time expires. A draw.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        out,
        dict(
            schema="duckfu.film-plan.v1",
            edition="tactical",
            fps=50,
            declutter_labels=True,
            tactical_hud=True,
            team_markers=False,
            continuous=False,
            film_gate="PENDING_HUMAN_REVIEW",
            duration_s=at,
            source_run=str(rb.resolve()),
            shots=shots,
            commentary=cues,
            source_bouts={"A": str(ra.resolve()), "B": str(rb.resolve())},
            selection="Edited showcase of two free bouts, with an explicit NEW BOUT reset and A/B source labels on every shot. A selected for its faster of two uninterrupted upright miss-and-counter sequences; B selected for cover, side response and a double ring-out. Selection is not an efficacy estimate.",
            evidence=dict(
                stable_counters=sa["stable_counter_sequences"], edge_ringouts=outs
            ),
            builder_sha256=digest(__file__),
        ),
    )
    print(json.dumps(dict(duration_s=at, shots=len(shots), cues=len(cues)), indent=2))


if __name__ == "__main__":
    main()
