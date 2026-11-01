"""Prepare an uninterrupted reference and a measured team-support review."""

import argparse
import json
from pathlib import Path

from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest
from combat_director import shot
from combat_contact_detail import detail


def prepare(run, evaluation, out):
    run, out = Path(run), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty directory for a new review plan")
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((run / "manifest.json").read_text())
    if (
        not m["capture"]
        or m["result"]["kind"] == "NO_CONTEST_PHYSICS"
        or m["physical"]["penetration_max_m"] > 0.003
        or m["physical"]["penetrating_p99_m"] > 0.0015
    ):
        raise ValueError("Only a numerically valid full capture can be reviewed")
    report = json.loads(Path(evaluation).read_text())
    row = next(
        r
        for r in report["rows"]
        if r["group"] == "open"
        and r["revision"] == 2
        and r["seed"] == m["seed"]
        and r["swap"] == m["config"]["swap"]
    )
    interventions = [
        e for e in row["strict_interventions"] if e["state"] == "TEAM_INTERVENTION"
    ]
    if len(interventions) < 2:
        raise ValueError("Do not manufacture a multi-intervention review")
    full = shot(
        run,
        0,
        m["duration_s"],
        1,
        "follow",
        "FREE 2v2 / PUBLIC-STATE TEAM PLANS / FULL MATCH AT 1x",
    )
    shots = [full]
    names = {n: s["display_name"] for n, s in m["spawns"].items()}
    selection = []
    for i, e in enumerate(interventions[:2]):
        caption = f"{names[e['actor']]}: {names[e['previous_target']]} -> {names[e['target']]} / COVERING {names[e['protected']]}"
        slow = shot(
            run,
            e["change_s"] - 0.3,
            e["contact_s"] + 0.4,
            0.25,
            "pair",
            caption,
            "replay",
        )
        slow["focus_pair"] = [e["actor"], e["target"], e["protected"]]
        shots.append(slow)
        contact = detail(run, e["episode_id"], e["actor"])
        write_json(out / f"support-contact-{i + 1}.json", contact)
        shots.append(
            dict(
                run=str(run.resolve()),
                start_s=contact["source_time_s"],
                end_s=contact["source_time_s"],
                speed=0.0,
                frames=100,
                view="collision",
                kind="contact_detail",
                caption="SOLVED CONTACT / SINGLE-FRAME HOLD / COLLISION GEOMETRY",
                focus_pair=[e["actor"], e["target"]],
                measurement=contact,
                source_manifest_sha256=digest(run / "manifest.json"),
            )
        )
        selection.append(e)
    final = max(0, m["duration_s"] - m["physics"]["dt"])
    shots.append(
        dict(
            run=str(run.resolve()),
            start_s=final,
            end_s=final,
            speed=0.0,
            frames=100,
            view="follow",
            kind="result",
            caption="TIME LIMIT / DRAW",
            source_manifest_sha256=digest(run / "manifest.json"),
        )
    )
    common = dict(
        schema="duckfu.film-plan.v1",
        edition="reference",
        fps=50,
        simulation_only=True,
        slow_motion_uses_actual_states=True,
        film_gate="PENDING_HUMAN_REVIEW",
        audio="silent",
        source_result=m["result"],
        selected_interventions=selection,
        selection_note="Editorial selection from the frozen evaluation; source seed 110001, swapped side, contains retained target changes by both teams. Demonstration is not another independent trial.",
    )
    for name, timeline, continuous in [
        ("team_reference", [full.copy()], True),
        ("team_review", shots, False),
    ]:
        at = 0.0
        for s in timeline:
            frames = (
                s["frames"]
                if s["speed"] == 0
                else round((s["end_s"] - s["start_s"]) / s["speed"] * 50)
            )
            s.update(frames=frames, output_start_s=at)
            at += frames / 50
        dest = out / f"{name}.plan.json"
        if dest.exists():
            raise FileExistsError(dest)
        write_json(
            dest, dict(**common, shots=timeline, duration_s=at, continuous=continuous)
        )
    write_json(
        out / "selection.json",
        dict(
            source_run=str(run.resolve()),
            interventions=selection,
            same_target_peel_not_counted=True,
            existing_intervention_definition_unchanged=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    prepare(a.run, a.evaluation, a.out)
