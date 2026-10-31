"""Read-only recovery close-ups from the preselected first held-out match."""

import argparse
import json
from pathlib import Path

from combat_director import shot
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True)
p.add_argument("--report", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
run = Path(a.run)
manifest = json.loads((run / "manifest.json").read_text())
physical = manifest["physical"]
if (
    manifest["config"].get("recovery_mode") != "learned_stabilize"
    or not manifest["capture"]
    or manifest["result"]["kind"] == "NO_CONTEST_PHYSICS"
    or not all(
        physical[k]
        for k in ("finite", "clock_ok", "external_forces_zero", "model_unchanged")
    )
    or physical["penetration_max_m"] > 0.003
    or physical["penetrating_p99_m"] > 0.0015
    or any(physical["solver_warnings"])
):
    raise ValueError(
        "Need a numerically qualified, fully recorded learned-recovery match"
    )
report = json.loads(Path(a.report).read_text())
row = next(r for r in report["rows"] if Path(r["directory"]).resolve() == run.resolve())
recoveries = sorted(
    (r for r in row["recovery_episodes"] if r["common_complete_after_s"] is not None),
    key=lambda r: r["start_s"],
)
if len(recoveries) < 2:
    raise ValueError(
        "Need two actual completed recovery episodes; never manufacture them"
    )
shots = [
    dict(
        shot(
            run,
            0,
            manifest["duration_s"],
            view="follow",
            caption="FULL UNCUT MATCH / LEARNED STABILIZATION + MEASURED HANDOFF",
        ),
        continuous_segment=True,
    )
]
for i, r in enumerate(recoveries[:2]):
    start = r["start_s"]
    complete = start + r["common_complete_after_s"]
    shots.append(
        dict(
            shot(
                run,
                start - 0.35,
                complete + 0.45,
                speed=0.25,
                view="follow",
                caption="SLOW REPLAY / JOINT CONTROL REGAINS BALANCE"
                if i == 0
                else "SLOW REPLAY / RECOVER, STAND, RETURN TO THE MATCH",
                kind="replay",
            ),
            focus_actor=r["actor"],
            body_overlay=r["actor"],
            camera_distance_m=0.66,
            camera_elevation_deg=-20,
            focus_labels_only=True,
        )
    )
    if i == 0:
        shots.append(
            dict(
                shot(
                    run,
                    complete - 0.45,
                    complete + 0.65,
                    speed=0.25,
                    view="follow",
                    caption="FOOT DETAIL / STABLE SUPPORT BEFORE HANDOFF",
                    kind="replay",
                ),
                focus_actor=r["actor"],
                body_overlay=r["actor"],
                camera_distance_m=0.49,
                camera_offset_world_m=[0, 0, -0.08],
                camera_elevation_deg=-18,
                show_names=False,
            )
        )
end = manifest["duration_s"] - manifest["physics"]["dt"]
shots.append(
    dict(
        run=str(run.resolve()),
        start_s=end,
        end_s=end,
        speed=0.0,
        frames=100,
        view="wide",
        kind="result",
        caption=manifest["result"]["kind"].replace("_", " ") + " / RESEARCH SIMULATION",
        source_manifest_sha256=digest(run / "manifest.json"),
    )
)
at = 0.0
for s in shots:
    s["frames"] = s.get(
        "frames",
        round((s["end_s"] - s["start_s"]) / s["speed"] * 50) if s["speed"] else 100,
    )
    s["output_start_s"] = at
    at += s["frames"] / 50
if at >= 120:
    raise ValueError("Recovery film must remain under two minutes")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
write_json(
    out,
    dict(
        schema="duckfu.film-plan.v1",
        edition="body_recovery",
        declutter_labels=True,
        fps=50,
        continuous=False,
        film_gate="PENDING_HUMAN_REVIEW",
        duration_s=at,
        shots=shots,
        editorial_selection="First predeclared held-out seed, unswapped; complete uninterrupted match followed by its first two completed recovery episodes, not chosen by winner or best latency.",
        report_sha256=digest(a.report),
        builder_sha256=digest(__file__),
    ),
)
print(
    json.dumps(
        dict(duration_s=at, shots=len(shots), first_episodes=recoveries[:2]), indent=2
    )
)
