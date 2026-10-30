"""Event-driven, read-only Duck Fu film plan; never runs or changes a match."""

import argparse
import json
from pathlib import Path

from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json


def shot(run, start, end, speed=1.0, view="wide", caption="", kind="match"):
    manifest = json.loads((run / "manifest.json").read_text())
    start, end = max(0.0, start), min(manifest["duration_s"] - 0.00025, end)
    if end <= start:
        raise ValueError("Empty physical shot")
    return dict(
        run=str(run.resolve()),
        start_s=start,
        end_s=end,
        speed=speed,
        view=view,
        caption=caption,
        kind=kind,
        source_manifest_sha256=digest(run / "manifest.json"),
    )


def plan(run, edition):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["schema"] != "duckfu.match-run.v1" or not manifest["capture"]:
        raise ValueError("An actual fully captured competition is required")
    if not all(
        manifest["physical"][k]
        for k in ("finite", "external_forces_zero", "model_unchanged", "clock_ok")
    ):
        raise ValueError("Physical audit failed")
    physical = manifest["physical"]
    if (
        physical["penetration_max_m"] > 0.003
        or physical["penetrating_p99_m"] > 0.0015
        or any(physical["solver_warnings"])
        or manifest["result"]["kind"] == "NO_CONTEST_PHYSICS"
    ):
        raise ValueError("Numerically invalid competitions cannot become final films")
    events = [
        json.loads(s) for s in (run / "runtime_events.jsonl").read_text().splitlines()
    ]
    contacts = [e for e in events if e["state"] == "VALID_STRIKE"]
    duration = manifest["duration_s"]
    if not contacts and edition != "reference":
        raise ValueError(
            "No measured response event: do not manufacture an action film"
        )
    first = contacts[0]["time_s"] if contacts else 3.0
    last = contacts[-1]["time_s"] if contacts else duration / 2
    if edition == "reference":
        shots = [
            shot(
                run, 0, duration, caption="CONTINUOUS NORMAL SPEED / NO CUTS / NO FOLEY"
            )
        ]
    elif edition == "hero":
        # Replays are explicit. Chronology segments are all from this one run.
        # The source is not forced to reach an outcome at a particular second.
        opening = max(0.0, first - 0.8)
        shots = [
            shot(
                run,
                opening,
                opening + 1.5,
                0.5,
                "pair",
                "DUCK FU / REAL CONTACT. REAL CONSEQUENCES.",
                "replay",
            ),
            shot(
                run, 0, 3.0, 1.0, "wide", "RED vs BLUE / FIXED ARENA. NO HEALTH BARS."
            ),
            shot(
                run,
                max(3.0, first - 1.0),
                min(duration, max(3.0, first - 1.0) + 17.0),
                1.0,
                "follow",
                "READ THE MOVE. FIND THE OPENING.",
            ),
        ]
        for event in (contacts[0], contacts[-1]):
            t = event["time_s"]
            shots.append(
                shot(
                    run,
                    t - 0.65,
                    t + 0.35,
                    0.25,
                    "pair",
                    "SLOW REPLAY / THE CONTACT IS PHYSICAL",
                    "replay",
                )
            )
        shots.append(
            shot(
                run,
                max(0.0, duration - 9.0),
                duration,
                1.0,
                "wide",
                "BACK TO THE MATCH / EVERY DUCK CAN LOSE",
            )
        )
    elif edition == "short":
        t = last
        shots = [
            shot(
                run,
                max(0, t - 2),
                min(duration, t + 3),
                1.0,
                "pair",
                "DUCK FU / 2 TEAMS. 4 PHYSICAL ROBOTS.",
            ),
            shot(
                run,
                t - 0.65,
                t + 0.35,
                0.25,
                "pair",
                "0.25x / NO HIDDEN KNOCKBACK",
                "replay",
            ),
            shot(
                run,
                max(0, duration - 9),
                duration,
                1.0,
                "follow",
                "ROSCLAW x MICRODUCK / SIMULATION",
            ),
        ]
    elif edition == "technical":
        focus = contacts[0]["time_s"]
        shots = [
            shot(
                run,
                0,
                min(17, duration),
                1.0,
                "wide",
                "01 / SHARED WORLD / 50 Hz MOTOR POLICIES",
            ),
            shot(
                run,
                max(0, focus - 2),
                min(duration, focus + 3),
                0.5,
                "pair",
                "02 / MANEUVER COMMITMENT / PUBLIC STATE ONLY",
                "replay",
            ),
            shot(
                run,
                max(0, focus - 0.7),
                min(duration, focus + 0.8),
                0.25,
                "collision",
                "03 / VISUAL-COMPONENT CONVEX COLLIDERS",
                "replay",
            ),
            shot(
                run,
                max(0, focus - 0.7),
                min(duration, focus + 0.8),
                0.25,
                "pair",
                "04 / REAL CONTACT IMPULSE / NO ROOT FORCE",
                "replay",
            ),
            shot(
                run,
                0,
                min(30, duration),
                1.0,
                "follow",
                "05 / TACTICS SELECT. LEARNED MOTORS ACT. PHYSICS DECIDES.",
            ),
            shot(
                run,
                max(0, duration - 25),
                duration,
                1.0,
                "wide",
                "06 / REAL RING-OUT + TEN-SECOND DOWN COUNT / NO INVINCIBILITY",
            ),
        ]
        recoveries = [
            e for e in events if e["state"] == "RECOVERED" and "duration_s" in e
        ]
        if recoveries:
            e = recoveries[0]
            recovery_shot = shot(
                run,
                e["time_s"] - e["duration_s"] - 0.5,
                e["time_s"] + 0.5,
                1.0,
                "pair",
                "RECOVERY / STABLE FOOT SUPPORT, NOT ONE UPRIGHT FRAME",
            )
            recovery_shot["focus_actor"] = e["actor"]
            shots.insert(
                -1,
                recovery_shot,
            )
    else:
        raise ValueError(edition)
    timeline, at = [], 0.0
    for s in shots:
        frames = round((s["end_s"] - s["start_s"]) / s["speed"] * 50)
        if frames < 1:
            continue
        s.update(output_start_s=at, frames=frames)
        at += frames / 50
        timeline.append(s)
    if edition != "reference":
        outcome = manifest["result"]
        caption = (
            f"{outcome['winner']} TEAM WINS"
            if outcome["kind"] == "WIN"
            else outcome["kind"].replace("_", " ")
        )
        timeline.append(
            dict(
                run=str(run.resolve()),
                start_s=duration - manifest["physics"]["dt"],
                end_s=duration - manifest["physics"]["dt"],
                speed=0.0,
                view="wide",
                kind="result",
                caption=caption + " / ROSCLAW x MICRODUCK",
                source_manifest_sha256=digest(run / "manifest.json"),
                output_start_s=at,
                frames=100,
            )
        )
        at += 2.0
    result = dict(
        schema="duckfu.film-plan.v1",
        edition=edition,
        fps=50,
        shots=timeline,
        duration_s=at,
        continuous=edition == "reference",
        simulation_only=True,
        source_run=str(run.resolve()),
        source_seed=manifest["seed"],
        source_result=manifest["result"],
        director_sha256=digest(__file__),
        slow_motion_uses_actual_states=True,
        film_gate="PENDING_HUMAN_REVIEW",
        audio="optional post-produced foley; never sensor audio",
    )
    if edition == "hero" and not 35 <= at <= 45:
        raise ValueError(
            f"Actual film plan is {at:.2f}s; choose a suitable completed match, do not pad physics"
        )
    if edition == "short" and not 15 <= at <= 20:
        raise ValueError("Insufficient actual sequence for the short edition")
    if edition == "technical" and at > 120:
        raise ValueError("Technical film exceeds the requested two minutes")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument(
        "--edition", choices=["hero", "short", "reference", "technical"], required=True
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = plan(Path(a.run), a.edition)
    write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "shots"}, indent=2))


if __name__ == "__main__":
    main()
