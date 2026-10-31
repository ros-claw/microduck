"""Local motor comparison from a real evade exit, retaining qvel/action history.

Other ducks follow recorded actuator targets. This is an offline component
probe, not a free-match result; the tested motor observes current bodies only.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import CombatBank, digest
from microduck_lab.sim.runtime import DuckRuntime
from microduck_lab.combat.tactics import wrap


def probe(directory, actor, target, time_s, family):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("states.npy", "scene.mjb", "controls.npz"):
        if digest(directory / name) != manifest["hashes"][name]:
            raise ValueError("Changed source capture")
    dt = manifest["physics"]["dt"]
    start, stride = round(time_s / dt), round(0.02 / dt)
    states = np.load(directory / "states.npy", mmap_mode="r")
    tape = np.load(directory / "controls.npz")["ctrl"]
    m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, states[start], manifest["state_spec"])
    mujoco.mj_forward(m, d)
    bank = CombatBank({"walk_candidate": manifest["config"]["walk_motor_path"]})
    ducks = {
        n: DuckRuntime(m, d, bank, prefix=n + "/", name=n) for n in manifest["spawns"]
    }
    duck, enemy = ducks[actor], ducks[target]
    duck.active_policy = "walk_candidate"
    duck.last_action[:] = (
        tape[max(0, start // stride - 1), duck.act_ids] - duck.default_pose
    ).astype(np.float32)
    audit = ContactAudit(m, ducks)
    initial_position = duck.trunk_pos().copy()
    initial_speed = d.qvel[duck.trunk_qvel_adr : duck.trunk_qvel_adr + 6].copy()
    minimum_upright, first_ready, trace = 1.0, None, []
    for step in range(min(round(2.5 / dt), len(states) - start)):
        if step % stride == 0:
            delta = enemy.trunk_pos()[:2] - duck.trunk_pos()[:2]
            distance = float(np.linalg.norm(delta))
            angle = wrap(math.atan2(delta[1], delta[0]) - duck.trunk_yaw())
            sign = 1.0 if angle > 0 else -1.0
            if abs(angle) < 0.25:
                twist = (
                    0.28 if distance > 0.24 else 0.0,
                    0.0,
                    np.clip(2.8 * angle, -0.6, 0.6),
                )
            elif family == "reverse_arc":
                twist = (-0.15, sign * 0.3, sign)
            elif family == "forward_arc":
                twist = (0.28, sign * 0.3, sign)
            elif family == "side_away_turn":
                twist = (0.0, -sign * 0.3, sign)
            elif family == "side_toward_turn":
                twist = (0.0, sign * 0.3, sign)
            else:
                twist = (0.0, 0.0, sign)
            upright = float(-duck.projected_gravity()[2])
            minimum_upright = min(minimum_upright, upright)
            ready = (
                0.18 < distance < 0.27
                and abs(angle) < 0.32
                and upright > 0.96
                and audit.support[actor]
            )
            if ready and first_ready is None:
                first_ready = step * dt
            trace.append(
                dict(
                    time_s=step * dt,
                    distance_m=distance,
                    angle_rad=angle,
                    upright=upright,
                    twist=list(map(float, twist)),
                )
            )
            d.ctrl[:] = tape[min(len(tape) - 1, (start + step) // stride)]
            duck.set_command(twist)
            duck.step()
        mujoco.mj_step(m, d)
        audit.read(m, d, step, dt, {})
    summary = audit.summary()
    return dict(
        family=family,
        first_geometry_and_support_ready_s=first_ready,
        minimum_upright=minimum_upright,
        initial_root_velocity=initial_speed.tolist(),
        displacement_m=float(np.linalg.norm(duck.trunk_pos() - initial_position)),
        maximum_penetration_m=summary["penetration_max_m"],
        physical_gate=summary["gate_proposal_pass"],
        warnings=[int(w.number) for w in d.warning],
        trace=trace,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--actor", default="lavender")
    p.add_argument("--target", default="cream")
    p.add_argument("--time", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    rows = [
        probe(a.run, a.actor, a.target, a.time, family)
        for family in (
            "reverse_arc",
            "forward_arc",
            "side_away_turn",
            "side_toward_turn",
            "yaw_only",
        )
    ]
    write_json(
        out,
        dict(
            schema="duckfu.pivot-component.v1",
            source_run=str(Path(a.run).resolve()),
            source_manifest_sha256=digest(Path(a.run) / "manifest.json"),
            start_s=a.time,
            rows=rows,
            note="Physical reset once to a real recorded qpos/qvel state; prior executed action retained. Other motors use a recorded tape. No free-match or newly trained skill claim.",
        ),
    )
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k != "trace"} for row in rows], indent=2
        )
    )


if __name__ == "__main__":
    main()
