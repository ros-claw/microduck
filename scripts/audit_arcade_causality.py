"""Replay actuator inputs and independently verify the contact-triggered exit."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, hashlib
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.arcade import BowlingLock
from microduck_lab.parkour.evidence import verify_capture


def audit(source):
    source = Path(source)
    report = json.loads((source / "audit.json").read_text())
    if report.get("difficulty") != "arcade":
        raise ValueError("Arcade capture required")
    verify_capture(source, report["capture"])
    m = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    d = mujoco.MjData(m)
    u = dict(np.load(source / "inputs.npz"))
    d.qpos[:] = u["initial_qpos"]
    d.qvel[:] = u["initial_qvel"]
    mujoco.mj_forward(m, d)
    lock = BowlingLock()
    guide_impulse = {}
    violations = []
    force = np.zeros(6)
    motor = m.actuator("portal/motor").id
    portal_x = float(m.body("portal").pos[0])
    for k in range(len(u["time"])):
        expected = (
            0.0
            if lock.unlocked and d.body("duck/trunk_base").xpos[0] <= portal_x + 0.25
            else -0.6
        )
        if abs(float(u["ctrl"][k, motor]) - expected) > 1e-12:
            violations.append(
                dict(kind="gate_without_qualified_strike", time=float(d.time))
            )
        for guide in ["gutter_left", "gutter_right"]:
            active = bool(u["eq_active"][k, m.equality(guide + "/hold").id])
            if active != (guide_impulse.get(guide, 0.0) <= 0.02):
                violations.append(
                    dict(
                        kind="guide_release_without_measured_boss_impact",
                        time=float(d.time),
                        guide=guide,
                    )
                )
        for name in ["playball", "pin0", "pin1", "pin2"]:
            if np.any(u["xfrc_applied"][k, m.body(name).id]):
                violations.append(
                    dict(
                        kind="externally_driven_bowling_prop",
                        time=float(d.time),
                        prop=name,
                    )
                )
        d.ctrl[:] = u["ctrl"][k]
        d.eq_active[:] = u["eq_active"][k]
        d.xfrc_applied[:] = u["xfrc_applied"][k]
        for _ in range(100):
            mujoco.mj_step(m, d)
            lock.sample(m, d)
            for i, c in enumerate(d.contact):
                names = [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
                if "boss" not in names:
                    continue
                for guide in ["gutter_left", "gutter_right"]:
                    if guide in names:
                        mujoco.mj_contactForce(m, d, i, force)
                        guide_impulse[guide] = (
                            guide_impulse.get(guide, 0)
                            + max(0, float(force[0])) * m.opt.timestep
                        )
    limited = bool(
        m.actuator_forcelimited[motor]
        and np.allclose(m.actuator_forcerange[motor], [-6, 6])
    )
    output = dict(
        capture=report["capture"],
        passed=bool(
            lock.unlocked and not violations and limited and not d.warning.number.any()
        ),
        violations=violations,
        gate_force_limit_N=6,
        gate_force_limited=limited,
        bowling=lock.report(),
        events=lock.events,
        guide_boss_impulses_Ns=guide_impulse,
        method="Initialize once; replay recorded 50 Hz inputs; inspect causal contacts at every .2 ms step. Gate command and guide weld activation must follow previously measured contact state. No applied force on ball/pins.",
    )
    (source / "causality.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    args = parser.parse_args()
    audit(args.source)
