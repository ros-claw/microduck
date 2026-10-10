"""Actual running-robot/sweeper contacts, automatic stand-policy recovery."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.hazards import Hazard, HazardDriver
from microduck_lab.parkour.skills import Maneuver, lane_command
from microduck_lab.parkour.audit import Audit


def trial(args):
    height, y, speed, *extra = args
    mass, phase = extra[:2] if extra else (0.04, 0.0)
    torque = extra[2] if len(extra) > 2 else 0.025
    h = Hazard(
        "sweep",
        "sweeper",
        1.0,
        y,
        z=height,
        speed=speed,
        rotor_mass=mass,
        drive_torque=torque,
    )
    hard_contacts = bool(extra[3]) if len(extra) > 3 else False
    m, d, r, tr = build_world(hazards=[h], rails=True, hard_contacts=hard_contacts)
    driver = HazardDriver([h])
    d.qpos[m.jnt_qposadr[m.joint("sweep/joint").id]] = phase
    mujoco.mj_forward(m, d)
    audit = Audit()
    skill = None
    recoveries = []
    minimum = 1.0
    samples = []
    for i in range(450):
        if d.time < 2:
            r.active_policy = "stand"
            r.set_command()
        elif audit.fallen:
            if skill is None:
                skill = Maneuver("RECOVER", float(d.time))
            status = skill.update(m, d, r)
            if status != "RUNNING":
                recoveries.append(
                    dict(
                        status=status,
                        duration=float(d.time) - skill.started,
                        t=float(d.time),
                    )
                )
                audit.skill_result(skill, float(d.time))
                skill = None
        else:
            target = (
                -np.sign(y) * 0.15
                if any(q["status"] == "SUCCESS" for q in recoveries)
                and r.trunk_pos()[0] < 1.4
                else 0.0
            )
            r.active_policy = "run"
            r.command[:3] = lane_command(r, target, 0.7)
        driver.step(m, d, float(r.trunk_pos()[0]))
        r.step()
        for _ in range(100):
            mujoco.mj_step(m, d)
            audit.sample(m, d, r)
        minimum = min(minimum, float(d.xmat[r.trunk_body_id, 8]))
        samples.append(
            dict(
                t=float(d.time),
                pos=r.trunk_pos().tolist(),
                up=float(d.xmat[r.trunk_body_id, 8]),
            )
        )
        if r.trunk_pos()[2] < -0.3:
            break
    return dict(
        hard_contacts=hard_contacts,
        height=height,
        y=y,
        speed=speed,
        mass=mass,
        phase=phase,
        drive_torque=torque,
        min_upright=minimum,
        recoveries=recoveries,
        audit=audit.report(),
        samples=samples,
        final=samples[-1],
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = []
        for r in pool.map(
            trial,
            itertools.product(
                [0.05, 0.08, 0.12, 0.20, 0.24], [-0.15, 0.0, 0.15], [-2.2, 2.2]
            ),
        ):
            out.append(r)
            print(
                r["height"],
                r["y"],
                r["speed"],
                r["min_upright"],
                r["recoveries"],
                flush=True,
            )
    (ROOT / "artifacts/neon-escape-v2/trip-recovery-battery.json").write_text(
        json.dumps(out)
    )
