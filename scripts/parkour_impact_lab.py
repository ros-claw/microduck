"""Finite-mass capsule impact battery with every-substep contact auditing.

A motorized sled approaches a standing duck at a measured speed. Motor force
is bounded and is cut on first contact; no obstacle or duck pose is prescribed
once the experiment starts. This is an impact test, not a game win metric.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, math, sys
from pathlib import Path
from dataclasses import dataclass
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world


@dataclass
class Striker:
    speed: float
    height: float
    angle: float

    def add_to(self, spec):
        a = math.radians(self.angle)
        self.axis = np.array([math.cos(a), math.sin(a), 0.0])
        b = spec.worldbody.add_body(
            name="striker", pos=[*(-0.40 * self.axis[:2]), self.height]
        )
        b.add_joint(
            name="striker/slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=self.axis,
            damping=0,
        )
        b.add_geom(
            name="striker/geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[
                -0.07 * math.sin(a),
                0.07 * math.cos(a),
                0,
                0.07 * math.sin(a),
                -0.07 * math.cos(a),
                0,
            ],
            size=[0.014],
            mass=0.15,
            rgba=[1, 0.05, 0.1, 1],
            priority=1,
            solref=[0.002, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )
        motor = spec.add_actuator(
            name="striker/motor",
            target="striker/slide",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        motor.gainprm[0] = 1
        motor.forcelimited = True
        motor.forcerange = [-0.5, 0.5]


def trial(speed, height, angle, timeconst=0.002, impedance=0.95, margin=0.0):
    striker = Striker(speed, height, angle)
    m, d, r, _ = build_world(hazards=[striker])
    gid = m.geom("striker/geom").id
    m.geom_solref[gid] = [timeconst, 1.0]
    m.geom_solimp[gid] = [impedance, 0.999, 0.0005, 0.5, 2.0]
    m.geom_margin[gid] = margin
    aid = m.actuator("striker/motor").id
    vadr = m.jnt_dofadr[m.joint("striker/slide").id]
    penetrations = []
    impulse = 0.0
    first = None
    minimum_up = 1.0
    vpost = None
    energy0 = None
    energy1 = None
    targets = set()
    for i in range(round((3.0 + 0.5 / speed) * 50)):
        r.active_policy = "stand"
        r.set_command()
        r.step()
        for _ in range(100):
            d.ctrl[aid] = (
                0.0
                if d.time < 2 or first is not None
                else np.clip(3 * (speed - d.qvel[vadr]), -0.5, 0.5)
            )
            mujoco.mj_step(m, d)
            for ci, c in enumerate(d.contact):
                if gid not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == gid else c.geom1
                if not m.body(m.geom_bodyid[other]).name.startswith("duck/"):
                    continue
                if c.dist > 0:
                    continue
                if first is None:
                    first = dict(
                        t=float(d.time),
                        striker_speed=float(d.qvel[vadr]),
                        duck_velocity=r.trunk_linvel().tolist(),
                    )
                    energy0 = float(sum(d.energy))
                targets.add(m.body(m.geom_bodyid[other]).name)
                penetrations.append(max(0.0, -float(c.dist)))
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, force)
                impulse += max(0.0, force[0]) * m.opt.timestep
            minimum_up = min(minimum_up, float(d.xmat[r.trunk_body_id, 8]))
            if first is not None and d.time >= first["t"] + 0.1 and vpost is None:
                vpost = r.trunk_linvel().tolist()
                energy1 = float(sum(d.energy))
    p99 = float(np.quantile(penetrations, 0.99)) if penetrations else None
    mx = max(penetrations, default=None)
    return dict(
        speed=speed,
        height=height,
        angle=angle,
        solref=timeconst,
        solimp_min=impedance,
        margin=margin,
        first=first,
        targets=sorted(targets),
        contact_samples=len(penetrations),
        p99_penetration_m=p99,
        max_penetration_m=mx,
        normal_impulse_Ns=float(impulse),
        post_impact_velocity=vpost,
        min_upright=minimum_up,
        energy_change_100ms=None if energy1 is None else energy1 - energy0,
        energy_note="Total system mechanical energy; active duck motors perform work, not a conservation residual.",
        warnings=d.warning.number.tolist(),
        passed=bool(
            mx is not None
            and mx < 0.0015
            and p99 < 0.001
            and not d.warning.number.any()
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solref", type=float, default=0.002)
    p.add_argument("--impedance", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = []
    for speed in [0.2, 0.4, 0.6, 0.8]:
        for height in [0.055, 0.115, 0.23]:
            for angle in [0, 30, 60]:
                r = trial(speed, height, angle, a.solref, a.impedance, a.margin)
                out.append(r)
                print(
                    speed,
                    height,
                    angle,
                    r["passed"],
                    r["max_penetration_m"],
                    flush=True,
                )
    Path(a.output).write_text(
        json.dumps(
            dict(results=out, passed=sum(r["passed"] for r in out), count=len(out)),
            indent=2,
        )
        + "\n"
    )
