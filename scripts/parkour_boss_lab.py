"""Isolate boss/prop dynamics; no duck on the boss path in this component test."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json
from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.hazards import Hazard, HazardDriver


def run():
    hazards = [
        Hazard("bar", "push_bar", 0.9),
        Hazard("box", "crate", 1.5, z=0.65),
        Hazard("boss", "boulder", -0.7, z=0.25, speed=0.5),
    ]
    m, d, r, tr = build_world(hazards=hazards, rails=True)
    driver = HazardDriver(hazards)
    d.qpos[r.trunk_qpos_adr] = 8
    mujoco.mj_forward(m, d)
    samples = []
    contacts = {}
    max_pen = 0
    for i in range(600):
        r.step()
        driver.step(m, d, 2.0)
        for _ in range(100):
            mujoco.mj_step(m, d)
            for ci, c in enumerate(d.contact):
                names = tuple(sorted([m.geom(c.geom1).name, m.geom(c.geom2).name]))
                if "boss/geom" in names and (
                    "bar/geom" in names or "box/geom" in names
                ):
                    contacts.setdefault("|".join(names), float(d.time))
                    max_pen = max(max_pen, -float(c.dist))
        if i % 5 == 0:
            samples.append(
                dict(
                    t=float(d.time),
                    boss=d.xpos[m.body("boss").id].tolist(),
                    bar_angle=float(d.qpos[m.jnt_qposadr[m.joint("bar/joint").id]]),
                    box=d.xpos[m.body("box").id].tolist(),
                )
            )
    return dict(
        samples=samples,
        contacts=contacts,
        events=driver.events,
        warnings=d.warning.number.tolist(),
        max_prop_penetration=max_pen,
        passed=bool(
            samples[-1]["boss"][0] > 2.0
            and min(s["boss"][2] for s in samples) > 0.20
            and max(abs(s["boss"][1]) for s in samples) < tr.width / 2
            and len(contacts) == 2
            and not d.warning.number.any()
        ),
    )


if __name__ == "__main__":
    out = run()
    (ROOT / "artifacts/neon-escape-v2/boss-prop-final.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print({k: v for k, v in out.items() if k != "samples"})
    print(out["samples"][-1])
