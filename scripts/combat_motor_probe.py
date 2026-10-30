"""Independent CPU contact-duration/head-command probe; no tactical coach.

These open-loop commands test a proposed motor task, not autonomous combat.
Only native motor inputs are written after the world's initialization.
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.observations import snapshot
from microduck_lab.combat.world import build


def trial(job):
    separation, drive, head, seed = job
    cfg, physics = Config(separation_m=separation, defense="stand"), Physics()
    m, d, ducks, _ = build(cfg, physics, seed)
    audit = ContactAudit(m, ducks)
    bodies, controls = [], []
    first = None
    initial = None
    for k in range(round((drive + 2.2) / physics.dt)):
        t = k * physics.dt
        if k % physics.substeps == 0:
            public = snapshot(ducks, audit, cfg)
            bodies.append(
                [[*b.xy, b.upright, b.speed, b.foot_support] for b in public.values()]
            )
            if initial is None and t >= 0.8:
                initial = np.array(public["cream"].xy)
            for name, duck in ducks.items():
                active = name == "lavender" and 0.8 <= t < drive + 0.8
                duck.active_policy = "walk" if name == "lavender" else "stand"
                duck.set_command(
                    twist=(0.4 if active else 0, 0, 0),
                    head=head if active else (0, 0, 0, 0),
                )
                duck.step()
            controls.append(d.ctrl.copy())
        mujoco.mj_step(m, d)
        audit.read(m, d, k, physics.dt, {})
        if first is None and audit.touching["lavender"]:
            first = t
    b = np.array(bodies)
    tail = b[-15:, 0]
    return dict(
        separation=separation,
        drive_s=drive,
        head=head,
        seed=seed,
        first_contact_s=first,
        target_effect_m=float(np.linalg.norm(b[40:, 1, :2] - initial, axis=1).max()),
        actor_stable=bool(
            ((tail[:, 2] > 0.94) & (tail[:, 3] < 0.16) & (tail[:, 4] > 0.5)).all()
        ),
        actor_min_upright=float(b[:, 0, 2].min()),
        audit=audit.summary(),
        body=b.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    jobs = [
        (s, t, h, 5101)
        for s, t, h in itertools.product(
            (0.32, 0.43),
            (1.25, 2.2),
            ((0, 0, 0, 0), (-0.3, -0.15, 0, 0), (0.25, -0.25, 0, 0)),
        )
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(trial, jobs))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print({k: v for k, v in row.items() if k not in ("body", "audit")}, flush=True)


if __name__ == "__main__":
    main()
