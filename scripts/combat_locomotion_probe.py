"""Matched-model command/stop qualification, including the failed orbit corner."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import mujoco
import numpy as np
from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.world import build, digest
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.observations import snapshot


def trial(job):
    motor, command, seed = job
    cfg = Config(defense="absent", radius_m=2.0)
    phi = Physics(native_ccd=True, dt=0.000125)
    m, d, ducks, _ = build(cfg, phi, seed)
    a, b = ducks.values()
    a.bank.paths["walk_candidate"] = motor
    audit = ContactAudit(m, ducks)
    rows = []
    jvmax = 0.0
    for k in range(round(6.0 / phi.dt)):
        now = k * phi.dt
        if k % phi.substeps == 0:
            public = snapshot(ducks, audit, cfg)
            me = public[a.name]
            a.active_policy = "walk_candidate"
            a.set_command(twist=command if 1.0 <= now < 4.0 else (0.0, 0.0, 0.0))
            a.step()
            b.step()
            rows.append(
                [
                    now,
                    *me.xy,
                    me.upright,
                    me.speed,
                    me.yaw,
                    float(me.foot_support),
                    me.neck_pitch_delta,
                    me.head_pitch_delta,
                ]
            )
        mujoco.mj_step(m, d)
        audit.read(m, d, k, phi.dt, {})
        jvmax = max(jvmax, float(np.max(np.abs(d.qvel[a.joint_qvel_idx]))))
    v = np.array(rows)
    tail = v[-15:]
    yaw = np.unwrap(v[:, 5])
    report = audit.summary()
    return dict(
        command=command,
        seed=seed,
        final_stable=bool(
            ((tail[:, 3] > 0.94) & (tail[:, 4] < 0.16) & (tail[:, 6] > 0.5)).all()
        ),
        displacement_during_drive_m=(v[200, 1:3] - v[50, 1:3]).tolist(),
        yaw_change_rad=float(yaw[200] - yaw[50]),
        head_deviation_max_rad=float(np.abs(v[:, 7:9]).max()),
        joint_speed_max_rad_s=jvmax,
        minimum_upright=float(v[:, 3].min()),
        contacts=report,
        body=v.tolist(),
        warnings=[int(w.number) for w in d.warning],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motor", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = Path(a.out)
    if path.exists():
        raise FileExistsError(path)
    commands = [
        (x, y, w)
        for x, y, w in [
            (-0.3, 0.0, 0.0),
            (0.28, 0.0, 0.0),
            (0.0, 0.3, 0.0),
            (0.0, -0.3, 0.0),
            (-0.15, -0.3, -1.0),
            (-0.15, 0.3, 1.0),
            (-0.15, -0.3, 1.0),
            (-0.15, 0.3, -1.0),
            (0.28, -0.3, -1.0),
            (0.28, 0.3, 1.0),
            (0.0, -0.3, 1.0),
            (0.0, 0.3, -1.0),
        ]
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(
            pool.map(trial, [(a.motor, c, 22001 + i) for i, c in enumerate(commands)])
        )
    report = dict(
        motor_path=a.motor, motor_sha256=digest(a.motor), cases=len(rows), rows=rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print([{k: v for k, v in r.items() if k not in ("contacts", "body")} for r in rows])


if __name__ == "__main__":
    main()
