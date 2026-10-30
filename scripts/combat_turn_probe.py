"""Measured native-policy turn commands; no root yaw or motor overrides."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import mujoco
import numpy as np

from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.world import build


def trial(command):
    m, d, ducks, _ = build(Config(defense="absent"), Physics(), 8101)
    a, b = ducks.values()
    samples = []
    for step in range(20000):
        now = step * 0.00025
        if step % 80 == 0:
            a.active_policy = "walk"
            a.set_command(twist=command if 1 <= now < 4 else (0.0, 0.0, 0.0))
            a.step()
            b.step()
            samples.append(
                [now, *a.trunk_pos()[:2], a.trunk_yaw(), -a.projected_gravity()[2]]
            )
        mujoco.mj_step(m, d)
    v = np.array(samples)
    yaw = np.unwrap(v[:, 3])
    return dict(
        command=command,
        yaw_change_rad=float(yaw[200] - yaw[50]),
        path_m=float(np.linalg.norm(np.diff(v[50:201, 1:3], axis=0), axis=1).sum()),
        minimum_upright=float(v[:, 4].min()),
        solver_warnings=[int(w.number) for w in d.warning],
        body=v.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = Path(a.out)
    if path.exists():
        raise FileExistsError(path)
    commands = [
        (vx, vy, 1.0)
        for vx, vy in itertools.product((0.0, 0.15, 0.28, 0.4), (-0.3, 0.0, 0.3))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(trial, commands))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print([{k: v for k, v in r.items() if k != "body"} for r in rows])


if __name__ == "__main__":
    main()
