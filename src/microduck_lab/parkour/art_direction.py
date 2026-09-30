"""Renderer-only industrial set dressing. No collision or state changes."""

import numpy as np
import mujoco


def line(scene, a, b, color=(1.0, 0.42, 0.08, 1.0), width=2.0):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array(color),
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a), np.asarray(b)
    )
    g.emission = 0.65
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def box(scene, pos, size, color):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(size),
        np.asarray(pos),
        np.eye(3).ravel(),
        np.asarray(color),
    )
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def industrial(scene, m, d, report, camera):
    # Only the far side is dressed with tall scenery, keeping every physical
    # obstacle visible from the active camera. Heights are deterministic art.
    far_side = -1 if camera[1] > 0 else 1
    for i, x in enumerate(np.arange(-2.8, report["track"]["end"], 0.65)):
        y = far_side * (0.85 + 0.12 * (i % 3))
        h = 0.35 + 0.12 * (i % 4)
        box(scene, [x, y, h / 2], [0.10, 0.10, h / 2], [0.09, 0.12, 0.17, 1.0])
        line(
            scene,
            [x - 0.10, y - 0.105, 0.10],
            [x - 0.10, y - 0.105, h - 0.04],
            (0.1, 0.55, 0.75, 1.0),
            2.0,
        )
        for z in np.arange(0.12, h, 0.12):
            line(
                scene,
                [x - 0.075, y - 0.105, z],
                [x + 0.075, y - 0.105, z],
                (0.15, 0.35, 0.46, 1.0),
                1.0,
            )
    # Floor panels and edges stop at the true void; decorative lines never
    # cover the gap. White ticks help viewers perceive actual forward speed.
    width = report["track"]["width"]
    for g in range(m.ngeom):
        if not m.geom(g).name.startswith("floor/"):
            continue
        x, y, z = d.geom_xpos[g]
        sx, sy, sz = m.geom_size[g]
        for xx in np.arange(x - sx + 0.06, x + sx, 0.20):
            for side in (-1, 1):
                line(
                    scene,
                    [xx, side * (min(sy, width / 2) - 0.018), z + sz + 0.0006],
                    [xx + 0.05, side * (min(sy, width / 2) - 0.018), z + sz + 0.0006],
                    (0.55, 0.65, 0.68, 1.0),
                    1.0,
                )
    for near, far in report["level"]["gaps"]:
        for x, sign in [(near, -1), (far, 1)]:
            for y in np.arange(-width / 2 + 0.01, width / 2 - 0.035, 0.045):
                line(
                    scene,
                    [x + sign * 0.012, y, 0.001],
                    [x + sign * 0.05, y + 0.028, 0.001],
                    (1.0, 0.6, 0.05, 1.0),
                    3.0,
                )
    # The crate silhouette is exactly its physical 13 cm box. Lines are thin
    # cosmetic edges, attached to its measured rigid-body transform.
    center = d.body("crate").xpos
    R = d.body("crate").xmat.reshape(3, 3)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for a in (-1, 1):
            for b in (-1, 1):
                u = np.zeros(3)
                v = np.zeros(3)
                u[axis] = -0.0652
                v[axis] = 0.0652
                u[others] = v[others] = [a * 0.0652, b * 0.0652]
                line(scene, center + R @ u, center + R @ v, (1.0, 0.48, 0.04, 1.0), 2.0)
    if report.get("difficulty") == "arcade":
        # Purely visual arrows sit on the existing bowling platform.
        for x in [5.05, 5.25, 5.45]:
            line(
                scene, [x - 0.05, -0.035, 0.001], [x, 0, 0.001], (0.1, 0.95, 0.9, 1), 3
            )
            line(scene, [x, 0, 0.001], [x - 0.05, 0.035, 0.001], (0.1, 0.95, 0.9, 1), 3)
        for i, pin in enumerate(["pin0", "pin1", "pin2"]):
            down = any(
                e["type"] == "PIN_DOWN" and e["pin"] == pin and e["t"] <= d.time
                for e in report["events"]
            )
            color = (0.1, 0.95, 0.55, 1) if down else (0.9, 0.55, 0.08, 1)
            line(
                scene,
                [6.32, -0.075 + i * 0.075, 0.001],
                [6.37, -0.075 + i * 0.075, 0.001],
                color,
                7,
            )
