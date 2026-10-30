"""R0 reset-pose recovery battery; R1/R2 require actual impact trajectories.

Initialization quaternion/height is written exactly once, before stepping.
The height comes from actual collision geometry, never a penetrating spawn.
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.config import Config, Physics, strict_load
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.observations import snapshot
from microduck_lab.combat.recovery import RecoveryGuard
from microduck_lab.combat.world import build


def lowest_native(m, d, prefix):
    lowest = math.inf
    for i in range(m.ngeom):
        # Most original body colliders are unnamed; ownership comes from body.
        if not m.body(int(m.geom_bodyid[i])).name.startswith(prefix) or not (
            m.geom_contype[i] & 4
        ):
            continue
        pos, rotation = d.geom_xpos[i], d.geom_xmat[i].reshape(3, 3)
        size, typ = m.geom_size[i], m.geom_type[i]
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[i]
            start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            low = (m.mesh_vert[start : start + count] @ rotation[2]).min() + pos[2]
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
            low = pos[2] - np.abs(rotation[2]) @ size
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            low = pos[2] - size[0]
        elif typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            low = pos[2] - size[0] - size[1] * abs(rotation[2, 2])
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            nz = rotation[2, 2]
            low = (
                pos[2] - size[1] * abs(nz) - size[0] * math.sqrt(max(0.0, 1 - nz * nz))
            )
        else:
            raise ValueError(f"Unmeasured geometry {typ}")
        lowest = min(lowest, float(low))
    return lowest


def trial(job):
    seed, side, group, physics_path = job
    cfg = Config(
        defense="absent",
        arena="fixed_disc" if group == "R3" else "lab",
        radius_m=0.75 if group == "R3" else 1.5,
    )
    physics = strict_load(physics_path, Physics) if physics_path else Physics()
    m, d, ducks, _ = build(cfg, physics, seed)
    duck = ducks["lavender"]
    rng = np.random.default_rng(seed)
    axis = np.array([(1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)][side], dtype=float)
    angle = (1 if side % 2 == 0 else -1) * (math.pi / 2 + rng.uniform(-0.15, 0.15))
    q = duck.trunk_qpos_adr
    friction = (0.9 if seed % 2 else 1.1) if group == "R3" else 1.0
    if group == "R3":
        # Explicit INITIALIZATION-only robustness perturbation, never used as
        # a per-player or runtime advantage in the competition profile.
        m.geom_friction[m.geom("stage").id, 0] = friction
    position = (
        (
            0.45 * math.cos(seed % 4 * math.pi / 2),
            0.45 * math.sin(seed % 4 * math.pi / 2),
        )
        if group == "R3"
        else (0.0, 0.0)
    )
    d.qpos[q : q + 3] = [*position, 0.4]
    d.qpos[q + 3 : q + 7] = [math.cos(angle / 2), *(axis * math.sin(angle / 2))]
    mujoco.mj_forward(m, d)
    d.qpos[q + 2] += 0.002 - lowest_native(m, d, duck.prefix)
    mujoco.mj_forward(m, d)
    if any(
        c.dist < -1e-7 and audit_geom in (c.geom1, c.geom2)
        for audit_geom in (m.geom("stage").id,)
        for c in d.contact
    ):
        raise ValueError("Invalid penetrating reset: do not evaluate this pose")
    initial = d.qpos[q : q + 7].tolist()
    audit, guard = ContactAudit(m, ducks), RecoveryGuard(duck.name)
    completed = None
    body = []
    for k in range(round(10 / physics.dt)):
        now = k * physics.dt
        if k % physics.substeps == 0:
            public = snapshot(ducks, audit, cfg)
            selected = guard.choose(now, public[duck.name])
            duck.active_policy = selected or "stand"
            duck.set_command(twist=(0, 0, 0), head=(0, 0, 0, 0))
            duck.step()
            other = ducks["cream"]
            other.active_policy = "stand"
            other.step()
            body.append(
                [
                    now,
                    public[duck.name].upright,
                    public[duck.name].z,
                    public[duck.name].speed,
                    public[duck.name].foot_support,
                    duck.active_policy,
                ]
            )
            if completed is None and any(
                e["state"] == "RECOVERED" for e in guard.events
            ):
                completed = now
        mujoco.mj_step(m, d)
        audit.read(m, d, k, physics.dt, {})
    return dict(
        seed=seed,
        side=side,
        group=group,
        initial_root_pose=initial,
        stage_friction=friction,
        reset_boundary="before first physics step",
        duration_s=10,
        completed_s=completed,
        within_3s=completed is not None and completed <= 3,
        events=guard.events,
        body=body,
        contacts=audit.summary(),
        finite=bool(np.isfinite(d.qpos).all()),
        warnings=[int(w.number) for w in d.warning],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--seed", type=int, default=6101)
    p.add_argument("--group", choices=["R0", "R3"], default="R0")
    p.add_argument("--physics-profile", default="")
    a = p.parse_args()
    if not 4 <= a.count <= 80:
        p.error("Use 4–80 independent reset poses")
    path = Path(a.out)
    if path.exists():
        raise FileExistsError(path)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(
            pool.map(
                trial,
                [
                    (a.seed + i, i % 4, a.group, a.physics_profile)
                    for i in range(a.count)
                ],
            )
        )
    times = [r["completed_s"] for r in rows if r["completed_s"] is not None]
    summary = dict(
        group=a.group,
        independent_states=a.count,
        success=sum(r["within_3s"] for r in rows),
        completed=len(times),
        unfinished=a.count - len(times),
        completed_only_p50_p90_s=np.percentile(times, [50, 90]).tolist()
        if times
        else None,
        rows=rows,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
