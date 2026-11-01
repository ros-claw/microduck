"""Extract real motor-boundary states, retaining momentum and action history.

Yaw/XY normalization is only for initialization of the single-robot training
task. Raw poses and velocities remain available for edge-aware CPU checks.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.world import CombatBank, digest
from microduck_lab.sim.runtime import DuckRuntime
from microduck_lab.combat.episode import write_json


def lowest_stage_collider(m, d, prefix):
    stage = m.geom("stage").id
    lowest = math.inf
    for g in range(m.ngeom):
        if not m.body(int(m.geom_bodyid[g])).name.startswith(prefix):
            continue
        if not (
            (m.geom_contype[g] & m.geom_conaffinity[stage])
            or (m.geom_contype[stage] & m.geom_conaffinity[g])
        ):
            continue
        pos, rotation, size, typ = (
            d.geom_xpos[g],
            d.geom_xmat[g].reshape(3, 3),
            m.geom_size[g],
            m.geom_type[g],
        )
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
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
            low = pos[2] - size[1] * abs(nz) - size[0] * math.sqrt(max(0, 1 - nz * nz))
        else:
            raise ValueError(f"Unmeasured native collider {typ}")
        lowest = min(lowest, float(low))
    return lowest


def extract(directory, split):
    directory = Path(directory)
    mfest = json.loads((directory / "manifest.json").read_text())
    if (
        not mfest["capture"]
        or mfest["result"]["kind"] == "NO_CONTEST_PHYSICS"
        or mfest["physical"]["penetration_max_m"] > 0.003
    ):
        raise ValueError("Require a valid physical source capture")
    for name in ("scene.mjb", "controls.npz", "states.npy"):
        if digest(directory / name) != mfest["hashes"][name]:
            raise ValueError("Changed source evidence")
    m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    d = mujoco.MjData(m)
    ducks = {
        n: DuckRuntime(m, d, CombatBank({}), prefix=n + "/", name=n)
        for n in mfest["spawns"]
    }
    states = np.load(directory / "states.npy", mmap_mode="r")
    controls = np.load(directory / "controls.npz")["ctrl"]
    events = [
        json.loads(s)
        for s in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    rows, metadata = [], []
    for number, event in enumerate(
        e for e in events if e["state"] == "RECOVERY_STARTED"
    ):
        actor, start = event["actor"], event["time_s"]
        duck = ducks[actor]
        for offset in (-0.16, 0.0, 0.12, 0.30, 0.60, 1.00, 1.60, 2.20):
            t = round((start + offset) * 50) / 50
            if not 0 < t < mfest["duration_s"] - 0.02:
                continue
            step = round(t / mfest["physics"]["dt"])
            mujoco.mj_setState(m, d, states[step], mfest["state_spec"])
            mujoco.mj_forward(m, d)
            raw = d.qpos[duck.trunk_qpos_adr : duck.trunk_qpos_adr + 7].copy()
            # mj_objectVelocity(BODY) is COM-centered. The training reset API
            # needs the free-joint/link origin velocity, which is qvel[:3].
            # Angular qvel is body-local; convert it to world orientation.
            native_velocity = d.qvel[duck.trunk_qvel_adr : duck.trunk_qvel_adr + 6]
            vel = np.r_[
                d.xmat[duck.trunk_body_id].reshape(3, 3) @ native_velocity[3:],
                native_velocity[:3],
            ]
            yaw = duck.trunk_yaw()
            rot = np.array(
                [
                    [math.cos(yaw), math.sin(yaw), 0],
                    [-math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )
            inverse_yaw = np.array([math.cos(yaw / 2), 0, 0, -math.sin(yaw / 2)])
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, inverse_yaw, raw[3:7])
            root = np.r_[0, 0, raw[2], quat, rot @ vel[3:], rot @ vel[:3]]
            previous = (
                controls[max(0, round(t * 50) - 1), duck.act_ids] - duck.default_pose
            )
            joints, speed = (
                d.qpos[duck.joint_qpos_idx].copy(),
                d.qvel[duck.joint_qvel_idx].copy(),
            )
            lowest = lowest_stage_collider(m, d, duck.prefix)
            info = dict(
                split=split,
                source_run=str(directory.resolve()),
                source_manifest_sha256=digest(directory / "manifest.json"),
                seed=mfest["seed"],
                swap=mfest["config"]["swap"],
                actor=actor,
                episode=number,
                recovery_start_s=start,
                time_s=t,
                offset_s=offset,
                source_step=step,
                upright=float(-duck.projected_gravity()[2]),
                lowest_stage_collider_m=lowest,
                joint_names=[
                    m.joint(int(m.actuator_trnid[a, 0])).name.split("/")[-1]
                    for a in duck.act_ids
                ],
                boundary_margin_m=float(
                    mfest["config"]["radius_m"] * math.cos(math.pi / 8)
                    - max(
                        np.array([math.cos(a), math.sin(a)]) @ raw[:2]
                        for a in np.arange(8) * math.pi / 4 + math.pi / 8
                    )
                ),
            )
            # Training cannot silently give support to a body hanging below a
            # ledge. Keep the record, but exclude it from the flat-floor pool.
            info["flat_floor_eligible"] = bool(
                lowest >= -0.0015
                and raw[2] > 0.025
                and info["boundary_margin_m"] > 0.12
            )
            rows.append(
                dict(
                    root=root,
                    joints=joints,
                    joint_velocity=speed,
                    last_action=previous,
                    raw_root=raw,
                    raw_world_velocity=np.r_[vel[3:], vel[:3]],
                )
            )
            metadata.append(info)
    return rows, metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, action="append")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["development", "validation"], required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    rows, meta = [], []
    for path in a.run:
        batch, info = extract(path, a.split)
        rows.extend(batch)
        meta.extend(info)
    if not rows:
        raise ValueError("No actual recovery states")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{k: np.array([r[k] for r in rows]) for k in rows[0]},
        eligible=np.array([m["flat_floor_eligible"] for m in meta]),
    )
    write_json(
        out.with_suffix(".json"),
        dict(
            schema="duckfu.recorded-recovery-states.v1",
            split=a.split,
            states_sha256=digest(out),
            states=len(rows),
            flat_floor_eligible=sum(m["flat_floor_eligible"] for m in meta),
            normalization="Yaw and XY changed only at the training reset; tilt, height, world-frame velocities in the yaw frame, joints and previous executed action retained.",
            rows=meta,
        ),
    )
    print(
        json.dumps(
            dict(
                states=len(rows),
                eligible=sum(m["flat_floor_eligible"] for m in meta),
                output=str(out),
            )
        )
    )
