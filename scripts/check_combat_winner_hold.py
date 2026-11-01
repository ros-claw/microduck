"""Continue a captured final state with zero-command neural standing.

This is a separate post-match audit, not extra match footage or a new result.
Eliminated robots keep their last servo targets as in the competition runtime.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.observations import snapshot
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.world import ASSETS, POLICIES, CombatBank, digest
from microduck_lab.sim.runtime import DuckRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument(
        "--policy",
        choices=["stand", "walk_candidate", "dash_candidate", "recovery_candidate"],
        default="stand",
    )
    a = p.parse_args()
    root, out = Path(a.run), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["capture"] or manifest["result"]["kind"] != "WIN":
        raise ValueError("Need an actual captured win")
    for f in ("scene.mjb", "states.npy"):
        if digest(root / f) != manifest["hashes"][f]:
            raise ValueError("Changed physical source")
    config = V2Config(**manifest["config"])
    if not 1 <= a.seconds <= 10:
        raise ValueError("Hold audit must last 1..10 seconds")
    m = mujoco.MjModel.from_binary_path(str(root / "scene.mjb"))
    d = mujoco.MjData(m)
    bank = CombatBank(
        {n: str(ASSETS / "microduck/policies" / f) for n, f in POLICIES.items()}
    )
    for name, path in [
        ("walk_candidate", config.walk_motor_path),
        ("dash_candidate", config.candidate_motor_path),
        ("recovery_candidate", config.recovery_motor_path),
    ]:
        if path:
            bank.paths[name] = path
    ducks = {
        n: DuckRuntime(m, d, bank, prefix=n + "/", name=n) for n in manifest["spawns"]
    }
    states = np.load(root / "states.npy", mmap_mode="r")
    mujoco.mj_setState(m, d, states[manifest["steps"] - 1], manifest["state_spec"])
    # Integration captures precede mj_step; perform the final recorded step.
    mujoco.mj_step(m, d)
    for duck in ducks.values():
        duck.last_action = (d.ctrl[duck.act_ids] - duck.default_pose).astype(np.float32)
    eliminated = json.loads((root / "referee.json").read_text())["eliminated"]
    alive = [n for n in ducks if n not in eliminated]
    audit = ContactAudit(m, ducks)
    model_hash = [
        v.tobytes()
        for v in (
            m.body_mass,
            m.body_inertia,
            m.actuator_forcerange,
            m.geom_contype,
            m.geom_conaffinity,
        )
    ]
    body, torque, finite, external = [], 0.0, True, True
    for step in range(round(a.seconds / m.opt.timestep)):
        if step % round(0.02 / m.opt.timestep) == 0:
            public = snapshot(ducks, audit, config)
            body.append(
                {
                    n: dict(
                        xy=list(public[n].xy),
                        z=public[n].z,
                        upright=public[n].upright,
                        speed=public[n].speed,
                        foot_support=public[n].foot_support,
                        margin_m=public[n].boundary_distance,
                    )
                    for n in alive
                }
            )
            for n in alive:
                ducks[n].active_policy = a.policy
                ducks[n].set_command()
                ducks[n].step()
        mujoco.mj_step(m, d)
        audit.read(m, d, step, m.opt.timestep, {})
        finite &= bool(np.isfinite(np.r_[d.qpos, d.qvel]).all())
        external &= not (np.any(d.xfrc_applied) or np.any(d.qfrc_applied))
        torque = max(torque, float(np.abs(d.actuator_force).max()))
    contacts = audit.summary()
    stable = all(
        row[n]["upright"] > 0.94 and row[n]["margin_m"] > 0.03 and row[n]["z"] > 0.05
        for row in body
        for n in alive
    )
    tail = body[-25:]
    stopped = all(
        max(row[n]["speed"] for row in tail) < 0.16
        and sum(row[n]["foot_support"] for row in tail) / len(tail) > 0.8
        for n in alive
    )
    unchanged = model_hash == [
        v.tobytes()
        for v in (
            m.body_mass,
            m.body_inertia,
            m.actuator_forcerange,
            m.geom_contype,
            m.geom_conaffinity,
        )
    ]
    write_json(
        out,
        dict(
            schema="duckfu.winner-hold.v1",
            source_manifest_sha256=digest(root / "manifest.json"),
            duration_s=a.seconds,
            alive=alive,
            zero_command_policy=a.policy,
            policy_sha256=digest(bank.paths[a.policy]),
            root_state_writes_after_initial_restore=False,
            stable=stable,
            stopped=stopped,
            finite=finite,
            external_forces_zero=external,
            model_unchanged=unchanged,
            torque_max_Nm=torque,
            solver_warnings=[int(w.number) for w in d.warning],
            contacts=contacts,
            body=body,
        ),
    )
    print(
        json.dumps(
            dict(
                alive=alive,
                stable=stable,
                stopped=stopped,
                finite=finite,
                external_forces_zero=external,
                model_unchanged=unchanged,
                torque_max_Nm=torque,
            )
        )
    )


if __name__ == "__main__":
    main()
