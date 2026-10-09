"""Controlled counterfactuals from real recorded falls, not new online matches.

Opponent controls and public floor releases follow the source recording while
the target's motors close the loop. Other bodies still respond to new contacts.
Last action is reconstructed from recorded position targets (float rounding).
No root writes after the single initialized state; no extra forces.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.arena.episode import STATE, sha
from microduck_lab.arena.multiplayer import ContactGraph
from microduck_lab.arena.tiles import Tile
from microduck_lab.arena.world import ASSETS
from microduck_lab.sim.runtime import DuckRuntime, PolicyBank


def trials(run, cases, strategies=("stand", "once", "retry")):
    run = Path(run)
    audit = json.loads((run/"audit.json").read_text())
    m = mujoco.MjModel.from_binary_path(str(run/"scene.mjb"))
    data = np.load(run/"trajectory.npz")
    states, ctrl = data["states"], data["ctrl"]
    rows = []
    bank = PolicyBank({"stand": str(ASSETS/"microduck/policies/alpha_stand.onnx"), "sitstand": str(ASSETS/"microduck/policies/alpha_sitstand.onnx")})
    for body, when in cases:
        start = round(when/.00025)
        for strategy in strategies:
            d = mujoco.MjData(m)
            mujoco.mj_setState(m, d, states[start], STATE)
            mujoco.mj_forward(m, d)
            ducks = {n: DuckRuntime(m, d, bank, prefix=n+"/", name=n) for n in audit["characters"]}
            target = ducks[body]
            target.last_action = (d.ctrl[target.act_ids]-target.default_pose).astype(np.float32)
            tiles = {i: Tile(i, m.equality(f"tile_{i}_support").id,
                             int(m.jnt_qposadr[m.joint(f"tile_{i}_free").id]),
                             int(m.jnt_dofadr[m.joint(f"tile_{i}_free").id])) for i in range(25)}
            graph = ContactGraph(m, ducks, tiles)
            releases = [(event["time"], tiles[event["tile"]].eq_id) for event in audit["events"]
                        if event["state"] == "RELEASED"]
            stable = 0.
            completion = None
            maxdepth = 0.
            skill_frames = 0
            for tick in range(300):
                elapsed = tick*.02
                reference = min(start+tick*80, len(ctrl)-1)
                d.ctrl[:] = ctrl[reference]
                burst = .8-1e-9 <= elapsed < 1.6-1e-9
                if strategy == "phase_aware":
                    burst = 2.0-1e-9 <= elapsed < 2.8-1e-9
                if strategy == "retry":
                    burst = elapsed >= .8-1e-9 and ((elapsed-.8+1e-9) % 2.4) < .8-1e-9
                target.active_policy = "sitstand" if completion is None and strategy != "stand" and burst else "stand"
                skill_frames += target.active_policy == "sitstand"
                target.set_command()
                target.step()
                for substep in range(80):
                    mujoco.mj_step(m, d)
                    absolute = when+elapsed+(substep+1)*.00025
                    for release_time, eq_id in releases:
                        if release_time <= absolute+1e-9:
                            d.eq_active[eq_id] = False
                    maxdepth = max(maxdepth, max((-c.dist for c in d.contact), default=0.))
                _, support, _ = graph.read(m, d, tiles)
                good = d.xmat[target.trunk_body_id, 8] > .95 and support[body] and np.linalg.norm(target.trunk_linvel()) < .12
                stable = stable+.02 if good else 0.
                if completion is None and stable >= .15:
                    completion = round(elapsed+.02, 3)
                if target.trunk_pos()[2] < -.3:
                    break
            row = dict(source=str(run), source_audit_sha256=sha(run/"audit.json"), body_id=body, initialized_sim_time_s=when,
                       strategy=strategy, completion_s=completion, skill_frames=skill_frames, penetration_max_m=maxdepth,
                       final_z=float(target.trunk_pos()[2]), warnings=d.warning.number.tolist(),
                       experiment="counterfactual fixed opponent controls and hazard releases, reconstructed last action; not online Jev")
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = trials("artifacts/jev-rumble/final-live-61204", [("graphite", 16.82), ("graphite", 29.50), ("sky", 42.38)])
    Path(a.out).write_text(json.dumps(rows, indent=2)+"\n")
