"""Fixed-input rim-contact benchmark: libccd/native CCD and dt/dt2.

This resets from an actual recorded pre-impact integration state, uses the
same motor tape, and holds its last target beyond the rejected run. It is a
local solver experiment, never a resumed competition or winner claim.
"""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.world import CombatBank, digest
from microduck_lab.sim.runtime import DuckRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--category", choices=["duck_stage", "duck_duck", "self"], default="duck_stage"
    )
    a = p.parse_args()
    directory, out = Path(a.run), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "contacts_summary.json").read_text())
    peak = summary["categories"][a.category]["maximum"]["step"]
    dt = manifest["physics"]["dt"]
    start = max(0, peak - round(0.1 / dt))
    states = np.load(directory / "states.npy", mmap_mode="r")
    tape = np.load(directory / "controls.npz")["ctrl"]
    rows = []
    for native in (False, True):
        for factor in (1, 2, 4):
            m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
            d = mujoco.MjData(m)
            m.opt.timestep = dt / factor
            if native:
                m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            else:
                m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            mujoco.mj_setState(m, d, states[start], manifest["state_spec"])
            mujoco.mj_forward(m, d)
            ducks = {
                n: DuckRuntime(m, d, CombatBank({}), prefix=n + "/", name=n)
                for n in manifest["spawns"]
            }
            audit = ContactAudit(m, ducks)
            stage_impulse = 0.0
            for k in range(round(0.3 / m.opt.timestep)):
                original = start + k // factor
                d.ctrl[:] = tape[min(len(tape) - 1, original // round(0.02 / dt))]
                mujoco.mj_step(m, d)
                contacts = audit.read(m, d, k, m.opt.timestep, {})
                stage_impulse += sum(
                    max(0.0, r[3]) * m.opt.timestep
                    for r in contacts
                    if audit.stage in r[:2]
                )
            result = audit.summary()
            rows.append(
                dict(
                    native_ccd=native,
                    dt=m.opt.timestep,
                    stage_normal_impulse_Ns=stage_impulse,
                    maximum_penetration_m=result["penetration_max_m"],
                    categories=result["categories"],
                    finite=bool(np.isfinite(d.qpos).all()),
                    warnings=[int(w.number) for w in d.warning],
                    final_root_positions={
                        n: duck.trunk_pos().tolist() for n, duck in ducks.items()
                    },
                )
            )
    result = dict(
        schema="duckfu.rim-solver-probe.v1",
        source_manifest_sha256=digest(directory / "manifest.json"),
        start_step=start,
        source_peak_step=peak,
        source_category=a.category,
        duration_s=0.3,
        same_motor_tape=True,
        held_last_target_after_source_end=True,
        local_component_test_not_closed_loop_game=True,
        rows=rows,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        [
            {
                k: v
                for k, v in r.items()
                if k not in ("categories", "final_root_positions")
            }
            for r in rows
        ]
    )


if __name__ == "__main__":
    main()
