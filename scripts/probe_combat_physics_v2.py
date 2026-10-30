"""Local fixed-input dt/dt2 checks and first closed-loop control divergence.

Resets once from a recorded physical state, then integrates normally. No pose
overwrite, resumed-game winner, or hardware convergence claim is made.
"""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from microduck_lab.combat.world import CombatBank, digest
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.sim.runtime import DuckRuntime
from microduck_lab.combat.episode import write_json


def fixed_window(directory, time_s, duration_s=0.4):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["capture"]:
        raise ValueError("Full capture required")
    for name in ("states.npy", "scene.mjb", "controls.npz"):
        if digest(directory / name) != manifest["hashes"][name]:
            raise ValueError("Source evidence changed")
    states = np.load(directory / "states.npy", mmap_mode="r")
    tape = np.load(directory / "controls.npz")["ctrl"]
    dt = manifest["physics"]["dt"]
    start = max(0, round((time_s - 0.1) / dt))
    count = min(round(duration_s / dt), manifest["steps"] - start)
    rows = []
    for factor in (1, 2):
        m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
        m.opt.timestep = dt / factor
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, states[start], manifest["state_spec"])
        mujoco.mj_forward(m, d)
        ducks = {
            n: DuckRuntime(m, d, CombatBank({}), prefix=n + "/", name=n)
            for n in manifest["spawns"]
        }
        audit = ContactAudit(m, ducks)
        stage_impulse = 0.0
        support = {n: 0 for n in ducks}
        for k in range(count * factor):
            d.ctrl[:] = tape[(start + k // factor) // round(0.02 / dt)]
            mujoco.mj_step(m, d)
            contacts = audit.read(m, d, k, m.opt.timestep, {})
            stage_impulse += sum(
                max(0.0, r[3]) * m.opt.timestep
                for r in contacts
                if audit.stage in r[:2]
            )
            for n in support:
                support[n] += int(audit.support[n])
        summary = audit.summary()
        rows.append(
            dict(
                dt=m.opt.timestep,
                window_s=count * dt,
                duck_impulse_Ns=sum(
                    e["summed_normal_impulse_Ns"] for e in summary["episodes"]
                ),
                stage_impulse_Ns=stage_impulse,
                physical_numerical_gate=summary["gate_proposal_pass"],
                maximum_penetration_m=summary["penetration_max_m"],
                categories=summary["categories"],
                pairs=[
                    dict(
                        pair=e["actor_pair"],
                        impulse_Ns=e["summed_normal_impulse_Ns"],
                        peak_N=e["peak_force_N"],
                        duration_s=e["duration_s"],
                    )
                    for e in summary["episodes"]
                ],
                final={
                    n: dict(
                        position_m=duck.trunk_pos().tolist(),
                        upright=float(-duck.projected_gravity()[2]),
                        foot_support_fraction=support[n] / (count * factor),
                    )
                    for n, duck in ducks.items()
                },
                finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
                warnings=[int(w.number) for w in d.warning],
            )
        )
    a, b = rows
    return dict(
        source=str(directory.resolve()),
        source_manifest_sha256=digest(directory / "manifest.json"),
        source_step=start,
        source_event_time_s=time_s,
        fixed_motor_targets=True,
        motor_hz=50,
        rows=rows,
        root_difference_max_m=max(
            float(
                np.linalg.norm(
                    np.array(a["final"][n]["position_m"]) - b["final"][n]["position_m"]
                )
            )
            for n in a["final"]
        ),
        duck_impulse_relative_difference=abs(
            a["duck_impulse_Ns"] - b["duck_impulse_Ns"]
        )
        / max(1e-12, b["duck_impulse_Ns"]),
        note="Local fixed-input component check, not an invariant closed-loop winner or proof of visual/hardware exactness.",
    )


def divergence(first, second):
    first, second = Path(first), Path(second)
    a, b = (np.load(p / "controls.npz") for p in (first, second))
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (first, second)]
    n = min(len(a["ctrl"]), len(b["ctrl"]))
    initial = [np.load(p / "initial_state.npz")["state"] for p in (first, second)]
    difference = np.max(np.abs(a["ctrl"][:n] - b["ctrl"][:n]), axis=1)
    phase = np.any(a["phases"][:n] != b["phases"][:n], axis=1)
    body = np.max(np.abs(a["body"][:n, :, :3] - b["body"][:n, :, :3]), axis=(1, 2))
    return dict(
        first=str(first.resolve()),
        second=str(second.resolve()),
        initial_state_identical=bool(np.array_equal(*initial)),
        thresholds={
            str(t): float(np.flatnonzero(difference > t)[0] / 50)
            if np.any(difference > t)
            else None
            for t in [0.0, 0.0001, 0.01, 0.1]
        },
        first_phase_divergence_s=float(np.flatnonzero(phase)[0] / 50)
        if np.any(phase)
        else None,
        compared_ticks=n,
        max_position_component_difference_m=float(body.max()),
        results=[m["result"] for m in manifests],
        interpretation="Differences are diagnostics; neither winner is declared physical ground truth.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--time", type=float)
    p.add_argument("--compare-run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    result = dict(schema="duckfu.v2-physics-probe.v1")
    if a.time is not None:
        result["fixed_window"] = fixed_window(a.run, a.time)
    if a.compare_run:
        result["closed_loop"] = divergence(a.run, a.compare_run)
    if len(result) == 1:
        p.error("Provide --time and/or --compare-run")
    write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
