"""Exact-input 5 kHz rod clearance; includes conservative full-visual proxies."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, hashlib
from pathlib import Path
import mujoco, numpy as np
from microduck_lab.parkour.evidence import verify_capture


def audit(source, start=12.0, stop=16.5, impact_time=None):
    source = Path(source)
    report = json.loads((source / "audit.json").read_text())
    verify_capture(source, report["capture"])
    m = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    d = mujoco.MjData(m)
    u = dict(np.load(source / "inputs.npz"))
    d.qpos[:] = u["initial_qpos"]
    d.qvel[:] = u["initial_qvel"]
    mujoco.mj_forward(m, d)
    rod = m.geom("sweeper/geom").id
    proxies = [
        g
        for g in range(m.ngeom)
        if m.geom(g).name.startswith("duck/")
        and m.geom(g).name.endswith("/hazard_proxy")
    ]
    nearest = dict(distance_m=1e3)
    contact_impulse = 0.0
    contacts = 0
    depth = 0.0
    peak_force = 0.0
    peak_t = None
    peak_geom = None
    f = np.zeros(6)
    segment = np.zeros(6)
    samples = []
    dense = []
    for i, t in enumerate(u["time"]):
        if t >= stop:
            break
        d.ctrl[:] = u["ctrl"][i]
        d.eq_active[:] = u["eq_active"][i]
        d.xfrc_applied[:] = u["xfrc_applied"][i]
        for j in range(100):
            # mj_step leaves derived geometry/contact data at the input state;
            # save that state, rather than the post-integration qpos, for the
            # exact force-bearing render frame.
            save_dense = (
                impact_time is not None and abs(float(d.time) - impact_time) < 0.05
            )
            if save_dense:
                dense_state = (
                    float(d.time),
                    d.qpos.copy(),
                    d.qvel.copy(),
                    d.ctrl.copy(),
                    d.eq_active.copy(),
                )
            mujoco.mj_step(m, d)
            if not start <= d.time <= stop:
                continue
            if save_dense:
                dense.append(dense_state)
            if i % 10 == 0 and j == 0:
                samples.append(
                    dict(t=float(d.time), duck=d.body("duck/trunk_base").xpos.tolist())
                )
            for g in proxies:
                distance = mujoco.mj_geomDistance(m, d, rod, g, 0.25, segment)
                if distance < nearest["distance_m"]:
                    nearest = dict(
                        t=float(d.time),
                        distance_m=float(distance),
                        geom=m.geom(g).name,
                        closest_points=segment.tolist(),
                    )
            for ci, c in enumerate(d.contact):
                if rod not in [c.geom1, c.geom2]:
                    continue
                other = c.geom2 if c.geom1 == rod else c.geom1
                if not m.body(m.geom_bodyid[other]).name.startswith("duck/"):
                    continue
                mujoco.mj_contactForce(m, d, ci, f)
                if f[0] > 1e-7:
                    contacts += 1
                    contact_impulse += float(f[0]) * m.opt.timestep
                    depth = max(depth, max(0.0, -float(c.dist)))
                    if f[0] > peak_force:
                        peak_force = float(f[0])
                        peak_t = float(d.time)
                        peak_geom = m.geom(other).name
    if dense:
        np.savez_compressed(
            source / "sweeper-highrate.npz",
            **{
                name: np.asarray([x[k] for x in dense])
                for k, name in enumerate(["time", "qpos", "qvel", "ctrl", "eq_active"])
            },
        )
    out = dict(
        peak_force_N=peak_force,
        peak_force_time=peak_t,
        peak_state_time=peak_t - m.opt.timestep if peak_t is not None else None,
        timestamp_convention="Force and nearest-distance timestamps denote step end; derived geometry and dense contact-frame state correspond to step start (end minus 0.2 ms).",
        peak_force_geom=peak_geom,
        capture=report["capture"],
        window_s=[start, stop],
        nearest=nearest,
        force_samples=contacts,
        normal_impulse_Ns=contact_impulse,
        max_contact_penetration_m=depth,
        clear_pass=nearest["distance_m"] >= 0 and contacts == 0,
        passed=nearest["distance_m"] > -0.0015
        and depth < 0.0015
        and not d.warning.number.any(),
        method="Initialize once; replay saved inputs; signed capsule–conservative visual-enclosing body boxes at every 0.2 ms physics step. No resampling or renderer interpolation.",
        route_samples=samples,
    )
    if dense:
        out["highrate_sha256"] = hashlib.sha256(
            (source / "sweeper-highrate.npz").read_bytes()
        ).hexdigest()
    (source / "sweeper-clearance.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "route_samples"}, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--impact-time", type=float)
    a = p.parse_args()
    audit(a.source, impact_time=a.impact_time)
