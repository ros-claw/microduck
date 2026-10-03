"""Replay a verified record, then physically simulate a new finish flourish.

The replay prefix retains historical Jev decisions. The suffix is explicitly
scripted skill scheduling, never a new unseen trial or a new live API run.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, hashlib, json, shutil, tarfile, time
from pathlib import Path
from dataclasses import asdict
import mujoco, numpy as np
from microduck_lab.sim.runtime import DuckRuntime, DEFAULT_POSE
from microduck_lab.parkour.policies import ParkourPolicyBank
from microduck_lab.parkour.arcade import ArcadeDriver
from microduck_lab.parkour.hazards import Hazard
from microduck_lab.parkour.audit import Audit, PropAudit
from microduck_lab.parkour.victory import VictoryDance
from microduck_lab.parkour.evidence import verify_capture

ROOT = Path(__file__).resolve().parents[1]


def run(source, output, branch_time=25.24):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Evidence must use a fresh directory")
    parent = json.loads((source / "audit.json").read_text())
    verify_capture(source, parent["capture"])
    if not parent["component_course_passed"]:
        raise ValueError("Qualified parent escape required")
    if any(s["t"] > branch_time for s in parent["skills"] if s["stage"] == "EXIT_ROLL"):
        raise ValueError("Branch must follow completed exit roll")
    output.mkdir(parents=True)
    files = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "src/microduck_lab/parkour").glob("*.json"))
        + [Path(__file__).resolve()]
    )
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    with tarfile.open(output / "source.tar.gz", "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    shutil.copy2(source / "source.tar.gz", output / "parent-source.tar.gz")
    shutil.copy2(source / "scene.mjb", output / "scene.mjb")
    m = mujoco.MjModel.from_binary_path(str(output / "scene.mjb"))
    d = mujoco.MjData(m)
    inputs = dict(np.load(source / "inputs.npz"))
    states = dict(np.load(source / "trajectory.npz"))
    d.qpos[:] = inputs["initial_qpos"]
    d.qvel[:] = inputs["initial_qvel"]
    mujoco.mj_forward(m, d)
    paths = {
        k: str(p)
        for k, p in [
            ("stand", ROOT.parent / "microduck/policies/alpha_stand.onnx"),
            ("run", ROOT.parent / "microduck/policies/alpha_walking.onnx"),
            ("victory_hop", ROOT / "policies/ropehop_centered.onnx"),
        ]
    }
    r = DuckRuntime(m, d, ParkourPolicyBank(paths), prefix="duck/")
    driver = ArcadeDriver([Hazard(**h) for h in parent["level"]["hazards"]])
    audit = Audit(interactive_props=("playball", "pin0", "pin1", "pin2"))
    props = PropAudit()
    trace = []
    rows = []
    ur = []
    victory = None
    prefix = 0
    wall = time.monotonic()
    for i in range(2200):
        t = float(d.time)
        if t < branch_time - 1e-6:
            d.ctrl[:] = inputs["ctrl"][i]
            d.eq_active[:] = inputs["eq_active"][i]
            d.xfrc_applied[:] = inputs["xfrc_applied"][i]
            m.geom_rgba[:] = inputs["geom_rgba"][i]
            stage = parent["trace"][i]["stage"]
            prefix += 1
        else:
            if victory is None:
                victory = VictoryDance(t)
                r.last_action = (d.ctrl[r.act_ids] - DEFAULT_POSE).astype(np.float32)
            victory.update(m, d, r)
            driver.step(m, d, float(r.trunk_pos()[0]))
            r.step()
            stage = "CELEBRATE" if victory.finished_at else "FINISH"
        ur.append(
            (
                float(d.time),
                d.ctrl.copy(),
                d.eq_active.copy(),
                d.xfrc_applied.copy(),
                m.geom_rgba.copy(),
            )
        )
        for j in range(100):
            mujoco.mj_step(m, d)
            audit.sample(m, d, r, intentional_rotation=stage in ("ROLL", "EXIT_ROLL"))
            props.sample(m, d, r)
            driver.sample(m, d)
            if victory:
                victory.sample(m, d, r)
            if j % 25 == 0:
                rows.append(
                    (
                        float(d.time),
                        d.qpos.copy(),
                        d.qvel.copy(),
                        d.ctrl.copy(),
                        d.eq_active.copy(),
                        m.geom_rgba.copy(),
                    )
                )
        trace.append(
            dict(
                t=float(d.time),
                stage=stage,
                pos=r.trunk_pos().tolist(),
                up=float(d.xmat[r.trunk_body_id, 8]),
                yaw=r.trunk_yaw(),
                boss=d.body("boss").xpos.tolist(),
                victory_phase=victory.phase if victory else None,
            )
        )
        if victory and victory.finished_at and d.time - victory.finished_at >= 7.0:
            break
    for k, name in enumerate(
        ["time", "qpos", "qvel", "ctrl", "eq_active", "geom_rgba"]
    ):
        actual = np.asarray([row[k] for row in rows[: prefix * 4]])
        if not np.array_equal(actual, states[name][: prefix * 4]):
            raise RuntimeError("Historical prefix mismatch: " + name)
    np.savez_compressed(
        output / "trajectory.npz",
        **{
            name: np.asarray([row[k] for row in rows])
            for k, name in enumerate(
                ["time", "qpos", "qvel", "ctrl", "eq_active", "geom_rgba"]
            )
        },
    )
    np.savez_compressed(
        output / "inputs.npz",
        initial_qpos=inputs["initial_qpos"],
        initial_qvel=inputs["initial_qvel"],
        **{
            name: np.asarray([row[k] for row in ur])
            for k, name in enumerate(
                ["time", "ctrl", "eq_active", "xfrc_applied", "geom_rgba"]
            )
        },
    )
    v = victory.report(m, d, r)
    a = audit.report()
    report = parent | dict(
        source_hashes=hashes,
        source_archive_sha256=hashlib.sha256(
            (output / "source.tar.gz").read_bytes()
        ).hexdigest(),
        wall_seconds=time.monotonic() - wall,
        trace=trace,
        audit=a,
        props=asdict(props),
        victory=v,
        finished=victory.finished_at is not None,
        stage="CELEBRATE",
        warnings=d.warning.number.tolist(),
        component_course_passed=bool(
            parent["component_course_passed"]
            and v["passed"]
            and a["hp"] > 0
            and not d.warning.number.any()
        ),
        passed=False,
    )
    report["continuation"] = dict(
        parent_audit_sha256=hashlib.sha256(
            (source / "audit.json").read_bytes()
        ).hexdigest(),
        parent_capture=parent["capture"],
        parent_source_archive_sha256=parent["source_archive_sha256"],
        branch_time=float(victory.started),
        prefix_control_ticks=prefix,
        prefix_exact=True,
        parent_wall_seconds=parent["wall_seconds"],
        scope="Historical live-Jev prefix replayed exactly; new actuator-only finish suffix. Selected demonstration, not a new live API or unseen trial.",
    )
    report["events"] = sorted(
        [e for e in parent["events"] if e["t"] <= victory.started + 1e-8]
        + [
            e
            for e in audit.events + props.events + driver.events + victory.events
            if e["t"] >= victory.started
        ]
        + [dict(type="VICTORY_START", t=victory.started)],
        key=lambda e: e["t"],
    )
    report["skills"] = [s for s in parent["skills"] if s["t"] < victory.started]
    report["policy_hashes"] = parent["policy_hashes"] | {
        "victory_hop": hashlib.sha256(
            Path(paths["victory_hop"]).read_bytes()
        ).hexdigest()
    }
    report["capture"] = {
        n: hashlib.sha256((output / n).read_bytes()).hexdigest()
        for n in ["scene.mjb", "trajectory.npz", "inputs.npz"]
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                victory=v,
                hp=a["hp"],
                passed=report["component_course_passed"],
                prefix=prefix,
                seconds=float(d.time),
            ),
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--branch-time", type=float, default=25.24)
    run(**vars(p.parse_args()))
