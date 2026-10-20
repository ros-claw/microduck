"""Counterfactual recovery experiments from actual recorded game states.

Other robots replay recorded motor targets but remain dynamic contact bodies.
Hazards follow recorded releases. This is not a new live Jev game or exact
replay after the intervention. No root/velocity reset after initialization.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.arena.episode import STATE, sha
from microduck_lab.arena.multiplayer import ContactGraph, observe
from microduck_lab.arena.jev_dynamic_recovery import DynamicRecoveryMotor
from microduck_lab.arena.jev_config import JevGameConfig
from microduck_lab.arena.tiles import Tile
from microduck_lab.arena.world import ASSETS
from microduck_lab.sim.runtime import DuckRuntime, PolicyBank

DATA = None
AUDIT = None
RUN = None
MODEL = None
BANK = None


def init():
    global MODEL, BANK
    MODEL = mujoco.MjModel.from_binary_path(str(RUN / "scene.mjb"))
    BANK = PolicyBank(
        {
            n: str(ASSETS / "microduck/policies" / p)
            for n, p in {
                "stand": "alpha_stand.onnx",
                "sitstand": "alpha_sitstand.onnx",
                "roulade": "roulade.onnx",
                "walk": "alpha_walking.onnx",
            }.items()
        }
    )


def trial(job):
    body, when, strategy = job
    m = MODEL
    d = mujoco.MjData(m)
    start = round(when / 0.00025)
    mujoco.mj_setState(m, d, DATA["states"][start], STATE)
    mujoco.mj_forward(m, d)
    ducks = {
        n: DuckRuntime(m, d, BANK, prefix=n + "/", name=n) for n in AUDIT["characters"]
    }
    target = ducks[body]
    target.last_action = (
        DATA["ctrl"][max(0, start - 1), target.act_ids] - target.default_pose
    ).astype(np.float32)
    tiles = {
        i: Tile(
            i,
            m.equality(f"tile_{i}_support").id,
            int(m.jnt_qposadr[m.joint(f"tile_{i}_free").id]),
            int(m.jnt_dofadr[m.joint(f"tile_{i}_free").id]),
        )
        for i in range(25)
    }
    graph = ContactGraph(m, ducks, tiles)
    releases = [
        (e["time"], tiles[e["tile"]].eq_id)
        for e in AUDIT["events"]
        if e["state"] == "RELEASED"
    ]
    first = None
    stable = 0.0
    peak = 0.0
    drift0 = target.trunk_pos().copy()
    sample = []
    motor = DynamicRecoveryMotor(AUDIT["characters"][body]["role"])
    if strategy == "dynamic_existing":
        prior = [
            e
            for e in AUDIT["events"]
            if e["state"] == "RECOVERY_STARTED"
            and e.get("body_id") == body
            and e["time"] <= when
        ]
        motor.recovering = True
        motor.recovery_since = prior[-1]["time"]
        motor.first_attempt_s = 2.0
        motor.skill_start = (
            motor.recovery_since + 2.0 if motor.recovery_since + 2.0 <= when else None
        )
    config = JevGameConfig(**AUDIT["config"])
    _, support, _ = graph.read(m, d, tiles)
    roll_done = False
    for tick in range(400):
        elapsed = tick * 0.02
        ref = min(start + tick * 80, len(DATA["ctrl"]) - 1)
        d.ctrl[:] = DATA["ctrl"][ref]
        target.active_policy = "stand"
        head = (0, 0, 0, 0)
        twist = (0, 0, 0)
        if strategy == "sit":
            target.active_policy = "sitstand"
        if strategy == "pulse" and elapsed < 0.8:
            target.active_policy = "sitstand"
        if strategy == "cycle" and (elapsed % 2.4) < 0.8:
            target.active_policy = "sitstand"
        if strategy == "roll" and elapsed < 0.5:
            target.active_policy = "roulade"
        if strategy == "roll1" and elapsed < 1.0:
            target.active_policy = "roulade"
        if strategy == "walk":
            target.active_policy = "walk"
        if strategy == "yaw" and elapsed < 0.8:
            twist = (0, 0, 1.0)
        if strategy == "back" and elapsed < 0.8:
            twist = (-0.15, 0, 0)
        if strategy == "head_up" and elapsed < 0.8:
            head = (0.5, -0.5, 0, 0)
        if strategy == "head_down" and elapsed < 0.8:
            head = (-0.5, 0.5, 0, 0)
        if strategy in ("dynamic", "dynamic_existing"):
            obs = observe(body, ducks, tiles, support, config, when + elapsed, set())
            obs["robot"]["up_cos"] = float(d.xmat[target.trunk_body_id, 8])
            for tile in obs["visible_tiles"]:
                tile.update(warning_remaining_s=None, damage=0.0)
            target.active_policy, twist, _ = motor.choose(obs, None)
        if strategy == "roll_feedback":
            if elapsed > 0.1 and d.xmat[target.trunk_body_id, 8] > 0.7:
                roll_done = True
            target.active_policy = (
                "roulade" if not roll_done and elapsed < 1.2 else "stand"
            )
        target.set_command(twist=twist, head=head)
        target.step()
        for sub in range(80):
            mujoco.mj_step(m, d)
            absolute = when + elapsed + (sub + 1) * 0.00025
            for tm, eq in releases:
                if tm <= absolute + 1e-9:
                    d.eq_active[eq] = False
            peak = max(peak, max((-c.dist for c in d.contact), default=0.0))
        _, support, _ = graph.read(m, d, tiles)
        up = float(d.xmat[target.trunk_body_id, 8])
        speed = float(np.linalg.norm(target.trunk_linvel()[:2]))
        good = up > 0.94 and bool(support[body]) and speed < 0.16
        stable = stable + 0.02 if good else 0.0
        if first is None and stable >= 0.18 - 1e-9:
            first = round(elapsed + 0.02, 3)
        if tick % 10 == 0:
            sample.append(
                dict(t=elapsed, up=up, speed=speed, z=float(target.trunk_pos()[2]))
            )
        if target.trunk_pos()[2] < -0.3:
            break
        if first is not None and elapsed >= first + 0.6:
            break
    row = dict(
        body_id=body,
        initialized_sim_time_s=when,
        strategy=strategy,
        completion_s=first,
        final_up=up,
        final_z=float(target.trunk_pos()[2]),
        displacement_m=float(np.linalg.norm(target.trunk_pos()[:2] - drift0[:2])),
        penetration_max_m=peak,
        warnings=d.warning.number.tolist(),
        samples=sample,
    )
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="artifacts/jev-rumble/phase-aware-live-70002")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--cases", nargs="+", default=["lavender:56", "lavender:60", "sky:58"]
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "stand",
            "sit",
            "pulse",
            "cycle",
            "roll",
            "roll1",
            "walk",
            "yaw",
            "back",
            "head_up",
            "head_down",
        ],
    )
    a = p.parse_args()
    RUN = Path(a.run)
    AUDIT = json.loads((RUN / "audit.json").read_text())
    with np.load(RUN / "trajectory.npz") as z:
        DATA = {k: z[k] for k in z.files}
    jobs = [
        (s.split(":")[0], float(s.split(":")[1]), v)
        for s in a.cases
        for v in a.strategies
    ]
    rows = []
    Path(a.out).parent.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(
        max_workers=a.workers,
        initializer=init,
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        for row in pool.map(trial, jobs):
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True
            )
            Path(a.out).write_text(
                json.dumps(
                    dict(
                        source_audit_sha256=sha(RUN / "audit.json"),
                        experiment="counterfactual fixed opponent motor targets and hazard releases; not live Jev",
                        rows=rows,
                    ),
                    indent=2,
                )
                + "\n"
            )
