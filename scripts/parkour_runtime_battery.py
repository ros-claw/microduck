"""Validate state-based handoff from running rolls and recovery."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json
from pathlib import Path
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.skills import Maneuver, lane_command
from microduck_lab.parkour.hazards import Hazard


def roll_trial(
    seed=0,
    floor_ref=None,
    policy=None,
    hard_contacts=False,
    entry_phase=None,
    bar_height=0.29,
):
    h = Hazard("bar", "push_bar", 0.9, z=bar_height)
    m, d, r, tr = build_world(
        hazards=[h], duck_floor_ref=floor_ref, hard_contacts=hard_contacts
    )
    if policy:
        r.bank.paths["roulade"] = str(Path(policy).resolve())
    rng = np.random.default_rng(seed)
    d.qpos[r.joint_qpos_idx] += rng.uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    skill = None
    result = None
    events = []
    hazard_pen = []
    floor_pen = []
    self_pen = []
    samples = []
    for i in range(450):
        if d.time < 2:
            r.active_policy = "stand"
            r.set_command()
        else:
            ready = r.trunk_pos()[0] > 0.9 - 0.37
            if entry_phase:
                side, vertical = entry_phase.split("_")
                contacts = []
                for ci, c in enumerate(d.contact):
                    names = [m.geom(g).name for g in (c.geom1, c.geom2)]
                    if f"duck/{side}_foot_collision" in names and any(
                        n.startswith("floor/") for n in names
                    ):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(m, d, ci, force)
                        if force[0] > 1e-5:
                            contacts.append(ci)
                vz = float(r.trunk_linvel()[2])
                ready = (
                    0.4 < r.trunk_pos()[0] < 0.66
                    and bool(contacts)
                    and (0.1 < vz < 0.3 if vertical == "rising" else -0.2 < vz < -0.05)
                )
            if skill is None and ready:
                skill = Maneuver("ROLL_CENTER", float(d.time))
                events.append(
                    dict(
                        type="SKILL_START",
                        t=float(d.time),
                        vx=float(r.trunk_linvel()[0]),
                    )
                )
            if skill is not None and skill.status == "RUNNING":
                status = skill.update(m, d, r)
                if status != "RUNNING":
                    result = dict(
                        status=status,
                        time=float(d.time) - skill.started,
                        rotation=skill.rotation,
                    )
            else:
                r.active_policy = "run"
                r.command[:3] = lane_command(r, 0.0, 0.7)
        r.step()
        for _ in range(100):
            mujoco.mj_step(m, d)
            for c in d.contact:
                names = [m.geom(c.geom1).name, m.geom(c.geom2).name]
                if any(n.startswith("floor/") for n in names) and any(
                    n.startswith("duck/") for n in names
                ):
                    floor_pen.append(max(0.0, -float(c.dist)))
                if all(n.startswith("duck/") for n in names):
                    self_pen.append(max(0.0, -float(c.dist)))
                if "bar/geom" in names and any(n.startswith("duck/") for n in names):
                    hazard_pen.append(max(0.0, -float(c.dist)))
        samples.append(
            dict(
                t=float(d.time),
                pos=r.trunk_pos().tolist(),
                up=float(d.xmat[r.trunk_body_id, 8]),
                skill_phase=skill.phase if skill else None,
                yaw=float(r.trunk_yaw()),
            )
        )
    return dict(
        policy=policy,
        entry_phase=entry_phase,
        bar_height=bar_height,
        hard_contacts=hard_contacts,
        ground_margin_m=0.0002 if (hard_contacts or floor_ref) else 0.0,
        max_self_penetration_m=max(self_pen, default=0.0),
        seed=seed,
        duck_floor_ref=floor_ref,
        max_floor_penetration_m=max(floor_pen, default=0.0),
        result=result,
        events=events,
        contacts=len(hazard_pen),
        max_pen=max(hazard_pen, default=0.0),
        final=samples[-1],
        samples=samples,
    )


if __name__ == "__main__":
    out = []
    for seed in range(10):
        r = roll_trial(seed)
        out.append(r)
        print({k: v for k, v in r.items() if k != "samples"}, flush=True)
    (ROOT / "artifacts/neon-escape-v2/running-roll-validation.json").write_text(
        json.dumps(out)
    )
