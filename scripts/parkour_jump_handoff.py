"""Full-rate gap proof plus measured landing-to-running handoff."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, argparse, itertools, hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.jump import GapAudit, CandidateJump
from microduck_lab.parkour.skills import lane_command


def trial(args):
    policy, gap, seed, *extra = args
    edge = extra[0] if extra else 0.10
    floor_ref = extra[1] if len(extra) > 1 else None
    hard_contacts = bool(extra[2]) if len(extra) > 2 else False
    m, d, r, tr = build_world(
        gaps=((edge, edge + gap),),
        duck_floor_ref=floor_ref,
        hard_contacts=hard_contacts,
    )
    r.bank.paths["jump"] = policy
    rng = np.random.default_rng(seed)
    d.qpos[r.joint_qpos_idx] += rng.uniform(-0.005, 0.005, 14)
    mujoco.mj_forward(m, d)
    audit = GapAudit()
    skill = None
    result = None
    samples = []
    for i in range(350):
        if i < 100:
            r.active_policy = "stand"
            r.set_command()
        else:
            if skill is None:
                skill = CandidateJump(float(d.time), edge + gap)
            if skill.status == "RUNNING":
                status = skill.update(m, d, r, audit)
                if status != "RUNNING":
                    result = dict(
                        status=status,
                        duration=float(d.time) - skill.started,
                        yaw=r.trunk_yaw(),
                        x=float(r.trunk_pos()[0]),
                    )
            else:
                r.active_policy = "run"
                r.joint_vel_delay = 0
                r.command[:3] = lane_command(r, 0.0, 0.7)
        r.step()
        for _ in range(100):
            mujoco.mj_step(m, d)
            audit.sample(m, d, r)
        samples.append(
            dict(
                t=float(d.time),
                pos=r.trunk_pos().tolist(),
                up=float(d.xmat[r.trunk_body_id, 8]),
                yaw=r.trunk_yaw(),
                phase=skill.phase if skill else None,
            )
        )
        if r.trunk_pos()[2] < -0.3:
            break
    return dict(
        hard_contacts=hard_contacts,
        policy=policy,
        policy_sha256=hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
        gap=gap,
        seed=seed,
        edge=edge,
        result=result,
        flights=audit.flights,
        final=samples[-1],
        samples=samples,
        passed=bool(
            result
            and result["status"] == "SUCCESS"
            and samples[-1]["up"] > 0.9
            and r.trunk_pos()[2] > 0.09
        ),
        warnings=d.warning.number.tolist(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = []
        for r in pool.map(
            trial,
            itertools.product(
                [str(Path(a.policy).resolve())], [0.12, 0.15, 0.18, 0.20], range(10)
            ),
        ):
            out.append(r)
            print(r["gap"], r["seed"], r["passed"], r["result"], flush=True)
    summary = {
        str(g): sum(r["passed"] for r in out if r["gap"] == g)
        for g in [0.12, 0.15, 0.18, 0.20]
    }
    Path(a.output).write_text(json.dumps(dict(summary=summary, results=out)))
    print(summary, flush=True)
