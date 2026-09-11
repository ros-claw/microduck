"""Headless, per-physics-step audit; legacy timing scores are NOT clean skips."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.demos.honest_skip import run_honest_classic_skip


class Audit:
    def __init__(self):
        self.steps = self.hop_steps = self.tilted_steps = 0
        self.contact_steps = dict(rope_jumper=0, rope_floor=0, jumper_nonfoot_floor=0)
        self.max_anchor_gap = 0.0
        self.min_rope_surface_z = 1.0
        self.trace = []

    def setup(self, m, d, info):
        self.dt = float(m.opt.timestep)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "" for b in m.geom_bodyid]
        self.rope = np.array([n.startswith("rope/rope_s") for n in names])
        self.sky = np.array([n.startswith("sky/") for n in bodies])
        self.feet = np.array([n in ("sky/left_foot_collision", "sky/right_foot_collision") for n in names])
        self.floor = names.index("floor")
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sky/trunk_base")
        self.rope_ids = np.flatnonzero(self.rope)
        self.warning_start = np.array(d.warning.number).copy()
        self.enabled_pairs = {
            "rope_floor": bool(any((m.geom_contype[g] & m.geom_conaffinity[self.floor]) or
                                    (m.geom_contype[self.floor] & m.geom_conaffinity[g]) for g in self.rope_ids)),
            "rope_jumper": bool(any((m.geom_contype[g] & m.geom_conaffinity[s]) or
                                     (m.geom_contype[s] & m.geom_conaffinity[g])
                                     for g in self.rope_ids for s in np.flatnonzero(self.sky))),
        }

    def __call__(self, m, d, info, hopping):
        self.steps += 1
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all()):
            raise RuntimeError(f"nonfinite state at {d.time}")
        if np.any(np.array(d.warning.number) > self.warning_start):
            raise RuntimeError(f"MuJoCo warning at physics step {self.steps} "
                               f"(expected t={self.steps * m.opt.timestep:.3f}, "
                               f"reported t={d.time:.3f}): {d.warning.number}")
        # MuJoCo may automatically reset after divergence: a time reset is a failure.
        if abs(d.time - self.steps * m.opt.timestep) > 1e-6:
            raise RuntimeError(f"simulation time reset at step {self.steps}: {d.time}")
        upright = float(d.xmat[self.trunk, 8])
        self.hop_steps += int(hopping)
        self.tilted_steps += int(hopping and upright < np.cos(np.pi / 4))
        seen = set()
        for c in d.contact:
            a, b = int(c.geom1), int(c.geom2)
            # Only solver-active contacts count, not just nearby candidates.
            if c.efc_address < 0:
                continue
            key = None
            if (self.rope[a] and self.sky[b]) or (self.rope[b] and self.sky[a]):
                key = "rope_jumper"
            elif (self.rope[a] and b == self.floor) or (self.rope[b] and a == self.floor):
                key = "rope_floor"
            elif (a == self.floor and self.sky[b] and not self.feet[b]) or (b == self.floor and self.sky[a] and not self.feet[a]):
                seen.add("jumper_nonfoot_floor")
            if key:
                seen.add(key)
        for key in seen:
            self.contact_steps[key] += 1
        if self.steps % 20 == 0:
            # Geometry endpoints, not body origins: the latter miss half a segment.
            g = self.rope_ids
            dz = np.abs(d.geom_xmat[g, 8]) * m.geom_size[g, 1]
            low = float(np.min(d.geom_xpos[g, 2] - dz - m.geom_size[g, 0]))
            self.min_rope_surface_z = min(self.min_rope_surface_z, low)
            gap = 0.0
            for e in range(m.neq):
                if m.eq_type[e] != mujoco.mjtEq.mjEQ_CONNECT:
                    continue
                a, b = m.eq_obj1id[e], m.eq_obj2id[e]
                pa = d.xpos[a] + d.xmat[a].reshape(3, 3) @ m.eq_data[e, :3]
                pb = d.xpos[b] + d.xmat[b].reshape(3, 3) @ m.eq_data[e, 3:6]
                gap = max(gap, float(np.linalg.norm(pa - pb)))
            self.max_anchor_gap = max(self.max_anchor_gap, gap)
            self.trace.append(dict(t=round(float(d.time), 4), hopping=bool(hopping),
                                   upright_cos=upright, trunk_z=float(d.xpos[self.trunk, 2]),
                                   rope_min_z=low, anchor_gap=gap))

    def result(self):
        return dict(physics_steps=self.steps, hop_steps=self.hop_steps,
                    collision_enabled=self.enabled_pairs,
                    contact_steps=self.contact_steps,
                    contact_seconds={k: v * self.dt for k, v in self.contact_steps.items()},
                    hopping_tilt_over_45deg_fraction=self.tilted_steps / max(1, self.hop_steps),
                    max_anchor_gap_m=self.max_anchor_gap,
                    min_rope_surface_z_m=self.min_rope_surface_z,
                    trace_sample_hz=50)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contacts", choices=["off", "floor", "jumper", "full"], default="off")
    p.add_argument("--solver", choices=["CG", "Newton"], default="CG")
    p.add_argument("--correct-rope-root", action="store_true", help="remove the invisible first-segment offset")
    p.add_argument("--connect-timeconst", type=float, default=0.04)
    p.add_argument("--rope-contact-timeconst", type=float, help="experimental rope contact stiffness, seconds")
    p.add_argument("--rope-floor-timeconst", type=float, help="stiffen only rope/floor explicit pairs")
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    audit = Audit()

    def setup(m, d, info):
        m.opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + args.solver.upper())
        audit.setup(m, d, info)
        if args.rope_contact_timeconst is not None:
            if args.rope_contact_timeconst < 2*m.opt.timestep:
                raise ValueError("contact time constant must be at least two physics steps")
            m.geom_priority[audit.rope_ids] = 1
            m.geom_solref[audit.rope_ids] = [args.rope_contact_timeconst, 1.]
            m.geom_solimp[audit.rope_ids] = [.95, .99, .001, .5, 2.]

    result = dict(config=vars(args) | {"output": str(args.output)}, mujoco_version=mujoco.__version__,
                  physical_success_rate=None,
                  score_note="Legacy score only checks timing. Use eval_physical_skip.py for geometric cycle scoring.")
    start = time.monotonic()
    try:
        metrics, _ = run_honest_classic_skip(None, seconds=args.seconds, seed=args.seed,
            turner_onnx=ROOT / "policies/turner_rope.onnx", hop_onnx=ROOT / "policies/ropehop_classic.onnx",
            render=False, rope_contacts=args.contacts, physics_observer=audit, model_setup=setup,
            legacy_rope_offset=not args.correct_rope_root, connect_timeconst=args.connect_timeconst,
            rope_floor_timeconst=args.rope_floor_timeconst)
        result.update(completed=True, legacy_timing=metrics)
    except RuntimeError as exc:
        result.update(completed=False, failure=str(exc))
    result.update(audit.result(), wall_seconds=time.monotonic() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result | {"trace": audit.trace}, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["completed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
