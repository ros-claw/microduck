"""Evaluate complete physical hops, including body contact and landing support.

This evaluates a single duck without a rope, NOT rope-skip success.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import hashlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from microduck_lab.sim.composer import compose_world, DuckSpec
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime


def evaluate(policy, seed=0, seconds=10., video=None):
    robot = ROOT.parent / 'microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml'
    world = compose_world(robot, [DuckSpec('sky', (0, 0))], playground=False)
    m, d = world.model, world.data
    m.opt.timestep = .001
    m.opt.iterations = 100
    bank = PolicyBank({'stand': str(ROOT.parent/'microduck/policies/alpha_stand.onnx'), 'hop': str(policy)})
    duck = DuckRuntime(m, d, bank, prefix='sky/')
    d.qpos[duck.joint_qpos_idx] = duck.default_pose + np.random.default_rng(seed).uniform(-.02,.02,14)
    mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in range(m.ngeom)]
    feet = {names.index('sky/left_foot_collision'), names.index('sky/right_foot_collision')}
    floor = next(g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    renderer = mujoco.Renderer(m, height=540, width=960) if video else None
    writer = None
    if video:
        import imageio.v2 as imageio
        pathlib.Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=25)
    camera = mujoco.MjvCamera()
    camera.lookat = [0, 0, .12]
    camera.distance = .65
    camera.azimuth, camera.elevation = 130, -15
    flight = pending = None
    support_time = 0.
    events = []
    nonfoot_steps = tilted_steps = 0
    max_drift = 0.
    origin = None
    try:
        for p in range(round((seconds + 2) * 50)):
            hopping = p >= 100
            if p == 100:
                origin = duck.trunk_pos()[:2]
            duck.active_policy = 'hop' if hopping else 'stand'
            duck.step()
            for _ in range(20):
                mujoco.mj_step(m, d)
                if np.any(d.warning.number) or not np.isfinite(d.qacc).all():
                    raise RuntimeError(f'invalid physics at {d.time}')
                grounded = set()
                for c in d.contact:
                    if c.efc_address >= 0 and floor in (c.geom1, c.geom2):
                        grounded.add(int(c.geom2 if c.geom1 == floor else c.geom1))
                nonfoot = bool(grounded - feet)
                upright = bool(d.xmat[duck.trunk_body_id, 8] > .8)
                supported = bool(grounded & feet) and not nonfoot and upright
                if not hopping:
                    support_time = support_time + .001 if supported else 0.
                    continue
                nonfoot_steps += nonfoot
                tilted_steps += not upright
                drift = float(np.linalg.norm(duck.trunk_pos()[:2] - origin))
                max_drift = max(max_drift, drift)
                good = upright and not nonfoot and drift <= .08
                if pending:
                    pending['valid'] &= good and supported
                    pending['landing_time'] += .001
                    if pending['landing_time'] >= .05 or not grounded:
                        pending['clean'] = bool(pending['valid'] and pending['landing_time'] >= .05)
                        events.append(pending)
                        pending = None
                if not grounded:
                    if flight is None:
                        flight = dict(takeoff=float(d.time), duration=0., apex=0.,
                                      valid=bool(support_time >= .02), landing_time=0.)
                    flight['duration'] += .001
                    flight['valid'] &= good
                    clearance = min(mujoco.mj_geomDistance(m,d,g,floor,1.,None) for g in feet)
                    flight['apex'] = max(flight['apex'], float(clearance))
                elif flight:
                    if flight['duration'] >= .03:
                        flight['valid'] &= good and supported and flight['apex'] >= .015
                        pending = flight
                    flight = None
                support_time = support_time + .001 if supported else 0.
            if writer and p % 2 == 0:
                from PIL import Image, ImageDraw
                camera.lookat[:2] = duck.trunk_pos()[:2]
                renderer.update_scene(d, camera=camera)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0,0,650,42), fill='black')
                draw.text((8,5), f'{pathlib.Path(policy).name} | single-duck test | t={d.time:.2f}s', fill='white')
                draw.text((8,23), f'Clean hops {sum(e["clean"] for e in events)}/{len(events)} | non-foot contact {nonfoot_steps*.001:.2f}s', fill='white')
                writer.append_data(np.array(im))
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    if pending:
        pending['clean'] = False
        pending['incomplete_landing'] = True
        events.append(pending)
    if flight and flight['duration'] >= .03:
        flight['clean'] = False
        flight['incomplete_flight'] = True
        events.append(flight)
    return dict(policy=str(policy), sha256=hashlib.sha256(pathlib.Path(policy).read_bytes()).hexdigest(),
                seed=seed, seconds=seconds, mujoco=mujoco.__version__,
                criteria=dict(min_flight_s=.03,min_clearance_m=.015,min_upright_cos=.8,
                              max_drift_m=.08,landing_support_s=.05,preflight_support_s=.02),
                clean_hops=sum(e['clean'] for e in events), attempts=len(events),
                nonfoot_contact_s=nonfoot_steps*.001, tilted_s=tilted_steps*.001,
                max_drift_m=max_drift, events=events)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('policy', type=pathlib.Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3])
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--video', type=pathlib.Path, help='optional video, one seed only')
    args = p.parse_args()
    if args.seconds <= 0 or (args.video and len(args.seeds) != 1):
        p.error('seconds must be positive; video requires one seed')
    results = []
    for seed in args.seeds:
        result = evaluate(args.policy, seed, args.seconds, args.video)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k != 'events'}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
