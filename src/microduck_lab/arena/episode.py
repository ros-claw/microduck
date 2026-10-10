"""DG-01 calibration: walk to neighbour, then intentionally release its support.

This is a scripted physics experiment, not autonomous survival or a referee.
Contact samples are pre-integration; states include the full integration state.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from .world import build_world, ASSETS

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out, seed=11, dt=.0005, duration=12.):
    if not np.isfinite(duration) or duration <= 10 or duration > 30:
        raise ValueError('DG-01 duration must be >10 and <=30 seconds')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, duck = build_world(dt)
    rng = np.random.default_rng(seed)
    d.qpos[duck.joint_qpos_idx] += rng.uniform(-.005, .005, 14)
    mujoco.mj_forward(m, d)
    mujoco.mj_saveModel(m, str(out/'scene.mjb'), None)
    states, controls, equalities = [], [], []
    times, poses, tile_z, supported = [], [], [], []
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    tiles = [m.body(f'tile_{i}').id for i in range(9)]
    eq = [m.equality(f'tile_{i}_support').id for i in range(9)]
    geom_tiles = {m.geom(f'tile_{i}_geom').id: i for i in range(9)}
    feet = {m.geom('cream/'+n+'_foot_collision').id for n in ('left','right')}
    events, depths = [], []
    nsub = round(.02/dt)
    wall_start = time.monotonic()
    locked_drift = 0.
    support_before_release = False
    motor_force_max = 0.
    time_reset = False
    receiver_contact = False
    next_contact = np.zeros(6)
    with gzip.open(out/'contacts.jsonl.gz', 'wt') as f:
        for step in range(round(duration/dt)):
            t = step*dt
            if step % nsub == 0:
                # Adjacent centre target; feedback is current pose, not root editing.
                delta = -duck.trunk_pos()[0]
                duck.active_policy = 'walk' if 1 <= t < 7 and abs(delta) > .045 else 'stand'
                speed = float(np.clip(delta*2.5, -.45, .45)) if 1 <= t < 7 else 0.
                duck.set_command(twist=(speed, float(np.clip(-duck.trunk_pos()[1], -.08, .08)),
                    float(np.clip(-duck.trunk_yaw(), -.6, .6))))
                duck.step()
            for i, warn, release in ((1, 4., 6.), (4, 8., 10.)):
                if step == round(warn/dt):
                    events.append(dict(time=t, tile=i, state='WARNING'))
                if step == round(release/dt):
                    d.eq_active[eq[i]] = 0
                    events.append(dict(time=t, tile=i, state='RELEASED'))
            mujoco.mj_getState(m, d, state, STATE)
            states.append(state.copy())
            controls.append(d.ctrl.copy())
            equalities.append(d.eq_active.copy())
            # mj_step reports contacts/forces evaluated at this pre-step state.
            mujoco.mj_step(m, d)
            time_reset |= abs(d.time-(t+dt)) > 1e-7
            motor_force_max = max(motor_force_max, float(np.max(np.abs(d.actuator_force))))
            contacts, supports = [], []
            for j in range(d.ncon):
                c = d.contact[j]
                mujoco.mj_contactForce(m, d, j, next_contact)
                a, b = int(c.geom1), int(c.geom2)
                contacts.append([a, b, float(c.dist), *next_contact.tolist(), *c.pos.tolist(), *c.frame[:3].tolist()])
                depths.append(max(0., -float(c.dist)))
                tile = geom_tiles.get(a, geom_tiles.get(b))
                if tile is not None and (a in feet or b in feet) and next_contact[0] > .05:
                    if d.eq_active[eq[tile]] and abs(c.frame[2]) > .5:
                        supports.append(tile)
                if m.geom('receiver').id in (a,b) and (a in feet or b in feet):
                    receiver_contact = True
            f.write(json.dumps(dict(step=step, t=t, contacts=contacts, supporting_tiles=sorted(set(supports))))+'\n')
            times.append(t)
            poses.append(d.qpos[duck.trunk_qpos_adr:duck.trunk_qpos_adr+7].copy())
            tile_z.append([float(d.qpos[m.jnt_qposadr[m.body(b).jntadr[0]]+2]) for b in tiles])
            supported.append(4 in supports)
            if 7 < t < 8:
                support_before_release |= 4 in supports
            for i in range(9):
                if d.eq_active[eq[i]]:
                    locked_drift = max(locked_drift, abs(tile_z[-1][i]+.025))
    elapsed = time.monotonic()-wall_start
    np.savez_compressed(out/'trajectory.npz', states=states, ctrl=controls, eq_active=equalities,
                        time=times, root=poses, tile_z=tile_z, support=supported)
    depth = np.asarray(depths)
    root = np.asarray(poses)
    audit = dict(schema='microduck.arena.greybox.v1', seed=seed, dt=dt, duration=duration,
        mujoco_version=mujoco.__version__, experiment='scripted neighbour walk and occupied support release',
        events=events, root_initial=root[0].tolist(), root_final=root[-1].tolist(),
        centre_error_at_7s_m=float(np.linalg.norm(root[min(round(7/dt),len(root)-1),:2])),
        centre_supported_before_warning=support_before_release,
        locked_tile_max_drift_m=locked_drift, receiver_foot_contact=receiver_contact,
        penetration_p99_m=float(np.quantile(depth,.99)) if len(depth) else 0.,
        penetration_max_m=float(depth.max()) if len(depth) else 0.,
        solver_warnings=[int(w.number) for w in d.warning], time_reset=time_reset, motor_force_max_Nm=motor_force_max,
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[3])):sha(p) for p in
            [Path(__file__), Path(__file__).with_name('world.py'), Path(__file__).parents[1]/'sim/runtime.py', Path(__file__).parents[1]/'sim/composer.py']},
        finite=bool(np.isfinite(root).all() and np.isfinite(d.qvel).all()),
        wall_s=elapsed, recording_real_time_factor=duration/elapsed,
        external_forces_zero=bool(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)),
        policy_sha256={n:sha(p) for n,p in duck.bank.paths.items()},
        robot_xml_sha256=sha(ASSETS/'microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml'),
        contact_columns=['geom1','geom2','distance','force_normal','force_t1','force_t2','torque_n','torque_t1','torque_t2','px','py','pz','nx','ny','nz'],
        sampling='states and contact forces at pre-integration t; root/tile_z diagnostics at t+dt',
        hashes={p.name:sha(p) for p in (out/'scene.mjb',out/'trajectory.npz',out/'contacts.jsonl.gz')})
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit
