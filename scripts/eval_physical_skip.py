"""Evaluate actual full rope revolutions, geometry, contacts and landing."""
import argparse
import json
import pathlib
import sys

import mujoco

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.sim.skip_metrics import PhysicalSkipAudit
from microduck_lab.demos.honest_skip import run_honest_classic_skip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hop',type=pathlib.Path,default=ROOT/'policies/ropehop_classic.onnx')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--seconds',type=float,default=50.)
    p.add_argument('--rope-floor-timeconst',type=float,default=None)
    p.add_argument('--integrator',choices=['Euler','implicitfast'],default='Euler')
    p.add_argument('--physics-dt',type=float,default=.001)
    p.add_argument('--hop-start-delay',type=float,default=8.)
    p.add_argument('--rope-joint-type',choices=['hinge','ball'],default='hinge')
    p.add_argument('--rope-radius',type=float,default=.003)
    p.add_argument('--geometric-timing',action='store_true')
    p.add_argument('--rope-length',type=float,default=.50)
    p.add_argument('--jumper-y',type=float,default=-.10)
    p.add_argument('--rope-phase-bias',type=float,default=0.)
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--rope-initial-phase',type=float,default=0.)
    p.add_argument('--rope-velocity-limit',type=float,default=40.,help='zero disables legacy numerical clipping')
    p.add_argument('--video',type=pathlib.Path,help='diagnostic video with physical score, including failures')
    p.add_argument('--output',type=pathlib.Path,required=True)
    args = p.parse_args()
    observer = PhysicalSkipAudit()
    def setup(m,d,info):
        m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        m.opt.integrator = getattr(mujoco.mjtIntegrator,'mjINT_'+args.integrator.upper())
        observer.setup(m,d,info)
    result = dict(hop=str(args.hop),seed=args.seed,seconds=args.seconds,
                  contacts='full',solver='Newton',connect_timeconst=.004,
                  corrected_rope_root=True,mujoco=mujoco.__version__)
    result['rope_floor_timeconst'] = args.rope_floor_timeconst
    result['integrator'] = args.integrator
    result['physics_dt'] = args.physics_dt
    result['hop_start_delay'] = args.hop_start_delay
    result['rope_joint_type'] = args.rope_joint_type
    result['rope_radius'] = args.rope_radius
    result['geometric_timing'] = args.geometric_timing
    result['rope_length'] = args.rope_length
    result['jumper_y'] = args.jumper_y
    result['rope_phase_bias'] = args.rope_phase_bias
    result['settle_seconds'] = args.settle_seconds
    result['rope_initial_phase'] = args.rope_initial_phase
    result['rope_velocity_limit'] = args.rope_velocity_limit
    def pass_time():
        value = observer.last_underfoot_crossing
        return value-round(args.settle_seconds*50)/50 if value is not None else None
    writer = None
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True,exist_ok=True)
        writer = imageio.get_writer(str(args.video),fps=25)
    def overlay(frame,t,legacy):
        import numpy as np
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        score = observer.scorer.result()
        draw.rectangle((0,0,1000,48),fill='black')
        draw.text((10,6),f'PHYSICAL CONTACT TEST | {args.hop.name} | t={t:.2f}s',fill='white')
        draw.text((10,27),f'Clean cycles {score["clean_skips"]}/{score["full_revolutions"]} | collisions ON | diagnostic, not accepted delivery',fill='white')
        frame = np.asarray(im)
        writer.append_data(frame)
        return frame
    try:
        metrics,_ = run_honest_classic_skip(None, seconds=args.seconds,seed=args.seed,
            turner_onnx=ROOT/'policies/turner_rope.onnx',hop_onnx=args.hop,
            render=bool(writer),overlay_fn=overlay if writer else None,
            rope_contacts='full',legacy_rope_offset=False,connect_timeconst=.004,
            rope_floor_timeconst=args.rope_floor_timeconst,
            physics_dt=args.physics_dt,
            hop_start_delay=args.hop_start_delay,
            rope_joint_type=args.rope_joint_type,
            rope_radius=args.rope_radius,
            rope_pass_time_fn=pass_time if args.geometric_timing else None,
            rope_length=args.rope_length,jumper_y=args.jumper_y,
            rope_phase_bias=args.rope_phase_bias,
            settle_seconds=args.settle_seconds,
            rope_initial_phase=args.rope_initial_phase,
            rope_velocity_limit=args.rope_velocity_limit,
            physics_observer=observer,model_setup=setup)
        result.update(completed=True,legacy_timing=metrics)
    except RuntimeError as exc:
        result.update(completed=False,failure=str(exc))
    finally:
        if writer:
            writer.close()
    result.update(observer.scorer.result())
    result['crossing_samples'] = observer.crossing_samples
    result['passed'] &= result['completed']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cycles','crossing_samples')},indent=2),flush=True)
    if not result['completed']:
        sys.exit(1)


if __name__=='__main__':
    main()
