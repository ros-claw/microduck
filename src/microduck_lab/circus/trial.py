"""Real four-robot rehearsal; no pose writes after spawn or collision disabling."""
from dataclasses import asdict,dataclass
import hashlib,json,math,time
from pathlib import Path
import mujoco
import numpy as np
from ..demos.honest_skip import run_honest_classic_skip
from ..sim.composer import DuckSpec
from ..sim.runtime import DuckRuntime
from ..sim.skip_metrics import PhysicalSkipAudit
from .mission import RelayRequest
from .relay import RelayGate,joint_cycle_result
from .geometry import outside_rope_region,local_rope_phase
from .diagnostics import FormationDiagnostics

ROOT=Path(__file__).resolve().parents[3]
SOURCES=['sim/classic_rope.py','sim/skip_metrics.py','sim/runtime.py',
         'demos/honest_skip.py','circus/relay.py','circus/geometry.py','circus/trial.py','circus/diagnostics.py']
# Snapshot loaded code once; concurrent edits cannot relabel an in-flight trial.
PROTOCOL_SOURCE={p:(ROOT/'src/microduck_lab'/p).read_text() for p in SOURCES}
PROTOCOL_SHA256=hashlib.sha256(b''.join(PROTOCOL_SOURCE[p].encode() for p in SOURCES)).hexdigest()

@dataclass(frozen=True)
class TrialConfig:
    kind: str = 'relay'
    seed: int = 0
    seconds: float = 30.
    dx: float = .075
    span: float = .448
    rope_length: float = .58
    sky_y: float = 0.
    hop_phase_mode: str = "midpoint"
    continuous_turn_phase: bool = False
    fixed_turn_rate: bool = False
    diagnostics: bool = False
    duo_y: float = 0.
    graphite_goal_y: float = 0.
    graphite_y: float = -.50
    entry_phase: float = 0.
    entry_controller: str = 'walk'
    entry_phase_reference: str = 'local'
    entry_head_tuck: float = 0.
    entry_speed: float = 0.  # 0 preserves the original direct-target probe
    entry_prehop: float = 0.
    walk_speed: float = .30
    switch_y: float = -.04
    turn_hz: float = 3.1
    sky_cycles: int = 5
    duo_cycles: int = 3
    graphite_cycles: int = 5

    def __post_init__(self):
        if self.hop_phase_mode not in ('midpoint','local'):raise ValueError('Unknown rope phase sensor')
        if self.kind not in ('duo','entry','relay','speed'):raise ValueError('Unknown physical trial kind')
        if not 0<self.seconds<=120 or not 0<self.turn_hz<=5 or not 0<self.walk_speed<=.6:
            raise ValueError('Invalid duration, cadence, or walking command')
        if not 0<=self.dx<=.20:raise ValueError('Unsupported horizontal spacing')
        if abs(self.graphite_y)<.25:raise ValueError('Entry spawn must be outside the rope region')
        if not 0<=self.entry_head_tuck<=.65:raise ValueError('Head tuck exceeds tested joint range')
        if not 0<=self.entry_speed<=.25 or not 0<=self.entry_prehop<=4:raise ValueError('Invalid entry ramp or preparation duration')
        if self.entry_phase_reference not in ('local','leader','shared'):raise ValueError('Unknown phase reference')
        if self.entry_controller not in ('walk','hop'):raise ValueError('Unknown entry controller')
        RelayRequest(self.sky_cycles,self.duo_cycles,self.graphite_cycles)


def run_trial(cfg:TrialConfig,*,video=None,capture_dir=None):
    names=('sky',) if cfg.kind=='speed' else ('sky','graphite')
    audits={name:PhysicalSkipAudit(name) for name in names}
    relay=RelayGate(RelayRequest(cfg.sky_cycles,cfg.duo_cycles,cfg.graphite_cycles))
    state=dict(steps=0,contacts=[],robot_contacts=[],positions=[],entry_started=None,entry_inside=None,
               entry_clean=0,last_entry_cycle=-1.,entry_faults=set(),bow_seen=False,supported_since=None)
    diagnostic=FormationDiagnostics() if cfg.diagnostics else None
    captured=[];capture_model=None
    writer=None
    if video:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(str(video),fps=25,macro_block_size=1,codec='libx264')

    def setup(m,d,info):
        nonlocal capture_model
        capture_model=m
        m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
        if m.nmocap:raise ValueError('Mocap is forbidden for rehearsal')
        for audit in audits.values():audit.setup(m,d,info)
        if diagnostic:diagnostic.setup(m,d,info)
        state['owners']=[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '').split('/')[0] for b in m.geom_bodyid]
        state['model_sha256']=hashlib.sha256(m.names+m.body_mass.tobytes()+m.geom_size.tobytes()+m.geom_contype.tobytes()+m.geom_conaffinity.tobytes()+m.eq_data.tobytes()).hexdigest()

    def physics(m,d,info,hopping):
        for audit in audits.values():audit(m,d,info,hopping)
        if diagnostic:diagnostic.physics(m,d)
        collision=False
        robot_pairs=set()
        robot_names={'lavender','cream',*names}
        for c in d.contact:
            owners={state['owners'][int(c.geom1)],state['owners'][int(c.geom2)]}
            if c.efc_address>=0 and len(owners)==2 and owners<=robot_names:
                robot_pairs.add('/'.join(sorted(owners)))
            if c.efc_address>=0 and owners=={'sky','graphite'}:collision=True
        if robot_pairs:
            for audit in audits.values():audit.scorer.state['faults'].add('inter_robot_contact')
            if not state['robot_contacts'] or d.time-state['robot_contacts'][-1]['t']>.02:
                state['robot_contacts'].append(dict(t=float(d.time),pairs=sorted(robot_pairs)))
            if d.time>=9 and cfg.kind=='relay':relay.fault('inter_robot_contact',d.time)
        if collision:
            for audit in audits.values():audit.scorer.state['faults'].add('inter_jumper_contact')
            if not state['contacts'] or d.time-state['contacts'][-1]>.02:state['contacts'].append(float(d.time))
            if d.time>=9 and cfg.kind=='relay':relay.fault('inter_jumper_contact',d.time)
        if state['entry_started'] is not None and ((cfg.kind=='entry' and state['entry_clean']<3) or (cfg.kind=='relay' and relay.stage=='entry')):
            for name in names:
                faults=audits[name].scorer.state['faults']
                physical=faults & {'rope_duck_contact','body_ground_contact','tilted','fallen_turner','loose_attachment','collisions_disabled','rope_below_floor','inter_jumper_contact','inter_robot_contact'}
                state['entry_faults'].update(physical)
                if physical and cfg.kind=='relay':relay.fault('entry:'+','.join(sorted(physical)),d.time)
        if state['steps']%100==0:
            state['positions'].append(dict(t=float(d.time),positions={n:d.body(n+'/trunk_base').xpos.tolist() for n in names}))
        if capture_dir is not None and state['steps']%25==0:
            captured.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
        state['steps']+=1

    def policy(m,d,info,rt):
        for name in ('lavender','cream'):rt[name].continuous_turn_phase=cfg.continuous_turn_phase
        if cfg.hop_phase_mode=='local' and not state.get('local_phase_configured'):
            for name in names:rt[name].hop_phase_source=lambda name=name:local_rope_phase(m,d,audits[name])
            state['local_phase_configured']=True
        if diagnostic:diagnostic.sample(m,d,rt)
        if cfg.kind=='speed':
            for n in ('lavender','cream'):rt[n].turn_frequency=cfg.turn_hz
            return
        g=rt['graphite'];sky=rt['sky'];now=float(d.time)
        if 'graphite_phase_source' not in state:state['graphite_phase_source']=g.hop_phase_source
        last={n:au.scorer.cycles[-1] if au.scorer.cycles else None for n,au in audits.items()}
        entry_side=1 if cfg.graphite_y>0 else -1
        inside=abs(g.trunk_pos()[1]-cfg.graphite_goal_y)<.045 and abs(g.trunk_pos()[0]-cfg.dx)<.05
        outside=outside_rope_region(m,d,audits['sky'],cfg.rope_length,-entry_side)
        if cfg.kind in ('entry','relay') and 'started_outside' not in state:
            state['started_outside']=outside_rope_region(m,d,audits['graphite'],cfg.rope_length,entry_side)
            if not state['started_outside']:
                state['entry_faults'].add('spawn_inside_swept_region')
                relay.fault('spawn_inside_swept_region',now)
        if cfg.kind=='relay':
            supported=False
            if relay.stage=='finale':
                # A servo-driven bow, with measured motion and stable return required.
                dt=now-relay.stage_started
                for n in list(rt):
                    if n in ('lavender','cream') and not state.get('stand_'+n):
                        old=rt[n];rt[n]=DuckRuntime(m,d,old.bank,prefix=n+'/',name=n)
                        rt[n].turn_frequency=old.turn_frequency;rt[n].phase_offset_s=old.phase_offset_s
                        state['stand_'+n]=True
                    robot=rt[n];robot.active_policy='stand';robot.set_command()
                    targets=robot.default_pose[5:9].copy();targets[0]+=.20*math.sin(math.pi*min(dt/2,1))
                    robot.head_override=targets
                if .5<dt<1.5 and abs(d.qpos[rt['sky'].joint_qpos_idx[5]]-rt['sky'].default_pose[5])>.08:state['bow_seen']=True
                feet_supported=set()
                for c in d.contact:
                    if c.efc_address<0:continue
                    for geom in (c.geom1,c.geom2):
                        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(geom)) or ''
                        if 'foot_collision' in name and m.geom('floor').id in (c.geom1,c.geom2):feet_supported.add(name.split('/')[0])
                stable=all(r.is_upright(.8) for r in rt.values()) and len(feet_supported)==4
                state['supported_since']=state['supported_since'] or now if stable else None
                supported=dt>2.5 and state['bow_seen'] and state['supported_since'] is not None and now-state['supported_since']>.5
            relay.consume(last['sky'],last['graphite'],t=now,graphite_inside=inside,sky_outside=outside,finale_supported=supported)
        if cfg.entry_head_tuck and cfg.kind in ('entry','relay'):
            prep_start=(max(11.,relay.stage_started+cfg.entry_prehop)-cfg.entry_prehop) if cfg.kind=='relay' else 11-cfg.entry_prehop
            blend=float(np.clip((now-prep_start)/.6,0,1)) if state['entry_inside'] is None else max(0.,1-(now-state['entry_inside'])/.8)
            target=g.default_pose[5:9].copy();target[:2]+=cfg.entry_head_tuck*blend
            g.head_override=target
        stage=relay.stage if cfg.kind=='relay' else ('duo' if cfg.kind=='duo' else 'entry')
        if stage=='sky_solo':g.active_policy='stand';g.set_command()
        elif stage=='entry':
            phase=sky.hop_phase_source()
            phase_error=abs(math.atan2(math.sin(phase-cfg.entry_phase),math.cos(phase-cfg.entry_phase)))
            entry_not_before=max(11.,relay.stage_started+cfg.entry_prehop) if cfg.kind=='relay' else 11.
            if state['entry_started'] is None and cfg.entry_prehop and now>=entry_not_before-cfg.entry_prehop:
                g.active_policy='jump';g.hop_target=np.array([cfg.dx,cfg.graphite_y,math.pi/2]);g.set_command()
                g.hop_phase_source=sky.hop_phase_source
            if state['entry_started'] is None and now>=entry_not_before and phase_error<.2:
                state['entry_started']=now
            if state['entry_started'] is not None:
                if cfg.entry_controller=='hop':
                    target_y=cfg.graphite_goal_y
                    if cfg.entry_speed:
                        distance=min(abs(cfg.graphite_goal_y-cfg.graphite_y),cfg.entry_speed*(now-state['entry_started']))
                        target_y=cfg.graphite_y-entry_side*distance
                    g.active_policy='jump';g.hop_target=np.array([cfg.dx,target_y,math.pi/2]);g.set_command()
                    g.hop_phase_source=sky.hop_phase_source if cfg.entry_phase_reference=='shared' or (cfg.entry_phase_reference=='leader' and not inside) else state['graphite_phase_source']
                    if inside and state['entry_inside'] is None:state['entry_inside']=now
                elif (g.trunk_pos()[1]-cfg.graphite_goal_y)*entry_side>abs(cfg.switch_y):
                    g.active_policy='walk';g.set_command(twist=(-entry_side*cfg.walk_speed,0.,0.))
                else:
                    g.active_policy='jump';g.set_command()
                    if inside and state['entry_inside'] is None:state['entry_inside']=now
                c=last['graphite'];s=last['sky']
                if c and c['end']>state['last_entry_cycle']:
                    state['last_entry_cycle']=c['end']
                    good=c['start']>=state['entry_started'] and c['clean'] and inside and s and s['clean'] and abs(s['end']-c['end'])<.002
                    state['entry_clean']=state['entry_clean']+1 if good else 0
        elif stage=='duo':
            g.active_policy='jump';g.set_command()
            if cfg.kind=='relay':g.hop_target=np.array([cfg.dx,cfg.graphite_goal_y,math.pi/2])
        elif stage=='exit':
            g.active_policy='jump';g.set_command();sky.active_policy='walk';sky.set_command(twist=(-entry_side*cfg.walk_speed,0.,0.))
            if cfg.entry_controller=='hop':
                sky.active_policy='jump';sky.hop_target=np.array([-cfg.dx,-entry_side*max(.5,cfg.rope_length/2+.18),math.pi/2])
            faults=audits['sky'].scorer.state['faults'] & {'rope_duck_contact','body_ground_contact','tilted','inter_jumper_contact'}
            if faults:relay.fault('exit:'+','.join(sorted(faults)),now)
        elif stage=='graphite_solo':
            sky.active_policy='stand';sky.set_command();g.active_policy='jump';g.set_command()

    def overlay(frame,t,legacy):
        from PIL import Image,ImageDraw,ImageFont
        im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        except OSError:font=ImageFont.load_default()
        draw.rectangle((0,0,960,88),fill='black')
        draw.text((15,8),f'ROSCLAW | FOUR-DUCK {cfg.kind.upper()} | physical trial | {t:.2f}s',fill='white',font=font)
        status=relay.stage if cfg.kind=='relay' else 'STATIC DUO TEST' if cfg.kind=='duo' else 'ENTRY EXPERIMENT'
        if t<9:status+=' | startup window (not scored)'
        elif cfg.kind=='duo':status+=' | scored full rope cycles | not a relay'
        if state['entry_faults']:status='ENTRY FAILED: '+', '.join(sorted(state['entry_faults']))
        if cfg.kind=='relay' and relay.failure:status='RELAY FAILED: '+relay.failure
        draw.text((15,34),status[:100],fill='yellow',font=font)
        draw.text((15,60),' / '.join(f'{n}: {a.scorer.result()["clean_skips"]}/{len(a.scorer.cycles)}' for n,a in audits.items()),fill='white',font=font)
        writer.append_data(np.asarray(im));return np.asarray(im)

    ducks=None if cfg.kind=='speed' else [DuckSpec('lavender',(-cfg.span/2,0),0),DuckSpec('cream',(cfg.span/2,0),math.pi),
        DuckSpec('sky',(-cfg.dx,cfg.sky_y),math.pi/2),DuckSpec('graphite',(cfg.dx,cfg.duo_y if cfg.kind=='duo' else cfg.graphite_y),math.pi/2)]
    error=None;start=time.monotonic()
    try:
        run_honest_classic_skip(None,seconds=cfg.seconds,seed=cfg.seed,render=bool(writer),overlay_fn=overlay if writer else None,
            turner_onnx=ROOT/'policies/turner_rope.onnx',hop_onnx=ROOT/'policies/ropehop_contact.onnx',ducks=ducks,
            rope_contacts='full',legacy_rope_offset=False,connect_timeconst=.004,rope_floor_timeconst=.0004,physics_dt=.0002,
            hop_start_delay=0,rope_joint_type='ball',rope_radius=.0015,rope_length=cfg.rope_length,jumper_y=0,settle_seconds=0,
            rope_initial_phase=math.pi/2,rope_velocity_limit=0,rope_pass_time_fn=lambda:audits['sky'].last_underfoot_crossing,
            fixed_turn_rate=cfg.fixed_turn_rate,max_turn_hz=cfg.turn_hz,turn_hz=cfg.turn_hz,model_setup=setup,physics_observer=physics,policy_observer=policy,
            asset_colors=True,presentation='studio' if writer or capture_dir is not None else 'lab')
    except (RuntimeError,ValueError) as exc:error=str(exc)
    finally:
        if writer:writer.close()
    if capture_dir is not None and captured:
        directory=Path(capture_dir);directory.mkdir(parents=True,exist_ok=True)
        mujoco.mj_saveModel(capture_model,str(directory/'scene.mjb'),None)
        np.savez_compressed(directory/'trajectory.npz',time=np.array([v[0] for v in captured]),qpos=np.stack([v[1] for v in captured]),qvel=np.stack([v[2] for v in captured]),ctrl=np.stack([v[3] for v in captured]))
    scores={n:a.scorer.result() for n,a in audits.items()}
    cycles=scores['sky']['cycles'];measured=len(cycles)/(cycles[-1]['end']-cycles[0]['start']) if cycles else None
    joint=joint_cycle_result(scores['sky'],scores['graphite']) if 'graphite' in scores else None
    passed=False
    if not error:
        if cfg.kind=='relay':passed=relay.result()['passed']
        elif cfg.kind=='duo':passed=joint['passed'] and not any(e['t']>=9 for e in state['robot_contacts'])
        elif cfg.kind=='entry':passed=state['entry_clean']>=3 and not state['entry_faults'] and not any(e['t']>=11 for e in state['robot_contacts'])
        elif cfg.kind=='speed':passed=scores['sky']['passed'] and measured is not None and abs(measured/cfg.turn_hz-1)<=.05
    return dict(kind=cfg.kind,seed=cfg.seed,config=asdict(cfg),passed=bool(passed),error=error,simulation_only=True,
        elapsed_s=time.monotonic()-start,protocol_sha256=PROTOCOL_SHA256,model_sha256=state.get('model_sha256'),
        policy_sha256={p:hashlib.sha256((ROOT/'policies'/p).read_bytes()).hexdigest() for p in ['ropehop_contact.onnx','turner_rope.onnx']},
        measured_hz=measured,jumpers=scores,joint_cycles=joint,relay=relay.result(),entry=dict(started_outside=state.get('started_outside'),started=state['entry_started'],inside=state['entry_inside'],clean_cycles=state['entry_clean'],faults=sorted(state['entry_faults'])),
        inter_robot_contacts=state['robot_contacts'],diagnostics=diagnostic.result() if diagnostic else None,inter_jumper_contact_times=state['contacts'],positions=state['positions'])
