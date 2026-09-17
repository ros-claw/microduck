"""Plan, rehearse, and practice declared circus tasks with real physics evidence."""
import argparse,hashlib,json,math,os,sys
from dataclasses import replace
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ['PYOPENGL_PLATFORM']=os.environ['MUJOCO_GL']
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.circus.mission import parse_request,speed_curriculum
from microduck_lab.circus.promotion import promotion_gate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def record_trial(cfg,directory,tag,video=None):
    from microduck_lab.circus.trial import run_trial,PROTOCOL_SOURCE,PROTOCOL_SHA256
    path=directory/(tag+'.json')
    if path.exists():raise FileExistsError(f'Evidence already exists: {path}; use a fresh output directory')
    write(directory/'protocols'/(PROTOCOL_SHA256+'.json'),PROTOCOL_SOURCE)
    result=run_trial(cfg,video=video);write(path,result)
    # Keep digest external to the evidence it signs.
    receipt=dict(kind=result['kind'],seed=cfg.seed,passed=result['passed'],error=result['error'],
        evidence=str(path),evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),protocol_sha256=result['protocol_sha256'])
    with (directory/'practice.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
    print(json.dumps(dict(trial=tag,passed=result['passed'],measured_hz=result['measured_hz'],error=result['error'])),flush=True)
    return result,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    plan=subs.add_parser('plan');plan.add_argument('request');plan.add_argument('--output',type=Path,required=True)
    run=subs.add_parser('rehearse');run.add_argument('--kind',choices=['relay','entry','duo','speed'],default='relay')
    run.add_argument('--config',type=Path,help='Replay a TrialConfig JSON or the config from a trial result; overrides simulation flags')
    run.add_argument('--plan',type=Path,help='Execute the counts from a compiled relay request')
    run.add_argument('--seed',type=int,default=0);run.add_argument('--seconds',type=float,default=30)
    run.add_argument('--dx',type=float,default=.075);run.add_argument('--span',type=float,default=.448)
    run.add_argument('--rope-length',type=float,default=.58);run.add_argument('--duo-y',type=float,default=0.)
    run.add_argument('--turn-hz',type=float,default=3.1);run.add_argument('--entry-phase',type=float,default=0.)
    run.add_argument('--entry-phase-reference',choices=['local','leader'],default='local')
    run.add_argument('--graphite-goal-y',type=float,default=0.)
    run.add_argument('--graphite-y',type=float,default=-.50)
    run.add_argument('--entry-controller',choices=['walk','hop'],default='walk')
    run.add_argument('--video',type=Path);run.add_argument('--output-dir',type=Path,required=True)
    practice=subs.add_parser('practice');practice.add_argument('--kind',choices=['entry','duo','speed'],required=True)
    practice.add_argument('--request',help='Optional supported speed request; its percentage drives the curriculum')
    practice.add_argument('--seeds',default='0,1');practice.add_argument('--holdout-seeds',default='701,702,703,704')
    practice.add_argument('--seconds',type=float,default=30);practice.add_argument('--max-candidates',type=int,default=5)
    practice.add_argument('--increase-percent',type=float,default=30);practice.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        result=parse_request(args.request);write(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return
    from microduck_lab.circus.trial import TrialConfig
    if args.command=='rehearse':
        cfg=TrialConfig(kind=args.kind,seed=args.seed,seconds=args.seconds,dx=args.dx,span=args.span,rope_length=args.rope_length,
            entry_phase_reference=args.entry_phase_reference,graphite_goal_y=args.graphite_goal_y,graphite_y=args.graphite_y,entry_controller=args.entry_controller,duo_y=args.duo_y,turn_hz=args.turn_hz,entry_phase=args.entry_phase)
        if args.config:
            config_data=json.loads(args.config.read_text())
            cfg=TrialConfig(**config_data.get('config',config_data))
        if args.plan:
            plan_data=json.loads(args.plan.read_text())
            if plan_data.get('kind')!='relay' or cfg.kind!='relay':parser.error('--plan currently requires a relay mission')
            from microduck_lab.circus.mission import RelayRequest
            request=RelayRequest(**plan_data['request'])
            cfg=replace(cfg,**request.__dict__)
        result,_=record_trial(cfg,args.output_dir,f'{cfg.kind}-seed{cfg.seed}',args.video)
        write(args.output_dir/'rehearsal.json',dict(passed=result['passed'],kind=cfg.kind,evidence=f'{cfg.kind}-seed{cfg.seed}.json'))
        return
    train=[int(s) for s in args.seeds.split(',')];holdout=[int(s) for s in args.holdout_seeds.split(',')]
    if not train or len(set(train))!=len(train) or len(set(holdout))!=len(holdout) or set(train)&set(holdout) or len(holdout)<4:
        parser.error('Use unique, disjoint practice and holdout seeds; at least four holdouts.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Use an empty practice directory to preserve evidence.')
    if args.max_candidates<1:parser.error('max-candidates must be positive')
    if args.request:
        request=parse_request(args.request)
        if request['kind']!='speed' or args.kind!='speed':parser.error('--request currently drives the speed curriculum')
        args.increase_percent=100*request['increase_fraction']
    base=TrialConfig(kind=args.kind,seconds=args.seconds)
    if args.kind=='speed':
        ref=json.loads((ROOT/'artifacts/takeover/contact-final-video.json').read_text())['cycles']
        hz=len(ref)/(ref[-1]['end']-ref[0]['start'])
        candidates=[replace(base,turn_hz=v) for v in speed_curriculum(hz,args.increase_percent/100)]
    elif args.kind=='duo':
        candidates=[replace(base,dx=.075,span=s,rope_length=l) for s,l in [(.448,.58),(.60,.73),(.65,.78),(.70,.83)]]
    else:candidates=[replace(base,dx=0.,graphite_y=.5,graphite_goal_y=.16,entry_controller='hop',entry_phase_reference='leader',entry_phase=v) for v in [0.,math.pi/2,math.pi,-math.pi/2]]
    summary=dict(kind=args.kind,backend='physical parameter search; not PPO retraining',training_seeds=train,holdout_seeds=holdout,
        promoted=False,attempts=[],status='PRACTICING')
    winner=None
    for k,candidate in enumerate(candidates[:args.max_candidates]):
        receipts=[]
        for seed in train:
            _,receipt=record_trial(replace(candidate,seed=seed),args.output_dir,f'candidate{k}-practice-{seed}')
            receipts.append(receipt)
        accepted=all(r['passed'] for r in receipts)
        summary['attempts'].append(dict(candidate=k,config=candidate.__dict__,practice_passed=accepted))
        write(args.output_dir/'summary.json',summary)
        if accepted:
            winner=candidate
            if args.kind!='speed':break
        elif args.kind=='speed':break  # Do not advance curriculum past a failed level.
    if winner is not None:
        receipts=[]
        for seed in holdout:
            _,receipt=record_trial(replace(winner,seed=seed),args.output_dir,f'locked-holdout-{seed}');receipts.append(receipt)
        gate=promotion_gate(train,receipts,args.kind)
        summary['gate']=gate;summary['promoted']=gate['promoted'];summary['validated_config']=winner.__dict__
        # A slower validated level must not be described as achieving the requested increase.
        if args.kind=='speed':summary['requested_speed_reached']=winner.turn_hz>=candidates[-1].turn_hz-1e-6 and gate['promoted']
        if gate['promoted']:write(args.output_dir/'champion.json',dict(config=winner.__dict__,gate=gate,evidence=receipts))
    summary['status']='CHAMPION' if summary['promoted'] else 'NEEDS_LEARNING'
    if args.kind=='speed' and summary['promoted'] and not summary.get('requested_speed_reached'):
        summary['status']='LOWER_LEVEL_VALIDATED_TARGET_NOT_REACHED'
    write(args.output_dir/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
