"""Controlled four-duck geometry/cadence study; all trials retain physical contacts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.circus.trial import TrialConfig
from circus_director import record_trial


def cases(seconds=20, group="geometry"):
    if group=="matched-holdout":
        base=TrialConfig(kind="duo",seconds=seconds,diagnostics=True,span=.80,rope_length=.86,dx=.14)
        return {f"matched-seed{seed}":replace(base,seed=seed) for seed in (1101,1102,1103,1104)}
    if group=="length":
        base=TrialConfig(kind="duo",seconds=seconds,diagnostics=True,span=.80,dx=.14)
        return {f"length{length:.2f}-seed{seed}":replace(base,seed=seed,rope_length=length)
                for length in (.82,.86,.90) for seed in (0,1)}
    if group=="phase":
        base=TrialConfig(kind="duo",seconds=seconds,diagnostics=True,span=.80,rope_length=.93,dx=.14)
        return {f"{mode}-seed{seed}":replace(base,seed=seed,hop_phase_mode="local" if mode=="local" else "midpoint",continuous_turn_phase=mode=="integrated")
                for mode in ("midpoint","local","integrated") for seed in (0,1)}
    if group=="holdout":
        base=TrialConfig(kind="duo",seconds=seconds,diagnostics=True,span=.80,rope_length=.93,dx=.14)
        return {f"roomy-seed{seed}":replace(base,seed=seed) for seed in (1001,1002,1003,1004)}
    if group=="roomy-entry":
        base=TrialConfig(kind="entry",seconds=seconds,diagnostics=True,span=.80,rope_length=.93,dx=.14,graphite_y=-.65,entry_controller="hop",entry_phase_reference="shared")
        return {
            "roomy_entry_direct":base,
            "roomy_entry_prepared":replace(base,entry_speed=.12,entry_prehop=2.),
            "roomy_entry_pi":replace(base,entry_speed=.12,entry_prehop=2.,entry_phase=3.141592653589793),
        }
    if group=="entry":
        base=TrialConfig(kind="entry",seconds=seconds,diagnostics=True,span=.60,rope_length=.73,dx=.10,graphite_y=-.6,entry_controller="hop",entry_phase_reference="shared")
        return {
            "entry_direct":base,
            "entry_ramp":replace(base,entry_speed=.12),
            "entry_prepared":replace(base,entry_speed=.12,entry_prehop=2.),
            "entry_prepared_pi":replace(base,entry_speed=.12,entry_prehop=2.,entry_phase=3.141592653589793),
            "entry_integrated_clock":replace(base,entry_speed=.12,entry_prehop=2.,continuous_turn_phase=True),
            "old_entry_chronology":replace(base,span=.448,rope_length=.58,dx=0.,graphite_y=.5,graphite_goal_y=.16,entry_phase=3.141592653589793,entry_phase_reference="leader"),
        }
    base=TrialConfig(kind='duo',seconds=seconds,diagnostics=True,dx=.075)
    return {
        'original':base,
        'longer_only':replace(base,rope_length=.73),
        'wider_only':replace(base,span=.60),
        'wider_and_longer':replace(base,span=.60,rope_length=.73),
        'roomy_spacing':replace(base,span=.80,rope_length=.93,dx=.14),
        'roomy_slow':replace(base,span=.80,rope_length=.93,dx=.14,turn_hz=2.8),
        'roomy_fixed_rate':replace(base,span=.80,rope_length=.93,dx=.14,fixed_turn_rate=True),
        'tandem_roomy':replace(base,span=.60,rope_length=.73,dx=0,sky_y=-.12,duo_y=.12),
    }


def run(item):
    name,cfg,directory=item
    result,_=record_trial(cfg,directory/name,name)
    return name,dict(config=result['config'],passed=result['passed'],error=result['error'],
        measured_hz=result['measured_hz'],joint=result['joint_cycles'],
        scores={n:{k:v for k,v in score.items() if k!='cycles'} for n,score in result['jumpers'].items()},
        initial_collision_bounds=result['diagnostics']['initial_collision_bounds'],
        minimum_separation=result['diagnostics']['minimum_root_xy_separation_m'],
        first_contact_by_pair={pair:next(e for e in result['diagnostics']['contact_events'] if e['pair']==pair)
            for pair in sorted({e['pair'] for e in result['diagnostics']['contact_events']})})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--group',choices=['geometry','entry','holdout','roomy-entry','phase','length','matched-holdout'],default='geometry')
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.output_dir.exists():p.error('Use a fresh directory')
    args.output_dir.mkdir(parents=True)
    summary={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name,result in pool.map(run,[(name,cfg,args.output_dir) for name,cfg in cases(args.seconds,args.group).items()]):
            summary[name]=result
            (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(name,result['passed'],result['joint']['clean_cycles'],result['joint']['full_cycles'],flush=True)


if __name__=='__main__':main()
