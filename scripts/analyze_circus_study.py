"""Summarize saved physics trials and plot their geometry/contact evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def summarize(path):
    d=json.loads(path.read_text());diag=d.get('diagnostics')
    if not diag:return None
    events=diag['contact_events'];samples=diag['samples'];start=d['entry']['started']
    relevant=[e for e in events if start is None or e['t']>=start]
    first={}
    for e in relevant:first.setdefault(e['pair'],e)
    streak=best=0
    for c in (d.get('joint_cycles') or {}).get('cycles',[]):
        streak=streak+1 if c['clean'] else 0;best=max(best,streak)
    rates={n:dict(clean=r['clean_skips'],cycles=r['full_revolutions'],reasons=dict(Counter(reason for c in r['cycles'] for reason in c['reasons']))) for n,r in d['jumpers'].items()}
    unstable={n:next((s['t'] for s in samples if s['t']>=(start if start is not None else 0) and s['robots'][n]['upright']<.8),None) for n in d['jumpers']}
    return dict(path=str(path),config=d['config'],passed=d['passed'],error=d['error'],entry=d['entry'],
        measured_hz=d['measured_hz'],scores=rates,joint_clean=(d.get('joint_cycles') or {}).get('clean_cycles'),
        longest_joint_clean_streak=best,first_relevant_contacts=first,first_tilt=unstable)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    args=p.parse_args();rows=[]
    for path in sorted(args.directory.rglob('*.json')):
        if 'protocols' in path.parts or path.name in ('summary.json','analysis.json'):continue
        d=json.loads(path.read_text())
        if isinstance(d,dict) and d.get('diagnostics'):
            rows.append(summarize(path))
    (args.directory/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        print(Path(r['path']).stem, 'joint',r['joint_clean'],'streak',r['longest_joint_clean_streak'],
              'entry',r['entry']['clean_cycles'], 'pass',r['passed'],'first tilt',r['first_tilt'])
    # matplotlib is a reporting dependency only, not required by the simulator.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(rows),1,figsize=(12,max(3,2.4*len(rows))),squeeze=False)
    for ax,r in zip(axes[:,0],rows):
        d=json.loads(Path(r['path']).read_text());samples=d['diagnostics']['samples'];t=[s['t'] for s in samples]
        for n,color in [('sky','#0087c9'),('graphite','#d67612')]:
            ax.plot(t,[s['robots'][n]['upright'] for s in samples],color=color,label=n,linewidth=1)
        ax.axhline(.8,color='gray',ls='--',lw=.7)
        for pair,e in r['first_relevant_contacts'].items():
            if pair=='graphite/sky':ax.axvline(e['t'],color='red',ls=':',label='jumper contact')
        if r['entry']['started'] is not None:ax.axvline(r['entry']['started'],color='green',ls='--',label='entry starts')
        ax.set(ylim=(-1.05,1.05),ylabel='Upright cosine',title=f"{Path(r['path']).stem}: {r['joint_clean']} clean shared cycles; passed={r['passed']}")
        ax.legend(loc='lower left',ncol=4,fontsize=8)
    axes[-1,0].set_xlabel('Simulation time (s)');fig.tight_layout();fig.savefig(args.directory/'upright-contact-timeline.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
