"""Summarize recorded outcomes without dropping failed seeds."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'artifacts/neon-escape';rows=[]
for pattern,group in [('final-*/audit.json','development'),('holdout-jev-*/audit.json','fresh_seed_holdout')]:
 for p in sorted(root.glob(pattern)):
  d=json.loads(p.read_text());rolls=d['audit']['rolls']
  rows.append(dict(source=str(p.relative_to(root.parent.parent)),group=group,seed=d['seed'],brain=d['brain'],passed=d['passed'],termination=d['termination'],duration_s=d['duration_sim_s'],cleared=len(d['audit']['cleared']),contacts=d['audit']['hit'],clean_rolls=sum(x['clean'] for x in rolls),roll_min_upright_50hz=[min(s['upright'] for s in d['trace'] if roll['start']<=s['t']<=roll['end']) for roll in rolls],max_hazard_penetration_mm=d['audit']['max_hazard_penetration_m']*1000))
report=dict(note='Development seeds were exposed during exploration. Fresh seeds 9001/9002/9003 were evaluated after freezing the final controller, without adjusting it to their outcomes. Three trials per group are too few to establish reliability. Each actual roll also inverts the body; the strengthened inversion gate reproduces the filmed run exactly.',runs=rows)
(root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
