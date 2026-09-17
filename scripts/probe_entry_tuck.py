"""Bounded head-posture ablation; rope and robot contacts remain enabled."""
import os,sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.circus.trial import TrialConfig
from circus_director import record_trial

def run(tuck):
 c=json.loads((ROOT/'configs/circus/relay_roomy_experimental.json').read_text());c.update(kind='entry',seconds=25,entry_head_tuck=tuck)
 return record_trial(TrialConfig(**c),ROOT/f'artifacts/circus-film/entry-tuck-{tuck:.2f}',f'entry-{tuck:.2f}')[0]['passed']
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=2) as p:print(list(p.map(run,[0.,.35,.5,.65])),flush=True)
