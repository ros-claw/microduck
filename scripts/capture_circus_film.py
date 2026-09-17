"""Capture actual 200 Hz physical states for slow-motion camera replay."""
import argparse,json,os,sys,hashlib
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.circus.trial import TrialConfig,run_trial,PROTOCOL_SOURCE
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
c=json.loads(a.config.read_text());result=run_trial(TrialConfig(**c),capture_dir=a.output_dir)
reference=json.loads(a.reference.read_text());assert result['jumpers']==reference['jumpers'],'Capture changed audited cycles'
assert result['relay']==reference['relay'],'Capture changed relay verdict'
result['capture']=dict(hz=200,reference=str(a.reference),cycles_identical=True,interpolation=False,
 files={name:hashlib.sha256((a.output_dir/name).read_bytes()).hexdigest() for name in ['scene.mjb','trajectory.npz']})
(a.output_dir/'audit.json').write_text(json.dumps(result,indent=2)+'\n');(a.output_dir/'protocol.json').write_text(json.dumps(PROTOCOL_SOURCE,indent=2)+'\n')
print('CAPTURE VERIFIED',a.output_dir,result['passed'],flush=True)
