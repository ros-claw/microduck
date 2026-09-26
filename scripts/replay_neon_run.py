"""Re-simulate recorded Jev decisions; assert original trace, contacts and skills.

This does not call Jev again or assign robot poses. It is an explicit recording
replay for reproducible rendering, not another independent evaluation run.
"""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.demos import neon_run

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=json.loads((a.source/'audit.json').read_text())
    if ref['brain']!='jev':raise ValueError('A recorded Jev run is required')
    class RecordedLoop:
        def __init__(self,*args):self.records=copy.deepcopy(ref['decisions']);self.index=0
        def request(self,*args):return True
        def poll(self,legal,sim_time,locked=False):
            if self.index>=len(self.records) or self.records[self.index]['t']>sim_time+1e-8:return None
            event=self.records[self.index];self.index+=1
            if event['status']=='deadline':return None if locked else 'BRAKE'
            return event['applied']
        def close(self):pass
    neon_run.DecisionLoop=RecordedLoop;neon_run.JevClient=lambda **kwargs:None
    neon_run.time.sleep=lambda duration:None
    result=neon_run.run(seed=ref['seed'],seconds=ref['duration_sim_s']+.02,brain='jev',realtime=True,output=a.output,capture=True,deadline=ref['deadline_s'],confidence=ref['confidence_threshold'])
    assert result['trace']==ref['trace'],'Replay trace changed'
    assert result['audit']==ref['audit'],'Replay physics audit changed'
    assert result['skill_events']==ref['skill_events'],'Replay skill transitions changed'
    ref['capture']=result['capture'];ref['capture'].update(replayed_from=str(a.source),trace_identical=True,contacts_identical=True,live_api_calls_during_replay=0)
    (a.output/'audit.json').write_text(json.dumps(ref,indent=2)+'\n');print('VERIFIED PHYSICAL REPLAY',a.output,flush=True)
if __name__=='__main__':main()
