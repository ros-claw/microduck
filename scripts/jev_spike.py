"""Live latency/probability benchmark. No fabricated response or confidence."""
import sys,json,time,argparse,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.jev.client import JevClient
STATE=dict(robot=dict(speed_m_s=.4,upright=True,airborne=False),hazards=[dict(type='bar',distance_m=.7,height_m=.14)],left_lane='blocked',right_lane='clear',skills=dict(jump='ready',roulade='ready'),objective='reach_finish_safely')
ACTIONS=['KEEP_RUNNING','JUMP','ROULADE','DODGE_RIGHT','BRAKE']
def main():
    p=argparse.ArgumentParser();p.add_argument('--queries',type=int,default=1000);p.add_argument('--workers',type=int,default=1);p.add_argument('--output',type=Path,default=ROOT/'artifacts/neon-escape/jev-spike.json');a=p.parse_args()
    client=JevClient(timeout=3.);started=time.monotonic()
    def query(i):
        t=time.monotonic()
        try:return dict(index=i,ok=True,**client.decide(STATE,ACTIONS))
        except Exception as e:return dict(index=i,ok=False,error=type(e).__name__,status=getattr(e,'code',None),latency_ms=(time.monotonic()-t)*1000)
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(query,range(a.queries)):
            records.append(result)
            if len(records)%50==0:print('Jev queries',len(records),flush=True)
    good=[r for r in records if r['ok']];times=sorted(r['latency_ms'] for r in good)
    percentile=lambda p:times[min(len(times)-1,round((len(times)-1)*p))] if times else None
    summary=dict(requests=len(records),successes=len(good),errors=len(records)-len(good),concurrency=a.workers,wall_seconds=time.monotonic()-started,latency_ms={f'p{p}':percentile(p/100) for p in (50,90,95,99)},input_tokens=sum(r['usage'].get('input_tokens',0) for r in good),models=sorted(set(r['model'] for r in good)))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(summary=summary,state=STATE,candidates=ACTIONS,records=records),indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
