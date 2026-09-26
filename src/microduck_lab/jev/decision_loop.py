"""Single-flight asynchronous decisions with monotonic deadlines and revalidation."""
import time,hashlib,json
from concurrent.futures import ThreadPoolExecutor

class DecisionLoop:
    def __init__(self,client,deadline=1.5,confidence=.5):
        self.client=client;self.deadline=deadline;self.confidence=confidence
        self.pool=ThreadPoolExecutor(max_workers=1);self.future=None;self.records=[];self.serial=0
    def request(self,state,candidates,sim_time):
        if self.future is not None:return False
        self.serial+=1;self.sent=time.monotonic();self.state=state;self.candidates=list(candidates);self.sim_time=sim_time
        self.future=self.pool.submit(self.client.decide,state,list(candidates));self.expired=False;return True
    def poll(self,legal,sim_time,locked=False):
        if self.future is None:return None
        age=time.monotonic()-self.sent
        if not self.future.done():
            if age>self.deadline and not self.expired:
                self.expired=True;self.records.append(dict(id=self.serial,t=sim_time,status='deadline',latency_ms=age*1000));return None if locked else 'BRAKE'
            return None
        try:result=self.future.result();error=None
        except Exception as e:result=None;error=type(e).__name__
        self.future=None
        status='accepted';action=result['action'] if result else 'BRAKE'
        if error:status='api_error'
        elif age>self.deadline:status='stale'
        elif locked:status='skill_committed'
        elif action not in legal:status='no_longer_legal'
        elif result['confidence']<self.confidence or result['need_system_two']>.7:status='uncertain'
        if status!='accepted':action='BRAKE'
        record=dict(id=self.serial,t=sim_time,request_sim_time=self.sim_time,state_hash=hashlib.sha256(json.dumps(self.state,sort_keys=True).encode()).hexdigest(),state=self.state,candidates=self.candidates,status=status,applied=None if locked else action,decision=result,error=error)
        self.records.append(record)
        return None if locked else action
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
