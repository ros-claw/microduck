"""Re-run the film protocols with exhaustive, read-only 5 kHz contact statistics."""
import json, sys, os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mujoco
from microduck_lab.circus import trial
from microduck_lab.circus.diagnostics import FormationDiagnostics

class FullContactAudit(FormationDiagnostics):
    def setup(self,m,d,info):
        super().setup(m,d,info)
        self.stats={};self.steps=0
        self.model_info=dict(timestep=float(m.opt.timestep),mocap_bodies=int(m.nmocap),body_exclusions=int(m.nexclude),
            rope_geoms=[dict(name=self.geom_names[g],radius_m=float(m.geom_size[g,0]),solref=m.geom_solref[g].tolist(),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g])) for g,o in enumerate(self.owners) if o=='rope'],
            jumper_collision_geoms=[dict(name=self.geom_names[g],type=int(m.geom_type[g]),solref=m.geom_solref[g].tolist(),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g])) for g,o in enumerate(self.owners) if o in ('sky','graphite') and (m.geom_contype[g] or m.geom_conaffinity[g])])
    def physics(self,m,d):
        super().physics(m,d);self.steps+=1
        for i,c in enumerate(d.contact):
            owners=[self.owners[int(g)] for g in (c.geom1,c.geom2)]
            if 'rope' not in owners or c.efc_address<0:continue
            key='/'.join(sorted(owners));s=self.stats.setdefault(key,dict(contact_samples=0,first_s=float(d.time),max_penetration_m=0.,max_normal_force_n=0.,normal_impulse_ns=0.,max_penetration_after_9s_m=0.))
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
            depth=max(0.,-float(c.dist));s['contact_samples']+=1
            s['normal_impulse_ns']+=float(force[0])*m.opt.timestep
            s['max_normal_force_n']=max(s['max_normal_force_n'],float(force[0]))
            if d.time>=9:s['max_penetration_after_9s_m']=max(s['max_penetration_after_9s_m'],depth)
            if depth>s['max_penetration_m']:
                s.update(max_penetration_m=depth,peak_s=float(d.time),peak_geoms=[self.geom_names[int(g)] for g in (c.geom1,c.geom2)],peak_solref=c.solref.tolist(),peak_solimp=c.solimp.tolist())
        self.warnings={str(mujoco.mjtWarning(i)):int(w.number) for i,w in enumerate(d.warning) if w.number}
    def result(self):
        r=super().result();r['full_contact_audit']=dict(steps=self.steps,model=self.model_info,pairs=self.stats,warnings=self.warnings);return r

def main():
    source=sys.argv[1];trial.FormationDiagnostics=FullContactAudit
    stiff='--stiff' in sys.argv
    if stiff:
        from microduck_lab.demos import honest_skip
        original=honest_skip.build_classic_world
        def build(*args,**kwargs):
            m,d,info=original(*args,**kwargs)
            for g in range(m.ngeom):
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
                if name.startswith('rope/rope_s'):
                    m.geom_priority[g]=1;m.geom_solref[g]=[.002,1.];m.geom_solimp[g]=[.95,.99,.001,.5,2.]
            return m,d,info
        honest_skip.build_classic_world=build
    config='duo_matched' if source=='duo' else 'relay_roomy_experimental'
    cfg=trial.TrialConfig(**json.loads((ROOT/f'configs/circus/{config}.json').read_text()))
    result=trial.run_trial(cfg)
    ref=json.loads((ROOT/f'artifacts/circus-film/{source}/audit.json').read_text())
    identical=result['jumpers']==ref['jumpers'] and result['relay']==ref['relay']
    if not stiff:assert identical, 'Observer altered reference outcome'
    report=dict(source=source,stiff_experiment=stiff,cycles_identical=identical,passed=result['passed'],**result['diagnostics']['full_contact_audit'])
    out=ROOT/f'artifacts/physical-fidelity/{source}{"-stiff" if stiff else ""}.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(source=source,pairs=report['pairs'],warnings=report['warnings']),indent=2),flush=True)
if __name__=='__main__':main()
