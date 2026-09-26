"""Render an audited physical run. Cameras never change the robot trajectory."""
import os,sys,json,hashlib,argparse,math
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ['PYOPENGL_PLATFORM']=os.environ['MUJOCO_GL']
import imageio.v2 as imageio
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from publish_contact_video import font

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
    report=json.loads((a.source/'audit.json').read_text())
    for name,expected in report['capture']['files'].items():
        assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==expected
    m=mujoco.MjModel.from_binary_path(str(a.source/'scene.mjb'));d=mujoco.MjData(m);states=np.load(a.source/'trajectory.npz')
    r=mujoco.Renderer(m,height=1080,width=1920);end=float(states['time'][-1]);shots=[]
    rolls=[x for x in report['audit']['rolls'] if x['clean']]
    if rolls:
        start=max(.02,rolls[0]['start']-.1);stop=min(end,rolls[0]['end']+.2)
        shots=[(states['time'][0],start,1.,'chase'),(start,stop,.5,'roll'),(stop,end,1.,'chase')]
    else:shots=[(states['time'][0],end,1.,'chase')]
    times=np.array([s['t'] for s in report['trace']]);decisions=[x for x in report['decisions'] if x.get('decision')]
    def frame(t,speed,kind):
        i=int(np.argmin(abs(states['time']-t)));d.qpos[:]=states['qpos'][i];d.qvel[:]=states['qvel'][i];d.ctrl[:]=states['ctrl'][i];d.time=float(states['time'][i]);mujoco.mj_forward(m,d)
        robot=d.body('duck/trunk_base').xpos
        cam=mujoco.MjvCamera();cam.lookat=robot+np.array([.22,0,.08]);cam.distance=1.1;cam.azimuth=145;cam.elevation=-22
        if kind=='chase' and 24<=t<26:
            ball=d.body('boss_ball').xpos
            cam.lookat=ball+np.array([.08,0,.02]);cam.distance=1.15;cam.azimuth=90;cam.elevation=-16
        if kind=='roll':cam.lookat=robot+np.array([.05,0,.08]);cam.distance=.85;cam.azimuth=90;cam.elevation=-12
        r.update_scene(d,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
        im=Image.fromarray(r.render());draw=ImageDraw.Draw(im,'RGBA')
        trace=report['trace'][min(len(times)-1,int(np.searchsorted(times,t)))];progress=np.clip(trace['position'][0]/8,0,1)
        draw.rectangle((0,0,1920,82),fill=(6,12,26,235));draw.text((36,19),'ROSCLAW  /  NEON ESCAPE',font=font(33,True),fill=(233,246,255))
        draw.text((1390,22),f'SCORE {trace["score"]:04d}',font=font(31,True),fill=(32,241,221))
        draw.rectangle((36,93,1884,99),fill=(28,48,65,220));draw.rectangle((36,93,36+1848*progress,99),fill=(20,239,218,255))
        label='JEV / RECORDED RUN' if report['brain']=='jev' else 'RULE BASELINE'
        recent=[x for x in decisions if x['t']<=t];last=recent[-1] if recent else None
        draw.rounded_rectangle((1450,122,1880,268),radius=13,fill=(8,17,32,220))
        draw.text((1472,137),label,font=font(26,True),fill=(54,235,220))
        if last:
            dec=last['decision'];draw.text((1472,180),f'{dec["latency_ms"]:.0f} ms  |  confidence {dec["confidence"]:.2f}',font=font(22),fill='white')
            draw.text((1472,218),last['status'].replace('_',' ').upper(),font=font(20),fill=(184,198,220))
        else:draw.text((1472,188),'Awaiting decision' if report['brain']=='jev' else 'No model calls',font=font(24),fill='white')
        draw.text((38,147),('BALL BLOCKED BY BAR' if 24<=t<26 and kind=='chase' else trace['action'].replace('_',' ')),font=font(38,True),fill=(250,242,230))
        if trace['hits']:draw.text((38,211),'CONTACT / RUN FAILED',font=font(28,True),fill=(255,77,97))
        elif any(x['end']<=t and x['clean'] for x in rolls):draw.text((38,211),'CLEAN ROLL  +200',font=font(28,True),fill=(23,242,195))
        draw.rectangle((0,1005,1920,1080),fill=(6,12,26,235))
        slow=f'{speed:g}x SLOW MOTION' if speed!=1 else '1x PHYSICAL TIME'
        draw.text((36,1026),f'SEED {report["seed"]}   |   {slow}   |   SIM {t:05.2f}s',font=font(25),fill=(194,217,235))
        draw.text((1050,1026),'MuJoCo physics 5 kHz  /  Motor policies 50 Hz',font=font(25),fill=(194,217,235))
        if t<2:
            draw.rounded_rectangle((220,380,1700,610),radius=20,fill=(5,13,25,225))
            draw.text((285,405),'A DECISION MODEL. A PHYSICAL BODY.',font=font(48,True),fill='white')
            draw.text((285,490),'Reach the finish. Roll, steer, and avoid contact.',font=font(36),fill=(35,240,218))
            draw.text((285,548),'Real API decisions. Joint-level motor policies. No robot teleportation.',font=font(25),fill=(195,213,230))
        return np.asarray(im)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        if a.preview:
            for j,t in enumerate([3.,rolls[0]['start']+1 if rolls else 10.,min(end,20.)]):
                Image.fromarray(frame(t,.5 if j==1 else 1.,'roll' if j==1 else 'chase')).save(a.output.parent/f'neon-preview-{j}.png')
            return
        fps=50;frames=0
        with imageio.get_writer(str(a.output),fps=fps,codec='libx264',macro_block_size=1,ffmpeg_params=['-crf','19','-preset','medium','-threads','4','-movflags','+faststart']) as writer:
            for start,stop,speed,kind in shots:
                count=int(math.ceil((stop-start)/speed*fps))
                for j in range(count):writer.append_data(frame(min(stop,start+j/fps*speed),speed,kind));frames+=1
                print('Rendered',kind,round(start,2),round(stop,2),flush=True)
            final=Image.fromarray(frame(end,1.,'chase'));draw=ImageDraw.Draw(final,'RGBA');draw.rounded_rectangle((270,380,1650,630),radius=20,fill=(5,13,25,235))
            draw.text((350,405),'FINISH / CLEAN RUN' if report['passed'] else 'ATTEMPT / NOT A CLEAN FINISH',font=font(48,True),fill=(40,238,210) if report['passed'] else (255,112,115))
            draw.text((350,490),f'{len(report["audit"]["cleared"])} obstacles cleared  |  {len(rolls)} verified rolls',font=font(35),fill='white')
            draw.text((350,555),'Prototype: gap jumping and general fall recovery are not yet validated.',font=font(25),fill=(197,215,230))
            for _ in range(150):writer.append_data(np.asarray(final));frames+=1
        Image.fromarray(frame(rolls[0]['start']+1 if rolls else 3,.5,'roll')).save(a.output.with_suffix('.jpg'),quality=92)
        manifest=dict(source=str(a.source),source_audit_sha256=hashlib.sha256((a.source/'audit.json').read_bytes()).hexdigest(),video_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),fps=fps,frames=frames,duration_s=frames/fps,resolution=[1920,1080],slow_motion='nearest actual saved state; no interpolation',source_capture_hz=report['capture']['hz'],brain=report['brain'],passed=report['passed'],shots=shots)
        a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(a.output,frames/fps,flush=True)
    finally:r.close()
if __name__=='__main__':main()
