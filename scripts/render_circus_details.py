"""Long English detail film: audited 200 Hz states, original materials, labeled slow motion."""
import argparse,hashlib,json,math,os,subprocess,wave
import imageio.v2 as imageio
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ['PYOPENGL_PLATFORM']=os.environ['MUJOCO_GL']
from publish_contact_video import ROOT,font,mujoco,np,Image,ImageDraw,imageio_ffmpeg
CACHE=ROOT/'artifacts/circus-film';OUT=ROOT/'out'
# seconds, source, source start, playback speed, camera, heading, explanation
SHOTS=[
 (8,'duo',11,.5,'wide','FOUR ROBOTS. TWO JUMPERS.','A close look at cooperative rope skipping in MuJoCo.'),
 (20,'duo',0,1,'wide','ONE COMPLETE PHYSICAL RUN','Startup is included. Scoring begins after simulation time 9 seconds.'),
 (20,'duo',12,.125,'mouth_left','01 / THE LEFT MOUTH CONNECTION','The rope follows a mouth-held handle driven by the robot.'),
 (20,'duo',12,.125,'mouth_right','02 / THE RIGHT MOUTH CONNECTION','Both ends are physically attached. No motion-capture carriers.'),
 (24,'duo',12,.125,'feet_sky','03 / SKY: TAKEOFF, CLEARANCE, LANDING','Orange is the rope; the dark line on the floor is its shadow.'),
 (24,'duo',12,.125,'feet_graphite','04 / GRAPHITE: WATCH BOTH SOLES','A clean cycle requires sole clearance and a supported landing.'),
 (20,'duo',12,.25,'front','05 / TWO ROBOTS, THE SAME ROPE','Both jumpers must pass the same full cycle for it to count.'),
 (16,'duo',12,.25,'top','06 / SPACE FOR EVERY ROBOT','Turner spacing 0.80 m  |  Rope 0.86 m  |  Jumper spacing 0.28 m'),
 (24,'relay',12,.25,'entry','07 / MOVING ENTRY: STILL UNSOLVED','This attempt touches the rope during entry and is graded FAIL.'),
 (12,'relay',15.1,.125,'entry_jaw','08 / WHY ENTRY FAILS: JAW CONTACT','The rope touches the lower jaw at 15.595s. Red dots mark geometric contact.'),
 (12,'duo',12,.5,'wide','258 / 260 SHARED CLEAN CYCLES','Four independent 30-second static-duo runs; first 9s excluded in each.'),
]


def load(source):
 folder=CACHE/source
 audit=json.loads((folder/'audit.json').read_text())
 if not audit['capture']['cycles_identical']:raise ValueError('Capture lacks matching physical audit')
 for filename,digest in audit['capture']['files'].items():
  if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Captured source checksum mismatch: '+filename)
 m=mujoco.MjModel.from_binary_path(str(folder/'scene.mjb'))
 # Presentation only. Never enlarge the rope collision or visual geometry.
 for g in range(m.ngeom):
  if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith('rope/rope_s'):
   m.geom_rgba[g]=[1.,.26,.08,1.]
 return m,mujoco.MjData(m),np.load(folder/'trajectory.npz'),audit


def restore(m,d,states,t):
 index=int(np.argmin(abs(states['time']-t)))
 if t<states['time'][0]-.006 or t>states['time'][-1]+.006:raise ValueError('Shot exceeds captured trajectory')
 d.time=float(states['time'][index]);d.qpos[:]=states['qpos'][index];d.qvel[:]=states['qvel'][index];d.ctrl[:]=states['ctrl'][index]
 mujoco.mj_forward(m,d)
 return index


def camera_for(d,kind):
 camera=mujoco.MjvCamera();camera.lookat=[0,0,.19];camera.distance=1.24;camera.azimuth=-75;camera.elevation=-22
 if kind.startswith('mouth_'):
  name='lavender' if kind.endswith('left') else 'cream'
  camera.lookat=d.body(name+'/handle').xpos.copy();camera.lookat[2]+=.005
  camera.distance=.28;camera.azimuth=-90;camera.elevation=-10
 elif kind.startswith('feet_'):
  name=kind.removeprefix('feet_')
  camera.lookat=(d.body(name+'/ankle_left').xpos+d.body(name+'/ankle_right').xpos)/2
  camera.lookat[2]=max(.035,camera.lookat[2]-.018)
  camera.distance=.36;camera.azimuth=-135 if name=='sky' else -45;camera.elevation=-14
 elif kind=='front':
  camera.lookat=(d.body('sky/trunk_base').xpos+d.body('graphite/trunk_base').xpos)/2
  camera.lookat[2]+=.06;camera.distance=.78;camera.azimuth=-90;camera.elevation=-12
 elif kind=='top':
  camera.lookat=[0,0,.10];camera.distance=1.18;camera.azimuth=-90;camera.elevation=-72
 elif kind=='entry_jaw':
  camera.lookat=d.body('graphite/jaw_soft').xpos.copy();camera.distance=.32;camera.azimuth=-90;camera.elevation=-10
 elif kind=='entry':
  camera.lookat=[0,-.16,.17];camera.distance=1.55;camera.azimuth=-60;camera.elevation=-30
 return camera


def update_scene(renderer,m,d,kind):
 renderer.update_scene(d,camera=camera_for(d,kind))
 renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
 if kind=='entry_jaw':
  for c in d.contact:
   names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '' for g in (c.geom1,c.geom2)]
   if c.dist<=0 and any(n.startswith('rope/') for n in names) and any(n.startswith('graphite/') for n in names):
    scene=renderer.scene
    if scene.ngeom>=scene.maxgeom:break
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.003,.003,.003]),c.pos,np.eye(3).ravel(),np.array([1.,.02,.02,.9]))
    scene.ngeom+=1


def decorate(raw,shot,t,chapter,audit):
 im=Image.fromarray(raw);draw=ImageDraw.Draw(im)
 white=(245,246,240);muted=(193,207,217);gold=(255,211,92)
 draw.rectangle((0,0,1920,106),fill=(22,31,42));draw.rectangle((0,868,1920,1080),fill=(22,31,42))
 draw.text((54,24),'ROSCLAW / MICRODUCK',font=font(34,True),fill=white)
 draw.text((1330,33),f'{chapter:02d} / {len(SHOTS):02d}   MUJOCO SIMULATION',font=font(23),fill=muted)
 draw.text((54,886),shot[5],font=font(39,True),fill=gold)
 draw.text((54,943),shot[6],font=font(27),fill=white)
 speed=shot[3];badge='REAL TIME / 1x' if speed==1 else f'SLOW MOTION / {speed:g}x'
 draw.text((54,1019),badge+'  |  Saved physical states; no motion interpolation',font=font(23),fill=muted)
 draw.text((1510,1019),f'SOURCE {t:05.2f}s',font=font(23),fill=muted)
 if shot[1]=='relay':
  verdict='ENTRY ATTEMPT' if t<15.5954 else 'FAIL: ROPE CONTACT'
  draw.rounded_rectangle((54,129,485,183),radius=9,fill=(110,35,35) if t>=15.5954 else (30,50,65))
  draw.text((72,139),verdict,font=font(24,True),fill=white)
 elif t<9:
  draw.text((54,133),'STARTUP / NOT SCORED',font=font(25,True),fill=gold)
 else:
  completed=[c for c in audit['joint_cycles']['cycles'] if c['end']<=t]
  draw.text((54,133),f'SHARED CLEAN CYCLES  {sum(c["clean"] for c in completed)} / {len(completed)}',font=font(24,True),fill=gold)
 return np.asarray(im)


def ambient_track(duration):
 sr=48000;t=np.arange(round(sr*duration))/sr
 # Original quiet synthesis, explicitly post-production, not physics audio.
 y=np.zeros_like(t)
 for f,gain in [(110,.014),(164.8138,.008),(220,.005),(261.6256,.003)]:
  y+=gain*np.sin(2*np.pi*f*t)*(.8+.2*np.sin(2*np.pi*.025*t))
 y*=np.minimum(1,t/3)*np.minimum(1,(duration-t)/4)
 path=CACHE/'ambient.wav'
 with wave.open(str(path),'wb') as w:
  w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes((y*32767).astype('<i2').tobytes())
 return path


def main(preview=False):
 sources={name:load(name) for name in ('duo','relay')};renderers={name:mujoco.Renderer(v[0],height=1080,width=1920) for name,v in sources.items()}
 output=CACHE/'details-silent.mp4';schedule=[];elapsed=0;frames=0;fps=50
 try:
  if preview:
   for i,shot in enumerate(SHOTS):
    m,d,states,audit=sources[shot[1]];t=15.60 if shot[1]=='relay' else 12.20
    restore(m,d,states,t);ren=renderers[shot[1]];update_scene(ren,m,d,shot[4])
    Image.fromarray(decorate(ren.render(),shot,t,i+1,audit)).save(CACHE/f'preview-{i:02d}.png')
   return
  with imageio.get_writer(str(output),fps=fps,codec='libx264',macro_block_size=1,ffmpeg_params=['-crf','20','-preset','medium','-pix_fmt','yuv420p','-threads','4']) as video:
   for i,shot in enumerate(SHOTS):
    m,d,states,audit=sources[shot[1]];ren=renderers[shot[1]]
    for j in range(shot[0]*fps):
     t=shot[2]+j/fps*shot[3];index=restore(m,d,states,t)
     update_scene(ren,m,d,shot[4])
     video.append_data(decorate(ren.render(),shot,t,i+1,audit));frames+=1
    schedule.append(dict(start_s=elapsed,end_s=elapsed+shot[0],source=shot[1],source_start_s=shot[2],speed=shot[3],camera=shot[4],title=shot[5]));elapsed+=shot[0]
    print('RENDERED',i+1,shot[4],elapsed,'seconds',flush=True)
 finally:
  for r in renderers.values():r.close()
 audio=ambient_track(elapsed);final=OUT/'microduck_circus_details_en.mp4'
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(output),'-i',str(audio),'-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(final)],check=True)
 manifest=dict(video=str(final.relative_to(ROOT)),sha256=hashlib.sha256(final.read_bytes()).hexdigest(),duration_s=elapsed,fps=fps,frames=frames,size=[1920,1080],language='en',chapters=schedule,
  source_capture_hz=200,interpolation=False,simulation_only=True,rope_color='Orange for visibility; radius and geometry unchanged',audio='Original synthesized background ambience, not simulated contact audio',
  static_duo_holdout=dict(clean=258,total=260,runs_passed=4,runs=4,warmup_excluded_s=9),relay_passed=False,
  sources={name:v[3]['capture'] for name,v in sources.items()})
 (CACHE/'film.json').write_text(json.dumps(manifest,indent=2)+'\n');print('COMPLETE',final,flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preview',action='store_true');main(p.parse_args().preview)
