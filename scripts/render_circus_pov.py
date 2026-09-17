"""108-second English edit, including head-follow cameras and collision limitations."""
import argparse
import render_circus_details as base
from render_circus_details import *
SHOTS=[
 (8,'duo',11,.5,'wide','TWO JUMPERS. ONE PHYSICAL ROPE.','Four simulated Microduck robots coordinate through learned motor policies.'),
 (8,'duo',11,1,'front','WATCH THE RHYTHM / REAL TIME','An excerpt after startup. Both jumpers clear the same rotating rope.'),
 (12,'duo',12,.25,'pov_sky','JUMP ALONG / HEAD-FOLLOW CAMERA','Stabilized head-follow camera; the controller does not use camera vision.'),
 (10,'duo',12,.125,'mouth_left','HOW THE ROPE IS DRIVEN','Mouth-held handles transmit force through endpoint constraints.'),
 (12,'duo',12,.125,'feet_sky','SKY / FOLLOW THE ROPE UNDER BOTH FEET','Slow motion replays saved physical states, with no motion interpolation.'),
 (12,'duo',12,.125,'feet_graphite','GRAPHITE / CLEARANCE AND LANDING','A clean cycle requires no rope contact and a supported landing.'),
 (8,'duo',12,.5,'top','ROOM TO JUMP TOGETHER','0.80 m between turners  |  0.86 m rope  |  0.28 m between jumpers'),
 (10,'relay',13,.25,'pov_entry','ENTERING THE ROPE / HEAD-FOLLOW VIEW','An experimental entry attempt. This sequence does not pass the task.'),
 (10,'relay',14.5,.25,'entry','THE ENTRY IS NOT YET CLEAN','Watch the incoming robot meet the moving rope boundary.'),
 (10,'relay',15.3,.125,'entry_jaw','CONTACT IS REAL. PENETRATION IS ALSO REAL.','Red dots show contact. Soft constraints allow visible overlap at the jaw.'),
 (8,'duo',12,.5,'wide','WHAT WORKS / WHAT REMAINS','Static duo: 258/260 holdout cycles. Entry fails; rope-turner collisions excluded.'),
]
base.SHOTS=SHOTS
original_update=base.update_scene

def update_scene(renderer,m,d,kind):
 original_update(renderer,m,d,kind)
 if kind.startswith('pov_'):
  entry=kind=='pov_entry';name='graphite' if entry else 'sky'
  pos=d.body(name+'/jaw_soft').xpos.copy()+np.array([0,.055,.105] if entry else [.065,-.065,.09])
  target=np.array([-.14,0,.15]) if entry else d.body('graphite/trunk_base').xpos.copy()+np.array([0,0,.06])
  forward=target-pos;forward/=np.linalg.norm(forward)
  right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
  for camera in renderer.scene.camera:
   camera.pos[:]=pos;camera.forward[:]=forward;camera.up[:]=up
   camera.frustum_top*=1.7;camera.frustum_bottom*=1.7

# Reuse the exact trajectory loading and styling from the long film.
def main(preview=False):
 sources={name:load(name) for name in ('duo','relay')};renderers={name:mujoco.Renderer(v[0],height=1080,width=1920) for name,v in sources.items()}
 output=CACHE/'pov-silent.mp4';schedule=[];elapsed=0;frames=0;fps=50
 try:
  if preview:
   for i,shot in enumerate(SHOTS):
    m,d,states,audit=sources[shot[1]];t=shot[2]+shot[0]*shot[3]/2
    restore(m,d,states,t);ren=renderers[shot[1]];update_scene(ren,m,d,shot[4])
    Image.fromarray(decorate(ren.render(),shot,t,i+1,audit)).save(CACHE/f'pov-preview-{i:02d}.png')
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
 audio=ambient_track(elapsed);final=OUT/'microduck_circus_pov_en.mp4'
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(output),'-i',str(audio),'-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(final)],check=True)
 manifest=dict(video=str(final.relative_to(ROOT)),sha256=hashlib.sha256(final.read_bytes()).hexdigest(),duration_s=elapsed,fps=fps,frames=frames,size=[1920,1080],language='en',chapters=schedule,
  source_capture_hz=200,interpolation=False,simulation_only=True,rope_color='Orange for visibility; radius and geometry unchanged',audio='Original synthesized background ambience, not simulated contact audio',
  static_duo_holdout=dict(clean=258,total=260,runs_passed=4,runs=4,warmup_excluded_s=9),relay_passed=False,
  sources={name:v[3]['capture'] for name,v in sources.items()})
 (CACHE/'pov-film.json').write_text(json.dumps(manifest,indent=2)+'\n');print('COMPLETE',final,flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preview',action='store_true');main(p.parse_args().preview)
