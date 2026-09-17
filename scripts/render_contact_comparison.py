"""Compare original visual and collision geometry at a recorded failure. No simulation edits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_circus_details as f
m,d,s,a=f.load('relay');idx=f.restore(m,d,s,15.9158)
r=f.mujoco.Renderer(m,height=720,width=1280)
frames=[]
for collision in [False,True]:
 opt=f.mujoco.MjvOption()
 if collision:opt.geomgroup[2]=0;opt.geomgroup[3]=1
 r.update_scene(d,camera=f.camera_for(d,'entry_jaw'),scene_option=opt)
 im=f.Image.fromarray(r.render());draw=f.ImageDraw.Draw(im)
 draw.rectangle((0,0,1280,65),fill=(22,31,42))
 draw.text((24,15),'COLLISION GEOMETRY' if collision else 'RENDERED ASSET',font=f.font(30,True),fill='white')
 frames.append(im)
r.close()
canvas=f.Image.new('RGB',(2560,800),(22,31,42))
for i,im in enumerate(frames):canvas.paste(im,(i*1280,0))
f.ImageDraw.Draw(canvas).text((24,742),f'RELAY FAILURE / saved state {s["time"][idx]:.4f}s / max penetration at 5 kHz: 23.62 mm / diagnostic rendering only',font=f.font(27),fill='white')
canvas.save(f.ROOT/'artifacts/physical-fidelity/jaw-contact-comparison.jpg',quality=92)
