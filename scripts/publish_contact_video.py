"""Render native-asset publication videos from one audited physics trajectory.

Capture at 200 Hz; replay saved states, never interpolated or animated joints.
The vertical edit labels its slow motion and uses the same source trajectory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave

os.environ.setdefault('MUJOCO_GL','egl')
os.environ['PYOPENGL_PLATFORM'] = os.environ['MUJOCO_GL']
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from microduck_lab.demos.honest_skip import run_honest_classic_skip
from microduck_lab.sim.skip_metrics import PhysicalSkipAudit

CACHE = ROOT/'artifacts/presentation'
OUT = ROOT/'out'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
WHITE = (247,246,239)
MUTED = (192,204,213)
GOLD = (255,211,92)
FPS = 50
_fonts = {}


def font(size,bold=False):
    key=(size,bold)
    if key not in _fonts:
        _fonts[key]=ImageFont.truetype(BOLD if bold else FONT,size,index=2)
    return _fonts[key]


def label(draw,xy,text,size,fill=WHITE,bold=False):
    draw.text(xy,text,font=font(size,bold),fill=fill,stroke_width=0)


def capture():
    audit=PhysicalSkipAudit()
    states=[]
    ctx={}
    steps=0
    def setup(m,d,info):
        m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON
        m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
        audit.setup(m,d,info)
        ctx['model']=m
    def observe(m,d,info,hopping):
        nonlocal steps
        audit(m,d,info,hopping)
        if steps%25==0:
            states.append((d.time,d.qpos.copy(),d.qvel.copy()))
        steps+=1
    run_honest_classic_skip(None,seconds=20,seed=0,render=False,
        turner_onnx=ROOT/'policies/turner_rope.onnx',hop_onnx=ROOT/'policies/ropehop_contact.onnx',
        rope_contacts='full',legacy_rope_offset=False,connect_timeconst=.004,
        rope_floor_timeconst=.0004,physics_dt=.0002,hop_start_delay=0,
        rope_joint_type='ball',rope_radius=.0015,rope_length=.58,jumper_y=0,
        settle_seconds=0,rope_initial_phase=1.57079632679,rope_velocity_limit=0,
        rope_pass_time_fn=lambda: audit.last_underfoot_crossing,max_turn_hz=3.1,
        asset_colors=True,presentation='studio',physics_observer=observe,model_setup=setup)
    result=audit.scorer.result()
    reference=json.loads((ROOT/'artifacts/takeover/contact-final-video.json').read_text())
    # Styling must not change the scored dynamics, including failed/partial events.
    assert result['cycles']==reference['cycles'], 'Visual changes altered the reference rollout'
    assert result['passed'] and result['clean_skips']==33
    result.update(source='contact-final-video.json',seed=0,seconds=20,
        native_asset_colors=True,reference_cycles_identical=True,capture_hz=200,
        hop_sha256=hashlib.sha256((ROOT/'policies/ropehop_contact.onnx').read_bytes()).hexdigest(),
        crossing_samples=audit.crossing_samples)
    CACHE.mkdir(parents=True,exist_ok=True)
    mujoco.mj_saveModel(ctx['model'],str(CACHE/'studio.mjb'),None)
    np.savez_compressed(CACHE/'trajectory.npz',time=np.array([s[0] for s in states]),
        qpos=np.stack([s[1] for s in states]),qvel=np.stack([s[2] for s in states]))
    (CACHE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CAPTURE VERIFIED: identical 33/33 physical cycles; native materials',flush=True)


def load():
    m=mujoco.MjModel.from_binary_path(str(CACHE/'studio.mjb'))
    # Contrast against native yellow feet. Color only; actual radius unchanged.
    for g in range(m.ngeom):
        if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith('rope/rope_s'):
            m.geom_rgba[g]=[1.,.30,.10,1.]
    return m,mujoco.MjData(m),np.load(CACHE/'trajectory.npz'),json.loads((CACHE/'audit.json').read_text())


def scene_frame(renderer,m,d,states,t,vertical=False,close=False,orbit=0.):
    idx=int(np.clip(round((t-states['time'][0])/.005),0,len(states['time'])-1))
    d.time=float(states['time'][idx]);d.qpos[:]=states['qpos'][idx];d.qvel[:]=states['qvel'][idx]
    mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera()
    camera.lookat=[0,0,.24 if not close else .13]
    camera.distance=(1.48 if vertical else .96) if not close else .85
    camera.azimuth=-70+orbit
    camera.elevation=-25
    renderer.update_scene(d,camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
    return Image.fromarray(renderer.render())


def decorate(im,t,vertical=False,segment='normal',audit=None,poster=False):
    draw=ImageDraw.Draw(im)
    w,h=im.size
    pad=64 if vertical else 72
    label(draw,(pad,58 if vertical else 35),'MICRODUCK',28 if vertical else 26,bold=True)
    label(draw,(pad,102 if vertical else 76),'物理仿真  /  PHYSICS SIMULATION',22 if vertical else 20,MUTED)
    if vertical:
        if segment=='slow':
            title,subtitle='看清这一跳。','绳从脚下过，碰到就不算成功。'
        else:
            title,subtitle='三只小鸭，一根跳绳。','两只甩绳，一只跳。'
        if poster:
            label(draw,(pad,225),'三只小鸭',100,bold=True)
            label(draw,(pad,355),'真的跳过去了。',88,GOLD,bold=True)
        else:
            label(draw,(pad,225),title,64,bold=True)
            label(draw,(pad,330),subtitle,34,MUTED)
        badge='0.25× 慢动作回放' if segment=='slow' else '真实碰撞 · 原生模型配色'
        label(draw,(pad,1390),badge,32,GOLD,bold=True)
        draw.line((pad,1470,w-pad,1470),fill=(115,131,143),width=2)
        if segment=='hook' and not poster:
            label(draw,(pad,1504),'真的跳过去了。',72,bold=True)
            label(draw,(pad,1630),'3 ROBOTS  /  1 ROPE',28,MUTED)
        elif segment=='slow' and not poster:
            label(draw,(pad,1504),'离地 → 过绳 → 落地',51,bold=True)
            label(draw,(pad,1610),'同一段物理轨迹，四分之一速回放。',29,MUTED)
        else:
            label(draw,(pad,1504),'85.7%',96,GOLD,bold=True)
            label(draw,(pad+390,1534),'442 / 516 圈',38,bold=True)
            label(draw,(pad,1630),'8 组仿真测试合计；每组启动 9 秒不计',27,MUTED)
            label(draw,(pad,1685),'其中 3 组仍低于 80% · 继续挑战更稳定的配合',25,MUTED)
    else:
        label(draw,(pad,900),'三只小鸭，一根跳绳。',46,bold=True)
        label(draw,(pad,971),'原生模型配色  ·  嘴部驱动  ·  真实碰撞',25,MUTED)
        completed=[c for c in audit['cycles'] if c['end']<=t]
        score=f'{sum(c["clean"] for c in completed):02d} / {len(completed):02d}'
        label(draw,(1530,900),score,58,GOLD,bold=True)
        label(draw,(1530,978),'本次合格圈 / 完整圈',23,MUTED)
        if t<9:
            label(draw,(1530,1030),'启动阶段 · 尚未计分',20,MUTED)
        else:
            label(draw,(1530,1030),f'连续实速回放  {t:05.2f}s',20,MUTED)
    return np.asarray(im)


def mapping(out_t):
    # Source times are explicit. Slow motion is replayed at 200 Hz / 4 = 50 fps.
    if out_t<4: return 9+out_t,'hook'
    if out_t<8: return 12.16+(out_t-4)*.25,'slow'
    return 13.2+(out_t-8),'normal'


def writer(path):
    return imageio.get_writer(str(path),fps=FPS,codec='libx264',macro_block_size=1,
        ffmpeg_params=['-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4'])


def render(preview=False):
    m,d,states,audit=load()
    OUT.mkdir(exist_ok=True)
    if preview:
        for vertical in (False,True):
            w,h=(1080,1920) if vertical else (1920,1080)
            renderer=mujoco.Renderer(m,height=h,width=w)
            for t,segment in ((12.2,'normal'),(12.2,'slow')) if vertical else ((12.2,'normal'),(12.,'normal')):
                frame=scene_frame(renderer,m,d,states,t,vertical,segment=='slow')
                imageio.imwrite(CACHE/f'preview-{vertical}-{segment}-{t}.png',decorate(frame,t,vertical,segment,audit))
            renderer.close()
        return
    renderer=mujoco.Renderer(m,height=1080,width=1920)
    with writer(OUT/'microduck_studio_full.mp4') as video:
        for i in range(20*FPS):
            t=i/FPS
            frame=scene_frame(renderer,m,d,states,t)
            video.append_data(decorate(frame,t,audit=audit))
            if i%250==0:print(f'FULL {i}/1000',flush=True)
    renderer.close()
    renderer=mujoco.Renderer(m,height=1920,width=1080)
    duration=14.8
    with writer(OUT/'microduck_social_silent.mp4') as video:
        for i in range(round(duration*FPS)):
            t,segment=mapping(i/FPS)
            orbit=0 if segment=='slow' else 7*math.sin(i/FPS*.18)
            frame=scene_frame(renderer,m,d,states,t,True,segment=='slow',orbit)
            video.append_data(decorate(frame,t,True,segment,audit))
            if i%250==0:print(f'SOCIAL {i}/{round(duration*FPS)}',flush=True)
    frame=scene_frame(renderer,m,d,states,12.20,True)
    Image.fromarray(decorate(frame,12.2,True,poster=True)).save(OUT/'microduck_cover.jpg',quality=95)
    renderer.close()
    sound_design(audit,duration)


def sound_design(audit,duration):
    """Original quiet procedural foley, NOT recorded physical/simulation audio."""
    sr=48000
    audio=np.zeros(round(sr*duration),dtype=np.float64)
    rng=np.random.default_rng(7)
    # One event per complete pair of actual underfoot crossings.
    passes=[]
    for c in audit['crossing_samples']:
        if not c['overhead'] and (not passes or c['t']-passes[-1]>.12):passes.append(c['t'])
    for src in passes:
        times=[]
        if 9<=src<13:times.append((src-9,1.))
        if 12.16<=src<13.16:times.append((4+(src-12.16)*4,.25))
        if 13.2<=src<20:times.append((8+src-13.2,1.))
        for out_t,speed in times:
            n=int(sr*(.10 if speed==1 else .24))
            noise=rng.normal(size=n)
            noise=np.convolve(noise,np.ones(13)/13,mode='same')
            envelope=np.sin(np.linspace(0,math.pi,n))**2
            pulse=.11*noise*envelope
            k=round((out_t-.025)*sr)
            start=max(0,k);end=min(len(audio),k+n)
            if end>start:audio[start:end]+=pulse[start-k:end-k]
    audio=np.clip(audio,-.8,.8)
    wav=CACHE/'foley.wav'
    with wave.open(str(wav),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes((audio*32767).astype('<i2').tobytes())
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
        '-i',str(OUT/'microduck_social_silent.mp4'),'-i',str(wav),'-c:v','copy',
        '-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',str(OUT/'microduck_social.mp4')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',action='store_true')
    p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    if a.capture or not (CACHE/'trajectory.npz').exists():capture()
    render(a.preview)
