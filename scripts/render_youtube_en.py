"""English ROSClaw film; replays audited states and credits AI-assisted development."""
import argparse
import hashlib
import json
import math
import subprocess
import wave

from publish_contact_video import (
    CACHE, OUT, FPS, capture, load, font, writer, mujoco, np, Image, ImageDraw,
    imageio_ffmpeg,
)

# Output start/end, source start, playback speed, camera, title, explanation.
SHOTS = [
    (0,5,9,1,'wide','THREE ROBOTS. ONE ROPE.', 'Two turn. One jumps. One shared physics world.'),
    (5,11,12,.25,'mouth','THE ROBOTS DRIVE THE ROPE.', 'The rope ends attach to handles held in their mouths.'),
    (11,17,12,.25,'feet','WATCH THE FEET CLEAR IT.', 'Physical contact is enabled. Touching the jumper fails the cycle.'),
    (17,23,13,1,'wide','LEARNED MOTION. CLOSED-LOOP TIMING.', 'PPO motor policies + rope-phase feedback + contact-based evaluation.'),
    (23,28,14,1,'wide','442 / 516 CLEAN CYCLES  |  85.7%', '8 seeded runs; first 9s excluded. 5/8 runs pass; 3 remain below 80%.'),
    (28,32,16,1,'wide','ROSCLAW  /  OPEN-SOURCE PHYSICAL AI', 'AI-assisted development: GPT-6 Astra (medium reasoning).'),
]


def raw_frame(renderer,m,d,states,t,kind):
    idx=int(np.clip(round((t-states['time'][0])/.005),0,len(states['time'])-1))
    d.qpos[:]=states['qpos'][idx];d.qvel[:]=states['qvel'][idx];d.time=float(states['time'][idx])
    mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera()
    if kind=='mouth':
        camera.lookat=d.body('lavender/handle').xpos
        camera.distance=.27;camera.azimuth=-85;camera.elevation=-12
    elif kind=='feet':
        camera.lookat=(d.body('sky/ankle_left').xpos+d.body('sky/ankle_right').xpos)/2
        camera.distance=.33;camera.azimuth=-90;camera.elevation=-10
    else:
        camera.lookat=[0,0,.23];camera.distance=.96;camera.azimuth=-70;camera.elevation=-25
    renderer.update_scene(d,camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
    return Image.fromarray(renderer.render())


def overlay(im,shot,t):
    draw=ImageDraw.Draw(im);white=(247,246,239);muted=(190,204,216);gold=(255,211,92)
    draw.rectangle((0,0,1920,108),fill=(25,34,45))
    draw.rectangle((0,875,1920,1080),fill=(25,34,45))
    draw.text((60,30),'ROSCLAW  /  MICRODUCK',font=font(33,True),fill=white)
    draw.text((1440,36),'MUJOCO SIMULATION',font=font(25),fill=muted)
    draw.text((60,897),shot[5],font=font(45,True),fill=gold)
    draw.text((60,959),shot[6],font=font(29),fill=white)
    speed='0.25x SLOW MOTION' if shot[3]==.25 else '1x REAL-TIME PLAYBACK'
    draw.text((60,1025),speed+'  |  Audited simulation trajectory',font=font(23),fill=muted)
    footer='github.com/ros-claw/microduck' if shot[0]>=28 else 'Local ONNX control at 50 Hz'
    draw.text((1380,1025),footer,font=font(22),fill=muted)
    return np.asarray(im)


def audio_track(audit):
    sr=48000;audio=np.zeros(32*sr);rng=np.random.default_rng(16)
    events=[]
    for c in audit['crossing_samples']:
        if not c['overhead'] and (not events or c['t']-events[-1]>.12):events.append(c['t'])
    for start,end,src,speed,*_ in SHOTS:
        for event in events:
            t=start+(event-src)/speed
            if not start<=t<end:continue
            n=int(sr*(.12 if speed==1 else .25))
            noise=np.convolve(rng.normal(size=n),np.ones(13)/13,mode='same')
            pulse=.13*noise*np.sin(np.linspace(0,math.pi,n))**2
            k=round(t*sr);stop=min(len(audio),k+n)
            audio[k:stop]+=pulse[:stop-k]
    wav=CACHE/'youtube-en-foley.wav'
    with wave.open(str(wav),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes((np.clip(audio,-.9,.9)*32767).astype('<i2').tobytes())
    return wav


def main(preview=False):
    if not (CACHE/'trajectory.npz').exists():capture()
    m,d,states,audit=load();OUT.mkdir(exist_ok=True)
    renderer=mujoco.Renderer(m,height=1080,width=1920)
    try:
        if preview:
            for s in SHOTS:
                im=raw_frame(renderer,m,d,states,12.2,s[4])
                Image.fromarray(overlay(im,s,12.2)).save(CACHE/f'preview-en-{s[0]}.png')
            return
        silent=CACHE/'youtube-en-silent.mp4'
        with writer(silent) as video:
            for shot in SHOTS:
                for i in range((shot[1]-shot[0])*FPS):
                    t=shot[2]+i/FPS*shot[3]
                    video.append_data(overlay(raw_frame(renderer,m,d,states,t,shot[4]),shot,t))
                print('Finished segment',shot[0],flush=True)
        thumbnail=raw_frame(renderer,m,d,states,12.2,'wide')
        draw=ImageDraw.Draw(thumbnail)
        draw.rectangle((0,0,1920,240),fill=(25,34,45))
        draw.text((64,35),'ROSCLAW: ROBOTS LEARN TO SKIP',font=font(70,True),fill=(255,211,92))
        draw.text((64,143),'Built with GPT-6 Astra (medium)  |  MuJoCo simulation',font=font(37),fill=(247,246,239))
        thumbnail.save(OUT/'microduck_youtube_en_thumbnail.jpg',quality=95)
    finally:renderer.close()
    audio=audio_track(audit);output=OUT/'microduck_youtube_en.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(silent),'-i',str(audio),'-c:v','copy','-c:a','aac','-b:a','160k','-shortest','-movflags','+faststart',str(output)],check=True)
    result=dict(video=str(output.relative_to(OUT.parent)),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),language='en',duration_s=32,fps=50,size=[1920,1080],frames=1600,shots=SHOTS,audit='artifacts/presentation/audit.json',audio='Original procedural post-production foley; no narration or external music',ai_credit='GPT-6 Astra (medium): AI-assisted development, not runtime motor control')
    (CACHE/'youtube-en.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',action='store_true')
    main(parser.parse_args().preview)
