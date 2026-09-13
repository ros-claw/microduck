"""Three tracked close-ups of the audited trajectory, at quarter speed."""
import argparse
import json
import hashlib

from publish_contact_video import (
    CACHE, OUT, FPS, capture, load, font, writer, mujoco, np, Image, ImageDraw,
)

SHOTS = [
    ('mouth', '嘴部连接：鸭子怎样带动绳子', ['lavender/handle'], .26, -85, -12),
    ('feet', '脚下过绳：离地、净空、落地', ['sky/ankle_left', 'sky/ankle_right'], .32, -90, -10),
    ('jumper', '完整起跳：看清身体与绳子的配合', ['sky/trunk_base'], .64, -70, -18),
]


def frame(renderer, model, data, states, t, shot):
    key,title,bodies,distance,azimuth,elevation=shot
    idx=int(np.clip(round((t-states['time'][0])/.005),0,len(states['time'])-1))
    data.qpos[:]=states['qpos'][idx]; data.qvel[:]=states['qvel'][idx]
    data.time=float(states['time'][idx])
    mujoco.mj_forward(model,data)
    camera=mujoco.MjvCamera()
    camera.lookat=np.mean([data.body(name).xpos for name in bodies],axis=0)
    if key=='jumper': camera.lookat[2]+=.055
    camera.distance=distance; camera.azimuth=azimuth; camera.elevation=elevation
    renderer.update_scene(data,camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]=False
    im=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,1280,96),fill=(28,37,48))
    draw.rectangle((0,644,1280,720),fill=(28,37,48))
    draw.text((36,16),title,font=font(34,True),fill=(247,246,239))
    draw.text((36,60),'MICRODUCK  /  物理仿真 · 原生模型材质',font=font(18),fill=(192,204,213))
    draw.text((36,666),'0.25× 慢动作 · 同一轨迹多角度回放',font=font(26,True),fill=(255,211,92))
    draw.text((950,671),f'仿真时间 {t:.2f}s',font=font(22),fill=(192,204,213))
    return np.asarray(im)


def main(preview=False):
    if not (CACHE/'trajectory.npz').exists(): capture()
    model,data,states,audit=load()
    renderer=mujoco.Renderer(model,height=720,width=1280)
    try:
        if preview:
            for shot in SHOTS:
                Image.fromarray(frame(renderer,model,data,states,12.2,shot)).save(CACHE/f'preview-closeup-{shot[0]}.png')
            return
        path=OUT/'microduck_closeups.mp4'
        with writer(path) as video:
            for shot in SHOTS:
                for i in range(6*FPS):
                    video.append_data(frame(renderer,model,data,states,12+i/FPS*.25,shot))
                print(f'Finished {shot[0]}',flush=True)
        metadata=dict(path=str(path.relative_to(OUT.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fps=FPS,frames=900,duration_s=18,size=[1280,720],speed=.25,audio=False,
            source_range_s=[12,13.5],shots=[s[0] for s in SHOTS],
            trajectory_hz=200,interpolation=False,audit='artifacts/presentation/audit.json')
        (CACHE/'closeups.json').write_text(json.dumps(metadata,indent=2)+'\n')
    finally:
        renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    main(parser.parse_args().preview)
