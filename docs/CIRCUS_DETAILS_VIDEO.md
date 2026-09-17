# Four-duck slow-motion walkthrough / 四鸭慢动作细节视频

A **3 minute 20 second, 1920×1080, 50 fps** English walkthrough of the validated static-duo configuration and the remaining moving-entry failure.

这版重点是看清物理细节：完整试验、两端嘴部连接、两只跳跃鸭各自的双脚、正面协作、俯视间距，以及入场时下颌碰绳的位置。慢动作包含 0.5×、0.25× 与 0.125×，画面持续标注播放速度和原仿真时间。

[Full video / 完整视频](https://github.com/ros-claw/microduck/releases/download/circus-details-2026-09-17/microduck_circus_details_en.mp4) · [English subtitles](../out/microduck_circus_details_en.srt) · [中文字幕](../out/microduck_circus_details_zh-CN.srt)

## Chapters

```text
00:00 FOUR ROBOTS. TWO JUMPERS.
00:08 ONE COMPLETE PHYSICAL RUN
00:28 01 / THE LEFT MOUTH CONNECTION
00:48 02 / THE RIGHT MOUTH CONNECTION
01:08 03 / SKY: TAKEOFF, CLEARANCE, LANDING
01:32 04 / GRAPHITE: WATCH BOTH SOLES
01:56 05 / TWO ROBOTS, THE SAME ROPE
02:16 06 / SPACE FOR EVERY ROBOT
02:32 07 / MOVING ENTRY: STILL UNSOLVED
02:56 08 / WHY ENTRY FAILS: JAW CONTACT
03:08 258 / 260 SHARED CLEAN CYCLES
```

## What the film shows

- The static-duo source is the complete 20-second seed-0 simulation. Its per-jumper audit matches the original non-rendered trial exactly. Startup is included in the full-run chapter; scoring still excludes the first 9 seconds.
- The reported **258/260 shared clean cycles** come from the prior four independent 30-second runs, not from counting repeated camera shots as new tests.
- The moving-entry source is the previous 27-second relay attempt. Its failure record is unchanged: the rope touches Graphite's lower jaw at simulation time **15.5954 s**. The film does not claim successful entry or relay.
- Saved physical states are sampled at **200 Hz**. Camera replay restores those states without motion interpolation or generated animation. At 0.125×, 200 Hz capture supplies 25 distinct source states per output second; the 50 fps file repeats the nearest saved states where needed.
- Robot materials come from the assets. The rope is recolored orange for contrast; its radius and geometry are unchanged. Red dots in the failure close-up are rendering markers at detected geometric contact points, not additional physical objects.
- The quiet background sound is original procedural synthesis added in post-production. It is not simulated contact audio.

Recorded state replay is used only to render the already-audited simulation. It is not the controller or a way of moving robots during task evaluation.

## This round's control experiment / 本轮控制实验

接触记录确认了原入场尝试首先碰到下颌。本轮在同一入场配置中测试了颈部与头部各增加 0、0.35、0.50、0.65 rad 的动作目标，通过伺服执行，并缓慢施加、退出。没有修改机器人实际位姿或碰撞开关。

**四组均未通过。** 0.35 和 0.65 rad 还出现姿态失稳/身体触地；0.50 rad 没有解决碰绳。因此默认值保持 0，未替换已验证的双跳配置，也未声称本轮提高了物理成功率。单纯修改头部姿态不足以完成入场，需要继续研究全身运动、路线和绳相位的配合。

[Measured head-posture ablation](../artifacts/circus-film/entry-tuck-summary.json)

## Reproduce

Run from the repository root with the asset setup from the main README. Capture directories must not already exist. The binary scene and trajectory caches are generated locally and are not committed.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/capture_circus_film.py \
  --config configs/circus/duo_matched.json \
  --reference artifacts/circus-analysis/matched-video/duo-seed0.json \
  --output-dir artifacts/circus-film/duo

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/capture_circus_film.py \
  --config configs/circus/relay_roomy_experimental.json \
  --reference artifacts/circus-analysis/relay-roomy/relay-seed0.json \
  --output-dir artifacts/circus-film/relay

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/render_circus_details.py --preview
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/render_circus_details.py
```

Audit sources: [static duo](../artifacts/circus-film/duo/audit.json), [relay](../artifacts/circus-film/relay/audit.json). Film metadata and hashes: [manifest](../artifacts/circus-film/film.json).

Validation: 49 tests passed. The final 200-second file contains 10,000 frames, and a full video/audio decode completed without errors.
