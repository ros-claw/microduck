# Short film: head-follow views and a physical-contact audit

[Watch / download the 1:48 English film](https://github.com/ros-claw/microduck/releases/download/circus-pov-2026-09-17/microduck_circus_pov_en.mp4)

108 seconds · 1920×1080 · 50 fps · English on-screen explanations.

The edit adds stabilized head-follow views from beside the jumping robot and above the entering robot. These are presentation cameras with world-aligned position offsets and a look-at target, **not actual robot cameras or a vision-based controller**. Offsets keep the lens outside the robot mesh; camera orientation does not reproduce neck rotation. Original asset materials are retained. Slow motion uses existing 200 Hz physical states without motion interpolation or trajectory changes.

The film includes real-time jumping, both feet in slow motion, a mouth attachment close-up, a formation overview, an entry attempt, and the jaw-contact failure. It uses selected excerpts after startup; it is not a new full-run evaluation. The 258/260 score refers to the previously reported four static-duo holdout runs, excluding the first 9 seconds of each run.

**The model has substantial penetration during impacts.** Rope–turner and rope self-collisions are disabled. The relay entry fails. Read the [physical fidelity audit](PHYSICAL_FIDELITY_AUDIT.md) before describing this as fully realistic contact simulation.

| Film time | View |
| --- | --- |
| 0:00–0:08 | Four-robot overview |
| 0:08–0:16 | Real-time shared jumping |
| 0:16–0:28 | Jumper head-follow view |
| 0:28–0:38 | Mouth-held rope connection |
| 0:38–0:50 | Sky foot clearance, 0.125× |
| 0:50–1:02 | Graphite foot clearance, 0.125× |
| 1:02–1:10 | Formation overhead |
| 1:10–1:20 | Entry head-follow view |
| 1:20–1:30 | Entry external view |
| 1:30–1:40 | Failed jaw contact, 0.125× |
| 1:40–1:48 | Results and limitations |

## 中文说明

这版为 **1 分 48 秒**，新增跳绳鸭附近与入场鸭上方的稳定头部跟随镜头，并保留脚底慢动作、嘴部连接和失败接触特写。相机位置带有避开模型的偏移，朝向跟踪目标，不复现头部旋转，也不代表机器人具备视觉控制能力。

视频是原有物理轨迹的选段重放，不是新增独立验证。258/260 仅指此前四组原地双跳验证，每组排除前 9 秒启动。**碰绳时存在明显穿入；绳—甩绳鸭和绳自碰撞被关闭；移动入场仍失败。** 详见[物理真实性复核](PHYSICAL_FIDELITY_AUDIT.md)。

## Reproduction

Capture the original states using the commands in [the long-film notes](CIRCUS_DETAILS_VIDEO.md), then run:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/render_circus_pov.py
MUJOCO_GL=egl .venv/bin/python scripts/render_contact_comparison.py
```

[Render manifest](../artifacts/circus-film/pov-film.json) · [Contact audit script](../scripts/audit_circus_contacts.py)
