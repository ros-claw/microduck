# 发布素材

推荐上传 `out/microduck_social.mp4`，搭配 `out/microduck_cover.jpg`。

| 文件 | 用途 |
| --- | --- |
| `out/microduck_social.mp4` | 1080 × 1920、50 fps、14.8 秒竖版，含后期拟音 |
| `out/microduck_social_silent.mp4` | 同一竖版的无声版本，便于自行配乐 |
| `out/microduck_studio_full.mp4` | 1920 × 1080、50 fps、20 秒连续实速完整演示 |
| `out/microduck_cover.jpg` | 竖版封面，使用真实仿真帧 |

## 可直接使用的文案

**三只小鸭，一根跳绳：这次真的跳过去了！**

两只用嘴部连接带动绳子，一只找准时机起跳。最想让你看的是中间那段慢动作：离地、过绳、落地。

这是 MuJoCo 物理仿真。绳子与鸭子之间启用了真实碰撞，碰绳的圈不算成功。8 组测试共 516 圈，442 圈合格，合计 85.7%；每组最初 9 秒启动阶段不计，其中 3 组仍低于 80%，稳定性还在继续改进。

白色外壳、黄色嘴和脚、浅青色脚底，都是模型资产原本的配色。短片包含标注的四分之一速回放，声音为后期拟音。

#机器人 #强化学习 #物理仿真 #MuJoCo #MicroDuck

## 画面与剪辑

- 恢复模型原生材质，保留舵机、眼睛和脚底的细节；绳子使用珊瑚橙提高辨识度，未改变碰撞半径。
- 使用干净的摄影棚背景、主光和补光，主镜头朝向能看见跳跃者正脸的一侧。
- 竖版开头直接展示配合，随后用近景慢动作展示脚下过绳，结尾给出多组测试成绩与限制。
- 竖版 0–4 秒对应原轨迹 9–13 秒；4–8 秒对应 12.16–13.16 秒的 0.25× 回放；8–14.8 秒对应 13.2–20 秒。横版从启动到结束连续实速播放。
- 拟音为本项目用噪声合成的轻微绳声，按实际过绳事件对齐，不是仿真录音，也未使用外部音乐。

## 复现与核验

在仓库根目录运行：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/publish_contact_video.py --capture
```

需要仓库依赖、EGL 和 Noto Sans CJK 字体。第一次捕获将保存本地模型与 200 Hz 状态缓存；后续不带 `--capture` 可直接重新渲染。大型缓存被 `.gitignore` 排除。

`artifacts/presentation/audit.json` 保存本次核验：全部计分圈与 `artifacts/takeover/contact-final-video.json` 逐项一致，启动后 33/33 圈合格。回放直接读取仿真状态，不插值关节、不修改动作。多组成绩来源为 `artifacts/takeover/contact-summary.json`，不以这条成功演示替代多组验证。

原实验的默认外观保持兼容；新外观由 `asset_colors=True, presentation="studio"` 显式开启。
