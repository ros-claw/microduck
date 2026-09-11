# 绳位反馈训练（2026-09-11）

**当前结果：** 最终 3.1 Hz 上限配置在 8 组、每组 30 秒的柔性绳真实接触测试中，
共 **442/516 圈通过，合计 85.7%**；每组前 9 秒为启动窗口。5 组全通过，
另 3 组仍低于 80%，不能保证所有初始状态都稳定。
原始逐圈数据入口：[contact-summary.json](../artifacts/takeover/contact-summary.json)。
直接复现：`scripts/contact_skip.sh`。

完整视频：[out/contact_skip.mp4](../out/contact_skip.mp4)，20 秒、500 帧、
25 fps、960×544。最终配置 seed 0 的录像审计为 33/33；未裁掉启动阶段。
已查看脚下穿越和过顶帧，镜头能容纳整个绳圈。录像只是其中一组，批次结论
仍以全部 8 组数据为准。

目标仍是嘴连柔性绳、真实接触、完整跳绳成功率 ≥80%。本文的旋转细杆是
**训练装置**，不是三鸭跳绳交付，也不能用其通过率替代最终验收。

## 新增实现

- 训练任务 `Mjlab-SweepRopeHop-Flat-MicroDuck`：在上一轮稳定高跳上加入
  有质量、有碰撞、限力矩 0.03 N·m 的旋转胶囊体；半径 1.5 mm，转动半径
  23 cm，轴高 24 cm。电机目标 2.8–3.4 Hz 随机采样，实际转角由物理积分产生。
- 策略保持 61 维输入；原 body pose `[x,y,z,roll,pitch,yaw]` 的 z 目标随
  实测绳位变化，其余分量仍为零。没有把相位伪装成姿态分量。高度目标为
  `0.06 * max(0, 1 - (wrap(angle)/(π*0.65))²)` 米，零相位为从脚下经过。
- 官方导出写入 `hop_height_command=sweep-v1`，运行时要求显式提供绳位来源，
  缺失则报错。三鸭场景取柔性绳中间材料段相对双脚和嘴部轴高的实测角度；
  简化装置回放取真实关节角。旧策略不会自动加入这一指令。
- 新接触惩罚及完整跳跃奖励的接触记忆；途中擦杆后不能在落地时领取无碰撞奖励。
- 独立 CPU 回放 `scripts/eval_sweep_hop.py`：逐物理步检查杆与鸭接触、非脚
  部位触地、完整脚底净空、倾角、漂移和至少 50 ms 脚部支撑落地。标记
  `kind=training_apparatus_only`，不表示柔性绳验收通过。

## 已发现并修正的问题

最初批量训练遗漏了固定底座的环境坐标偏移：机器人随 `env_origins` 平移，
装置却留在世界原点。单环境测试能碰到障碍，但大多数批量环境不能。
发现训练碰撞惩罚恒为零，而独立 CPU 回放仍频繁碰杆后，停止了该轮训练。
其检查点 `sweep-3900.onnx` 是失败对照，不作为新策略交付。

修正后，在 reset 时从默认根姿态重新计算装置位置并加上当前环境原点。
不累加偏移，不在正常运行中移动底座；转杆仍由电机和碰撞积分。
实际 GPU 64 环境检查：相对环境原点的初始装置高度为 0.4597–0.4700 m；
静止鸭对照中 64/64 环境均检测到杆接触。测试还覆盖局部 reset 和重复 reset。
训练仓库实现提交：`18052aa`。接手前已有的 `uv.lock` 修改未纳入提交。

早期 apparatus 评分 v1 只检查一帧落地，v2 与柔性绳验收一致要求 50 ms 脚部支撑，且通过 geom 所属
body 识别无名称的非脚碰撞网格。v3 再逐脚检查细杆真正经过脚的位置，要求
双脚都在杆的覆盖范围内，并在各自穿越时有净空。旧输出保留为 `*-v1.json`、
`*-v2.json`，最终装置对照使用 v3。
柔性绳验收器没有因此放宽或改变。

## 复现

训练仓库先进行 64 环境、5 iteration 的 smoke，再从高跳 3799 继续训练：

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/train \
  Mjlab-SweepRopeHop-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 300 --agent.save-interval 100 --agent.logger tensorboard \
  --agent.resume True --agent.load-run bootstrap --agent.load-checkpoint model_3799.pt \
  --agent.run-name gpu_sweep_contacts
```

`bootstrap/model_3799.pt` 是上一轮 `microduck_ropehop_high` 的原始检查点，
放在 `logs/rsl_rl/microduck_ropehop_sweep/bootstrap/`。所有训练日志留在本地。
导出使用训练仓库 `scripts/export.py`，不能绕过观测归一化。

交付仓库中的装置回放与消融：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_sweep_hop.py MODEL.onnx \
  --seconds 12 --output artifacts/takeover/sweep-eval.json
# 同模型、同物理装置，把高度命令置零，检查反馈是否有作用：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_sweep_hop.py MODEL.onnx \
  --seconds 12 --no-feedback --output artifacts/takeover/sweep-no-feedback.json
```

最终柔性绳回放仍用 `scripts/eval_physical_skip.py`，full 接触、球关节绳、
0.2 ms 步长、绳地时间常数 0.4 ms、绳长 0.58 m、半径 1.5 mm、跳跃鸭 y=0、
初始绳平面 π/2、无预热、不钳制绳速。以其 `passed` 为验收结果。

## 第一个真正通过的柔性绳检查点

`policies/ropehop_contact.onnx` 来自修复后的训练
`2026-09-11_12-51-49_gpu_sweep_contacts/model_3900.pt`，SHA-256：
`4d2e1f85bc3243dfe6d05d9221a77b855a9dbabf734d17214181dc9d032abdd4`。

seed 0、20 秒完整回放，前 9 秒为启动窗口，之后 **33/33 个完整圈通过**。
每圈都检查双脚真实穿越、脚底净空、绳鸭碰撞、落地、嘴部连接和倾倒。
66 次脚下穿越中最小净空约 31.5 mm；最大绳地数值穿透约 1.63 mm。
记录：`artifacts/takeover/physical-sweep-contact-3900.json`。

**此处是尚未限制转绳速度的中间结果。** 后续 30 秒测试中，seed 1 为 1/70、seed 2 为
4/64、seed 3 为 64/64。前两组分别暴露持续落地支撑不足、以及擦绳问题。
因此当时只能证明真实跳绳可以完成；下面继续记录最终限速配置的结果。

同种子、同物理模型的对照：

| 跳跃策略 / 控制方式 | 合格圈 / 完整圈 |
|---|---:|
| 旧高跳 3799，原几何同步 | 0/34（上一轮 20 秒记录） |
| 新接触策略，绳位反馈 + 原几何同步 | 33/33 |
| 新接触策略，关闭高度反馈，保留原几何同步 | 15/33 |
| 新接触策略，保留高度反馈，转绳频率固定 | 9/33 |

因此目前有效的是两端时序调整与跳跃端反馈共同工作的系统，不能说已经实现
完全独立的双智能体协同。固定转绳频率的选项保留用于消融，不作为默认交付。
简化装置同模型开启/关闭反馈都为 22/34；它的结果不能替代真实柔性绳回放。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_physical_skip.py \
  --hop policies/ropehop_contact.onnx --seconds 20 --seed 0 \
  --rope-joint-type ball --physics-dt .0002 --rope-radius .0015 \
  --rope-floor-timeconst .0004 --hop-start-delay 0 --settle-seconds 0 \
  --geometric-timing --rope-length .58 --jumper-y 0 \
  --rope-initial-phase 1.57079632679 --rope-velocity-limit 0 \
  --output artifacts/takeover/contact-replay.json
```

## 最终限速配置与全部种子

原来的转绳速度会追随鸭子的跳跃频率持续上升；部分种子缺少连续落地支撑。
保留几何相位反馈，把转绳速度上限设为 3.1 Hz 后，原先失败的 seed 1、2
均变为 65/65。这一设置没有改变接触模型、步长、净空或落地验收阈值。

| seed | 合格圈 / 完整圈 | 比率 |
|---|---:|---:|
| 0 | 65/65 | 100% |
| 1 | 65/65 | 100% |
| 2 | 65/65 | 100% |
| 3 | 43/63 | 68.3% |
| 101 | 65/65 | 100% |
| 202 | 37/64 | 57.8% |
| 303 | 65/65 | 100% |
| 404 | 37/64 | 57.8% |
| **合计** | **442/516** | **85.7%** |

每组 30 秒，启动窗口 9 秒；种子改变三只鸭子初始舵机姿态（±0.02 rad）。
101、202、303、404 是限速参数确定后才用于验证的种子。三个失败种子仍有
间歇性绳脚接触、净空不足和落地支撑不足，均保留在分母中。这只是指定
MuJoCo 仿真配置下的测试批次结论，不是任意扰动下的成功保证。

后续训练检查点没有替代 3900：4000 在 seed 1、2 分别为 0/65、1/25；
4098 分别为 0/0、0/11，均未通过。训练分数提升并不能替代部署回放。
修复后的正式训练完成 300 iteration、14,745,600 个并行环境步；训练日志
和失败批量环境实验均保留。配置、奖励及运行时回归检查共 **40 项通过**。

最终复现及录制：

```bash
# 不渲染，逐物理步验收；默认 seed 0、30 秒：
scripts/contact_skip.sh
# 完整 20 秒录像，不裁去启动过程：
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl scripts/contact_skip.sh --seconds 20 \
  --video out/contact_skip.mp4 --output artifacts/takeover/contact-final-video.json
```

本机已验证渲染器为 `NVIDIA GB10/PCIe`。需要同时设置这两个 EGL 环境变量：
现有环境的 `PYOPENGL_PLATFORM` 默认值会强制 OSMesa，单改 `MUJOCO_GL` 不够。
无 EGL 的机器可使用 `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`，但渲染更慢。

下一步应针对失败种子的完整圈接触与落地历史训练，改善跨初态稳定性。
目前训练装置虽有真实接触，仍是刚性旋转杆；柔性绳形状、两端起旋混沌和
真实多智能体训练尚未在同一训练环境中端到端解决。
