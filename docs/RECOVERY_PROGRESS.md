# 修复进展（2026-09-11）

后续绳位反馈、真实接触训练及跨种子结果见 [SWEEP_TRAINING.md](SWEEP_TRAINING.md)。
本文件保留第一阶段的实验历程，以下“未达标”描述对应当时的旧检查点。

本轮在交付仓库和训练仓库同时修复。旧 ONNX 和交付视频没有被替换。
**尚未达到真实跳绳 80% 的目标。**

## 跳跃策略的根因与修复

- 单鸭、无绳测试也能重现 v10 的前扑/嘴部触地，因此不是单纯的绳碰撞问题。
- 旧 `HOME_FRAME` 的 `init_state.pos.z=0` 会覆盖 XML 的躯干高度 0.12 m；
  通用 `reset_base` 再加 0.01–0.05 m，使正常站立姿态的身体/腿部出生在地板内。
  新环境将初始躯干高度明确设为 0.12 m，只加 0–5 mm 的出生高度扰动。
- 旧跳跃奖励的 airborne/grounded 主要由脚部 site 高度判断。嘴撑地、脚抬起也能
  获得 apex/rhythm 奖励；landing 奖励没有要求真实脚部接触。
- 新建 `Mjlab-CleanRopeHop-Flat-MicroDuck`，与历史任务隔离：给所有真实碰撞网格
  命名，加入非脚部/地面传感器。身体触地立即终止并惩罚；apex 要求无脚部支撑、
  无身体接触且 upright cosine >0.8；landing 必须有真实脚部接触并保持同样姿态条件。
  不改变 61D 观测契约、PD 执行器类型和动作缩放，不修改历史 ONNX。
- 修正本项目 runner 的 checkpoint 加载：默认映射到运行设备，允许 CPU 载入
  CUDA 保存的 checkpoint，显式指定 map_location 仍然优先。

相关训练代码位于 `../microduck_rl/src/mjlab_microduck/tasks/`；
`tests/test_clean_ropehop_cfg.py` 检查身体碰撞覆盖、出生不穿地、撑头/倒置状态不能
获得跳跃奖励、CPU 加载映射、旧环境隔离。

## 更严格的 CPU 评估器

`scripts/eval_hop_quality.py` 统计单鸭完整跳跃，不是过绳成功率。每次跳跃要求：

- 起跳前有至少 20 ms 正常脚部支撑；无任何地面接触的飞行至少 30 ms。
- 用实际足底碰撞网格到地板的距离测量净空，峰值至少 15 mm。
- 全程 upright cosine >0.8，身体不接触地面，离起点不超过 80 mm。
- 落地后至少 50 ms 保持脚部支撑；未完成飞行/落地的末尾事件不算成功。

四个固定种子、每个 2 s 站立预热 + 10 s 跳跃：

| 策略 | 合格跳跃/尝试合计 | 每次运行非脚部触地 | 最大漂移 |
|---|---:|---:|---:|
| 交付 v10 | 0/124 | 4.185–4.234 s | 约 7.5–7.7 cm |
| 历史 v7 | 4/120 | 1.856–1.860 s | 约 70.7–75.4 cm |

v7 的姿态较好，但严重漂移，不能直接替换交付策略。v7 的原始 checkpoint 来自
ONNX 元数据：`microduck_ropehop/2026-09-05_18-47-22_microduck_ropehop/model_2500.pt`。
站立策略的负对照（seed 0、3 s）得到 0 次尝试、0 次身体触地，防止把站立当成功。

```bash
.venv/bin/python scripts/eval_hop_quality.py policies/ropehop_classic.onnx \
  --output artifacts/takeover/hop-v10-strict.json
.venv/bin/python scripts/eval_hop_quality.py ../microduck_rl/output_ropehop_v7.onnx \
  --output artifacts/takeover/hop-v7-strict.json
```

## 绳端几何修复与约束实验

三铰链绳的第一段现在从 wrapper 的连接点开始，消除了之前一节长度（约 20.8 mm）
的无形偏移。构建器默认使用修正几何；历史 demo 显式保留 legacy 配置以复现旧视频。
`--correct-rope-root` 打开修正版本，`--connect-timeconst` 用于测量连接强度。

在真实绳地+绳鸭碰撞、50 s 演示、四个种子条件下：

| 配置 | 无数值异常完成 | 最大锚点间距（各 seed，50 Hz 采样） |
|---|---:|---|
| 历史几何，CG，40 ms | 4/4 | 124.9、82.9、129.7、153.5 mm |
| 修正几何，CG，4 ms | 3/4 | seed 1 在 18.638 s 数值崩溃；不能采用 |
| 修正几何，Newton，4 ms | 4/4 | 4.1、9.6、16.0、9.6 mm |

修正版本显著减少连接漂移，但并未证明所有初始条件稳定；seed 2 起旋表现仍差。
这些实验沿用旧跳跃策略，不能将结果中的旧时序命中数当成跳绳成功率。
绳的软地面接触仍有约 12–21 mm 的采样穿透，后续仍需解决。

```bash
.venv/bin/python scripts/audit_honest_skip.py --contacts full --seconds 50 \
  --correct-rope-root --connect-timeconst .004 --solver Newton --seed 0 \
  --output artifacts/takeover/root-fixed-newton-seed0.json
```

完整数据在 `artifacts/takeover/`，包括失败的实验，不筛掉失败种子。

随后固定参数测了未参与调参的 101、202、303、404 四颗种子，均无数值异常完成。
最大采样锚点间距分别为 7.4、6.1、6.7、5.9 mm。404 仍明显失去良好的起旋/跳跃，
所以不能把 8/8 数值稳定当成 8/8 跳绳成功。

## 本轮短训练的实测结果

修正出生高度后，64 env × 5 iteration 的 CPU 冒烟测试通过，再从 v7 的
`model_2500.pt` 继续训练 100 iterations（64 env，24 steps/env/iteration，约 244 s）。
没有发生 NaN 终止。早先未修出生高度的短运行已经停止，没有将其模型混入本次结果。

使用仓库官方 `scripts/export.py` 导出（包含观测归一化、61D actor），最终候选是
`artifacts/takeover/ropehop_clean_2599.onnx`。四个种子各 10 秒 CPU 评估结果：

- 合格跳跃 **4/124**，尚不能作为可用策略。
- 最大漂移降至 **11.0–19.3 cm**，优于 v7 的 70.7–75.4 cm，但仍超出 8 cm 验收范围。
- 非脚部触地仍有 **2.087–2.154 s**，比 v7 的 1.86 s 更长。不能因为漂移改善就说
  整体跳跃质量已改善；需要进一步训练和检查嘴部姿态/落地过程。

训练记录：`artifacts/takeover/clean-hop-training.log`；完整评估：
`artifacts/takeover/hop-clean2599-strict.json`。候选视频记录在
`out/clean_hop_candidate_2599.mp4`（含 2 s 站立预热，诊断用途，非交付演示）。

复现（训练仓库根目录，CPU 不占用其他 GPU 服务）：

```bash
# 将 v7 checkpoint 复制到新实验目录的 bootstrap_v7/model_2500.pt 后：
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/train \
  Mjlab-CleanRopeHop-Flat-MicroDuck --env.scene.num-envs 64 \
  --agent.max-iterations 5 --agent.logger tensorboard --gpu-ids None

WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/train \
  Mjlab-CleanRopeHop-Flat-MicroDuck --env.scene.num-envs 64 \
  --agent.max-iterations 100 --agent.save-interval 50 --agent.logger tensorboard \
  --gpu-ids None --agent.resume True --agent.load-run bootstrap_v7 \
  --agent.load-checkpoint model_2500.pt
```

本轮测试：训练仓库 5 项，交付仓库绳碰撞/端点与跳跃负对照 9 项。GPU 冒烟启动时
报 CUDA OOM，当时机器上两个 VLLM 进程共占约 89 GiB GPU 内存；未停止它们，改用
CPU 完成了上述训练、官方导出与评估。

## GPU 释放后：完整落地奖励与实际绳路验收（进行中）

GPU 64 环境 / 5 iteration 冒烟通过。第一轮 2048 环境、500 iteration
从 2599 继续，最终 3098 的 CPU 回放反而退化：四个种子都只有 1 次合格跳跃，
每 10 秒非脚部触地 7.2–8.3 秒。该检查点没有替换交付策略。

找到奖励尺度漏洞：奖励管理器乘以 0.02 s，原身体触地终止只扣 0.2，
单次抬脚最高却能拿约 6 分。严格环境将终止扣分改为 5，并增加只在完整飞行后
脚部支撑连续 60 ms 才支付一次的落地奖励；限制头部大幅折叠。新增状态机测试
覆盖一次支付、站立不重复支付、飞行中碰身/漂移、空中重置不可领落地奖励。
修改后的 64 环境 / 5 iteration 冒烟及 6 项 CPU 测试通过。

第二轮 2048 环境训练目录：
`microduck_rl/logs/rsl_rl/microduck_ropehop_clean/2026-09-11_04-08-46_gpu_clean_landing`。
中间 3200 官方导出后的单鸭 CPU 回放（四种子各 10 秒）：43/128 合格跳跃，
四种子非脚部触地和倾倒时间均为 0；最大漂移仍为 9.8–36.7 cm。
这说明落地明显改善，但位置控制仍未解决，也不等于跳绳成功率。

新增 `scripts/eval_physical_skip.py`：每个物理步检查实际绳段穿过左右脚所在
截面时的净空、过顶、落地支撑、鸭身触地、绳鸭接触、嘴部连接间隙及转绳鸭姿态。
分母是同一段绳的完整转圈，摆动和零圈不能通过验收。超过 2 mm 的绳地穿透
也判该圈失败。完整圈需至少 20 圈、合格比例至少 80%，并在预热期完成起旋。
判定状态机有 12 个正反例测试；其几何检测仍需在实际成功动作上继续验证。

旧策略首次严格场景回放：旧时序 125/131，实际完整圈 130，合格圈 0。
原始记录 `artifacts/takeover/physical-baseline-seed0.json` 生成于新增地面穿透
否决项之前；它本来已经为零，不可当作新版完整验收记录。

绳地接触实验：直接将全部绳接触设成 4 ms 高刚度，会在起旋阶段数值崩溃。
因此新增可选 `rope_floor_timeconst`，编译时仅给 24 段绳与地面建立显式接触对，
不改变绳鸭/鸭地参数。2 ms 在种子 0–3 的 50 秒回放中均无数值异常，
50 Hz 采样穿透为 1.7、2.5、3.6、5.6 mm；嘴部间距仍有 9.9–36.9 mm。
尚未全达标，所以仍是显式实验参数，未改成默认交付配置。

### 原地反馈策略的独立验收

新增 `Mjlab-CenteredRopeHop-Flat-MicroDuck`：保留 61D 顺序，twist 槽提供根据
起跳位置/朝向计算的回正速度指令。训练与运行时使用相同公式；官方导出器添加
`hop_centering=v1` 元数据，运行时仅对带此标记的策略启用回正。仿真中使用世界
位置反馈；不能宣称这套定位已在实体硬件上实现。

2048 环境 / 500 iterations，预热来源为 clean landing 3200。训练目录：
`microduck_rl/logs/rsl_rl/microduck_ropehop_centered/2026-09-11_04-17-57_gpu_centered_hop`。
下表全部是导出 ONNX 后的独立 CPU 回放，**没有绳**：

| 检查点 | 种子 0–3，各 10 秒合格跳跃 | 最大漂移 | 身体触地/倾倒 |
|---|---:|---:|---:|
| centered 3300 | 116/128 | 9.3–10.1 cm | 0 |
| centered 3400 | 124/128 | 3.7–3.8 cm | 0 |
| centered 3500 | 124/124 | 3.2 cm | 0 |
| centered 3699（最终） | 124/128 | 4.4 cm | 0 |

选择 3500 作为当前单鸭基线，而非默认选择最后一个检查点。未参与调参的
101、202、303、404 各 30 秒均为 **93/94**，合计 **372/376**，身体触地和
倾倒仍均为零，最大漂移约 3.2 cm。末尾尚未落地的跳跃计失败，没有剔除。
原始数据：`artifacts/takeover/hop-centered3500-strict.json` 和
`hop-centered3500-heldout30s.json`。该策略的实际脚底净空中位数约 **2.5 cm**。

### 接绳后的结果与剩余问题

新增可选球关节表示，保持 24 段绳、两端嘴部连接及全部碰撞；它避免了三铰链
坐标表示的部分数值问题。球关节 + 0.5 ms 步长 + centered 3300 在 seed 0 完成
50 秒，验收窗口内 130 个完整转圈无身体触地/倾倒否决项，但 **每圈仍有绳脚
接触，合格 0/130**。减小绳半径至 1.5 mm 也没有自动解决跳绳。

实测定位到几何问题：旧布局（绳长 0.50 m、跳跃鸭 y=-0.10 m）中，绳实际穿过
脚位置时常在约 5–11 cm 高，脚底却只有约 2–3 cm 高。不能靠旧“最低点时序”
计数判定跳过。新增实际双脚截面穿越时刻反馈及逐次几何记录。将跳跃鸭移到
中心、试验 0.55–0.60 m 绳长后绳路降低，但仍有脚部碰撞与地面穿透。
这些布局参数均显式保存，尚未作为通过验收的配置。

随后启动独立 `Mjlab-HighRopeHop-Flat-MicroDuck`，从已验证的 centered 3500
继续：提高训练顶点目标，并要求更高的完整跳跃才支付落地奖励。保持已经验证的
低跳模型不变。该阶段结果待 CPU/完整场景回放，不能预先视为改进。

### 高跳训练结束

`microduck_rl/logs/rsl_rl/microduck_ropehop_high/2026-09-11_04-33-09_gpu_high_hop`
完成 2048 环境 × 300 iterations。3600、3700、3799 三个导出检查点在四个种子
各 10 秒的 CPU 回放中均为 31/31 合格跳跃，无身体触地或倾倒。
实际脚底顶点中位数分别为 4.05、4.01、3.95 cm。最终 3799 最大漂移为
2.8–3.0 cm，保存为 `policies/ropehop_high.onnx`；原低跳基线另存
`policies/ropehop_centered.onnx`，旧 `ropehop_classic.onnx` 没有替换。

两份新模型都通过官方导出器生成，包含归一化与 `hop_centering=v1`。

```bash
# 交付仓库内，无绳单鸭验收：
.venv/bin/python scripts/eval_hop_quality.py policies/ropehop_high.onnx \
  --seeds 101 202 303 404 --seconds 30 --output artifacts/takeover/high-heldout.json
# 可视化（有桌面 GL 时也可用正常 EGL/GLFW 环境）：
MUJOCO_GL=osmesa .venv/bin/python scripts/eval_hop_quality.py policies/ropehop_high.onnx \
  --seeds 0 --video out/clean_hop_high_3799.mp4 --output artifacts/takeover/high-video.json
```

同步相位偏置 ±50 ms 的真实绳接触试验仍未达标。读取两端嘴部轨迹排除了
“相向鸭子在世界坐标中反向甩绳”的猜测：稳态端点 y/z 相关系数分别为
0.991/0.988，数据在 `artifacts/takeover/drive-pins-high3600.json`。
没有据此猜测改写转绳控制器。后续继续检查初始绳与腿的相对位置，以及足部
完整几何净空；这些失败记录不计入成功样本。

高跳 3799 在未参与调参的 101、202、303、404 各 30 秒回放中，均为 **93/94**，
最大漂移 3.0–3.1 cm，身体触地和倾倒均为 0。最后一次尚未落地的跳跃计失败。
验证视频 `out/clean_hop_high_3799.mp4` 是完整 12 秒（2 秒站立、10 秒跳跃），
并非筛选出的短成功片段。

最后两组起始绳位对照：球关节绳、0.2 ms 步长、绳地接触时间常数 0.4 ms、
绳长 0.58 m、半径 1.5 mm、跳跃鸭 y=0、无站立预热、初始绳平面 ±π/2。
seed 0 各 20 秒都无数值异常，验收窗口各 34 圈没有松脱、倾倒、身体触地或
穿地否决项，最大绳地穿透为 1.64/1.62 mm。**两组仍为 0/34 合格跳绳，
均被绳脚接触及净空否决。** 不能归结为单纯的出生绳位问题。

当前判断：单鸭稳定性与部分接触求解问题已改善；下一阶段需要直接向跳跃策略
提供绳位/相位反馈，并在绳接触环境中训练。现有跳跃策略只根据位置回正，
转绳端的时序反馈无法充分避免脚部先接触绳。上述判断是下一步工作假设，
不是已经完成的训练成果。

本次新增及接手回归测试共 **31 项通过**（训练仓库 7，交付仓库 24）。训练代码
保存在本地分支 `codex/physical-skip-recovery`，提交 `db8bab2`。
训练仓库接手前已有的 `uv.lock` 修改未纳入本次提交。

最终汇总入口：`artifacts/takeover/summary.json`。发布到仓库的模型在 `policies/`；
中间导出 ONNX 和原始训练控制台日志留在本地 `artifacts/takeover/`，不纳入仓库。
JSON 评估记录保留全部已记录的失败配置。三鸭诊断录像为
`out/physical_skip_high_3799.mp4`，其 HUD 标明物理检查未通过。

补充数值保护对照：同一高跳 3799 / 球关节 / 0.2 ms 配置关闭旧代码的绳速钳制
（`--rope-velocity-limit 0`），仍无数值异常完成 20 秒，34 个完整圈，0 个合格
跳绳。见 `artifacts/takeover/physical-high3799-unclipped-seed0.json`。
现有参数默认仍保留历史 40 的钳制值以兼容旧铰链场景；新的球关节对照已证明
本次短回放的稳定性不依赖这一保护。

两份最终视频均已核对为完整 12 秒、25 fps、960×544。三鸭诊断视频的 9 个
验收窗口内完整圈均未通过，HUD 没有展示旧时序“成功率”。
