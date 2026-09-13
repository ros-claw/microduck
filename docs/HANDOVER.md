# Microduck 三鸭合作跳绳 — 交接文档

> **发布整理（2026-09-13）：** 当前入口为 [项目首页](../README.md)。已提供原生材质发布片、嘴部与脚下慢动作特写、[训练源码包](../training/README.md) 和 [文档索引](README.md)。以下旧交接正文作为历史保留。

> **接手后更新（2026-09-11）：** 请先读 [绳位反馈与真实接触训练记录](SWEEP_TRAINING.md)
> 和 [第一阶段修复记录](RECOVERY_PROGRESS.md)。下面保留的是 2026-09-10 的历史交接，
> 其中旧策略、关闭绳鸭碰撞的限制及复现命令不代表当前接触训练版本。

> 写给接手人：本文档记录这个项目的完整开发历程、所有实测物理结论、踩过的坑、
> 当前系统架构与复现命令。结论全部来自仿真实测（标注"实测"的都有对应实验），
> 未验证的推断会明确标注。请先读第 0、3 节（现状与诚实限制），再决定从哪接。

最后更新：2026-09-10

---

## 0. 现状一句话

**三只鸭子（两个转头 + 一个跳跃）在同一个 MuJoCo 物理世界里跳长绳，绳子两端
用 connect 约束连在两只转头鸭喙里的把手上，绳子由转头鸭的 RL 策略驱动旋转
（无任何 mocap 替身），50 秒连续跑，按"时序判定"计 122/131 = 93% 跳过。**
视频：`rosclaw-microduck/out/honest_skip_full.mp4`，代码已推到
`github.com/ros-claw/microduck`（commit `54fbb10`）。

**但请看第 3 节的诚实限制——"93% 跳过"的物理含义和用户的"抽象"批评都要正视。**

---

## 1. 项目地址与代码位置

| 内容 | 位置 |
|---|---|
| 交付仓库（demo/部署/视频） | `github.com/ros-claw/microduck`，本地 `~/workspace/microduck/rosclaw-microduck`，HEAD = `54fbb10` |
| 训练仓库（RL 环境） | `~/workspace/microduck/microduck_rl`（pollen-robotics/microduck_rl 的 fork，CR-05 工作已本地 commit、**未推送**——无该远端推送权限） |
| 官方基础策略（站/走） | `~/workspace/microduck/microduck/policies/`（pollen-robotics/microduck） |
| 交付视频 | `~/workspace/microduck/honest_skip_full.mp4` = `rosclaw-microduck/out/honest_skip_full.mp4` |
| 转头策略 ONNX | `rosclaw-microduck/policies/turner_rope.onnx`（73D obs） |
| 跳跃策略 ONNX | `rosclaw-microduck/policies/ropehop_classic.onnx`（= `microduck_rl/output_ropehop_v10.onnx`，61D obs） |

关键文件：
- 部署 demo：`rosclaw-microduck/src/microduck_lab/demos/honest_skip.py`（完整跳绳编排：转头+绳+跳跃+PLL+看护）
- 世界构建：`rosclaw-microduck/src/microduck_lab/sim/classic_rope.py`（`build_classic_world`，`rope_kind="triple"` + `connect_to="handles"`）
- 部署运行时：`rosclaw-microduck/src/microduck_lab/sim/runtime.py`（`DuckRuntime`/`TurnerDuckRuntime`/`RopeTurnerDuckRuntime`）
- 录制脚本：`rosclaw-microduck/scripts/record_honest_skip.py`
- 训练环境：`microduck_rl/src/mjlab_microduck/tasks/microduck_turner_rope_env_cfg.py`（`Mjlab-TurnerRope-Flat-MicroDuck`）
- 训练绳体：`microduck_rl/src/mjlab_microduck/robot/turner_rope_spec.py`
- 训练 MDP 函数：`microduck_rl/src/mjlab_microduck/tasks/mdp.py`（绳相位/旋转奖励/载具驱动/防甩钳制/侧别重置等）
- 物理笔记：`rosclaw-microduck/docs/physics-notes.md`

---

## 2. 目标与验收（用户原始需求）

1. 三只鸭子合作跳长绳（2 转头 + 1 跳跃），闭环 ROSClaw 流程。
2. **绳子末端必须真的连在鸭子嘴里**（不接受绳子自己在空中动）。
3. **绳子必须是鸭子甩起来的**（不接受载具/隐形手）。
4. 成功率 80%+，出好看的视频，推到 `ros-claw/microduck`。

达成情况：2、3 物理上达成（connect 到喙把手 + RL 转头策略驱动）；4 按时序判定
93%，已推送。**但"抽象"的批评成立与否，见下一节。**

---

## 3. 诚实的限制（接手前必须知道）

这些是我刻意做进系统里的妥协，用户觉得"抽象"主要源于此：

1. **绳 ↔ 跳跃者之间永远无物理接触**。重链以 3 m/s 擦过跳跃的鸭子会炸接触求解器
   （实测 QACC NaN 两次）。所以"跳过"的判定是**时序判定**：绳腹经过跳跃者脚下时，
   若鸭子恰好腾空且直立则记一次成功。绳不会真的绊到脚——视频里绳从脚边穿过也
   不会碰到。这是最"抽象"的一点：**跳绳是真跳过，但绳脚不相干**。
2. **部署中的绳是无接触的**（`contype=0`）：训练环境的绳无接触（Warp 稳定性妥协），
   部署为保持动力学一致也无接触。绳穿地板而过（视觉上扫过地面但不接触）。
   原始载具版（`classic_skip.py`）的绳有地面接触，那个看起来更"贴地"。
3. **起旋是混沌的**：同一策略不同初始扰动，起旋成功率差异很大（一颗种子 89%，
   另一颗直接起旋失败）。发布的视频是起旋顺利的那条。每次运行是确定性的
   （同种子同结果），但跨种子不鲁棒。
4. **转头鸭的头部动作在视频里不明显**：驱动主要靠全身摆动 + 头部小幅画圈
   （7.5 cm/3.1 Hz），不是人类甩绳那样的大臂动作。绳确实只受鸭子驱动（无其他
   接触源），但"鸭子在甩"的视觉冲击弱。
5. **"93%"是按上述时序判定的通过率**，不是"绳完美掠过脚底"的物理度量。
6. 跳跃者是**自主节拍跳跃**（3.1 Hz 自走），PLL 把绳的相位对齐到它的腾空窗口，
   不是跳跃者主动迎绳。

如果接手人想做"更真实的跳绳"，优先攻 1（解决重链-鸭子接触的数值稳定性，
见 7.4）和 4（更强的转头动作）。

---

## 4. 开发历程（按时间线，含所有失败分支）

### 4.1 v3：地面扫绳"蛇"版（已交付但被否）
最早的可用版本：绳子贴地左右扫（不是过顶大绳），跳跃者按节奏跳。93% holdout
成功率（`practice/champion.json`），但**那不是真正的跳大绳**——绳不过头顶。
用户否掉："太抽象"。代码仍在（`scripts/evolve_skip.py` 等）。

### 4.2 载具版经典大绳（94%，但被否为"不诚实"）
`demos/classic_skip.py`：两个 mocap 载具（隐形手）按 `ChainForcedDrive`
（线性共振种子 → 目标频率强迫画圈）驱动重链绳，形成过顶大环，v10 跳跃者跳，
**94/100 跳过、50 秒稳定**。但绳连在隐形载具上、由载具驱动——用户连续两次否掉：
"绳子似乎自己在空中乱动"、"绳子末端根本没连在鸭子嘴上、也不是鸭子甩起来的"。
**这次否定是项目转向的关键：必须从"能跑通"转向"物理诚实"。**

### 4.3 诚实版探索的失败分支（每个都实测过，别再走）
- **弹性电缆绳**（elasticity cable 插件）：≥2.8 Hz 才能维持环，且混沌易碎——任何
  扰动都杀死环；mujoco_warp 无该插件后端，上不了 GPU。弃。
- **单端自由绳（鞭子）由任何固定模式驱动起旋**：实测全部失败（固定画圈 1.2–4 Hz、
  相位锁定画圈、速度泵——绳腹角速度全为 0；绳落地后再起不来）。**单端绳起不了旋，
  这是 CR-05 第一版（自由绳进训练环）失败的根本原因。**
- **开环固定画圈驱动 RL**：策略看绳相位也救不了——固定相位画圈奖励与"按绳相位
  自适应泵"冲突，绳转奖励恒为 0（700 迭代不动）。
- **奖励加膨胀门控**（rope_rotate = 速率 × 半径高斯门）：悬垂绳半径≈0 → exp(-8)≈0，
  信号被门控杀死，**稀疏死锁**（700 迭代 flat 0）。教训：别用乘法硬门，改成无门控
  正速率支付（摆动半周也付小份额，全转付满额——严格单调改进）。

### 4.4 突破口：双端同相位画圈（双钉探针实测）
最小物理探针（CPU，两个 mocap 钉 + 链绳）实测：
- **两个钉同相位 3.1 Hz 画圈（r=0.05–0.06 m），双端绳从静止直接起旋并锁频**，
  无需种子、无需升降。这是整个诚实版赖以工作的物理基础。
- 起旋几何窗口很挑（L、跨距、钉高、半径都敏感，且双稳态混沌）。**钉高是关键**：
  z=0.228 不起旋（起旋期绳擦地被磨死），z=0.27 起旋。最终工作点：
  **跨距 0.276 m、钉高 0.27 m、L=0.50 m、r=0.06 m、3.1 Hz**（密度 500 时 r=0.075）。
- **绳必须能自拧才能转环**：双端绳每转一圈要自拧一圈。球关节链可以；
  铰链对（无轴向自由度）物理上转不成环（实测：球链 +29 圈，铰链对 0）。
  Warp 兼容方案 = **铰链三件套**（弯曲 y/z + 轴向扭转铰链，扭转铰链必须带
  `armature=1e-5`，否则小滚转惯量使 Hessian 秩亏直接崩）。
- **关节阻尼杀死起旋**：damping=0.002 就起不了，只能 0.0002；所有耗散只能靠
  防甩钳制提供，不能靠关节阻尼。

### 4.5 训练环境的 Warp 稳定性（CR-05，每个坑都实测）
把真链绳放进 Warp 训练环（绳钉在鸭喙把手、远端接刚性画圈载具），策略看绳相位、
赚旋转奖励。踩过的坑：
- **防甩钳制是刚需**：不钳时链鞭积分出无界能量，鸭子被弹到 ±1e9 m、jv 到 1e13。
  每 env step 把绳关节速度钳到 ±20 rad/s（事件 `clip_passive_joint_vel`）。
- **eq_data 编译器 bug**（与 CPU demo 同款）：MuJoCo 编译时按静止姿态自动算
  connect 锚点，实测把绳端锚在离载具 40 cm 处（等于没连）。启动事件把
  `wp_model.eq_data` 清零（`fix_connect_anchors`）。
- **距离门控 connect**：鸭子乱甩时两端被拉紧，两个约束互掐一步内炸求解器
  （jv 1e25）。按喙↔载具距离 >0.5 关掉 `eq_active`（`gated_connect_and_anti_whip`）。
- **mjlab 观测/数据是世界系（含 env origin）**：环境排布 2 m 间距，任何世界坐标
  逻辑（出生检查、绳腹轴、载具目标）必须减 `env.scene.terrain.env_origins`，
  否则第一步就全终止。
- **鸭子必须出生在绳的近端**：velocity 环境的 reset_base 会把鸭子随机撒 ±1 m，
  双端绳直接把散落的鸭子拽飞。出生点钉死在原点附近。
- **观测契约 bug**：mjlab 原生 `joint_pos/joint_vel` 观测在这里**不排除** `passive_*`
  关节（出来 86 维而不是 14 维，部署契约静默破坏）。用 `joint_pos_rel_servo`/
  `joint_vel_rel_servo` 显式过滤。
- **观测维度演进**：转头 69D → 加 `turner_amp`(1) = 70D → 加 `rope_phase`(3) = 73D。
  部署运行时 `RopeTurnerDuckRuntime` 严格对应 73D 布局。
- **权重移植**：从 6 cm 纯转头策略（70D）向绳环境（73D）移植——actor 第一层
  `mlp.0.weight` 和观测归一化器的第 60:63 列（rope_phase）插零，其余照抄。
  冷启动可行。
- **载具必须刚性**：给载具加顺从（弹簧追踪 + 绳拉伸拖拽）会吸走驱动能量——
  实测任何顺从系数都杀死起旋。远端必须刚性钉住。
- **镜像载具**（远端复制鸭嘴尖端的偏差 = "另一只同样的鸭子"）：训练出的策略
  部署未见明显更好（实测部署 39%），该分支未采用。

### 4.6 部署的两个最后拦路虎（都已修）
1. **出生姿态**：鸭子从 XML 直腿 `qpos0` 出生，而策略的 `joint_pos_rel` 是相对
   收腿基准（DEFAULT_POSE）——±0.45 rad 的观测偏移。站/跳策略鲁棒扛得住，
   专家转头策略瞬间输出暴力动作摔倒。**部署时必须把鸭子放到 DEFAULT_POSE 再起
   步**（`honest_skip.py` 已做）。这是转头策略一上 CPU 就摔的全部原因。
2. **时序 PLL 只调相位会极限环**（~50%）：跳跃者不是节拍器，速率慢漂 → 相位追
   不上。改成经典载具版同款：**速率 EMA 跟踪 + 滤波相位按 25%/周期积分**——
   直接从 ~50% 跳到 93%。

---

## 5. 当前系统架构

### 5.1 训练侧（microduck_rl）
- 环境 `Mjlab-TurnerRope-Flat-MicroDuck`：鸭子（喙把手钉铰链三件套绳，L=0.50、
  24 段、密度 500、damping 0.0002、扭转带 armature）+ 远端刚性载具按
  `rope_carrier_drive` 以 3.1 Hz、r=0.075 同相位画圈；绳远端软 connect（solref 0.04）
  到载具。
- 观测 73D：[lin_vel3, ang_vel3, grav3, jp14, jv14, last_act14, twist3,
  turner_phase2, turner_amp1, tip_rel3, rope_phase3, head_cmd4, body_cmd6]。
- 奖励：rope_rotate(w4) + turner_circle_track(w4, std 0.04) + turner_circle_size(w3)
  + upright(0.5) + stay_in_place(10) + action_rate(-0.05)。
- 侧别随机化：50% 环境鸭子朝向 -x（绳伸向 -x、载具在 -x），观测/奖励全用世界系
  ——**一个策略同时能当左右两个转头鸭**（部署时 lavender 朝 +x、cream 朝 -x）。
- 事件：防甩钳制 + 距离门控 connect（`gated_connect_and_anti_whip`）、载具驱动
  （`rope_carrier_drive`）、eq 锚点清零（`fix_connect_anchors`）、侧别重置
  （`reset_turner_side`，完全接管根姿态）、`root_state_sane`（有限但弹飞的状态也终止）。
- 终止：nan_state + root_sane + fallen_too_long。
- 训练：`uv run train Mjlab-TurnerRope-Flat-MicroDuck --env.scene.num-envs 2048`，
  ~6 s/iter（共享 GPU），4000+ 迭代。从 6 cm 纯转头检查点移植冷启动。

### 5.2 部署侧（rosclaw-microduck）
- `build_classic_world(rope_kind="triple", connect_to="handles", turner_sep=0.448,
  rope_length=0.50)`：三只鸭子 + 铰链三件套绳（无接触），两端 connect 到转头鸭
  把手**尖端**（z≈0.27），软 solref 0.04。
- 两只转头鸭跑 `RopeTurnerDuckRuntime`（73D）+ `turner_rope.onnx`，共享时钟
  （同相位），`turn_frequency=3.1`。
- 跳跃鸭跑 `DuckRuntime`（61D）+ `ropehop_classic.onnx`。
- 编排（`honest_skip.py`）：鸭子放 DEFAULT_POSE → 防甩钳制每物理步 → 绳转率
  持续 >15 rad/s 且绳腹扫低时跳跃者开始跳 → 时序 PLL（速率 EMA + 滤波相位积分）
  把绳对齐到腾空窗口 → 看护者（持续趴地→回站重跳）。
- 起旋与绳：转头鸭从 t=0 就画圈，绳自行起旋锁频 3.1 Hz（实测部署中 3.10 Hz）。

---

## 6. 复现命令速查

```bash
# 录制跳绳视频（用仓库自带策略，确定性复现 122/131）
cd ~/workspace/microduck/rosclaw-microduck
MUJOCO_GL=osmesa uv run python scripts/record_honest_skip.py
# 输出 ~/workspace/microduck/honest_skip_full.mp4

# 训练转头绳策略（从 6 cm 纯转头检查点移植）
cd ~/workspace/microduck/microduck_rl
uv run train Mjlab-TurnerRope-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.resume True --agent.load-run <run_dir> --agent.load-checkpoint model_XXXX.pt

# 冒烟测试（必做，64 envs 5 iter）
uv run train Mjlab-TurnerRope-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 导出 ONNX（含归一化，部署必须走这条路）
uv run scripts/export.py Mjlab-TurnerRope-Flat-MicroDuck --checkpoint-file <ckpt.pt>
```

## 7. 已知问题与给接手人的下一步建议

按优先级：

1. **绳↔跳跃者物理接触**（"抽象"的根源）：重链 3 m/s 擦跳跃者会炸接触求解器。
   方向：更轻的绳 / 更软的绳材料 / 接触求解器参数（更小 solimp、condim）/
   分层接触（只留绳腹附近几段与跳跃者接触）/ 或干脆接受时序判定但在视频上把
   绳画成会绕开（美术向，不推荐——用户要的是物理真实）。
2. **部署绳恢复地面接触**：训练时无接触是 Warp 稳定性妥协；若能让训练绳对地板
   接触（只排鸭子-绳接触，保绳-地板接触），部署的"扫地带擦痕"视觉和速率调节
   （擦地制动）都会更真实。需重测 Warp 稳定性。
3. **起旋鲁棒性**：起旋混沌，跨种子不稳。方向：训练时给初始状态更大随机性
   （扰动域随机化），让策略对各种起旋结果都鲁棒；或加一个短暂的脚本化
   "起旋种子"动作（鸭子头部可执行的线性摆，参考被否的载具种子思路但由鸭子执行）。
4. **真·双智能体训练**：当前是一只鸭 + 刚性载具，部署是两只鸭（都会让）。实测
   双鸭会被绳互相拽拢（两侧张力）。把第二只鸭子也放进训练环（共享策略、镜像观测），
   是治"两侧塌缩"的正道，但工程量大（mjlab 多实体 + 每鸭一次前向）。
5. **跌倒起身**：跳跃者或转头鸭摔倒后无 getup 策略，靠看护者回站（跳跃者有，
   转头鸭没有）。长时演示需要转头鸭的起身。
6. **转头动作更"显眼"**：当前驱动以全身摆动为主，头部小动作。若想视频里鸭子
   明显"在甩"，可在奖励里加大头部画圈权重/幅度（注意：3.1 Hz 头部画圈会把站立
   鸭子晃倒，需配合平衡，实测）。

## 8. 关键实验记录（可复跑的探针）

最小物理探针在 `/tmp/probe_*.py`（当时是临时的，建议接手人重建存档）：
- 双钉同相位画圈起旋（跨距/钉高/半径/频率/密度 扫描）→ 第 4.4 节窗口。
- 铰链对 vs 球链 vs 铰链三件套的转环能力 → 第 4.4 节扭转结论。
- 载具顺从对起旋的影响 → 任何顺从杀起旋。
- 驱动振幅不对称（5%）/相位抖动 对起旋的敏感性。

这些探针结构很简单（`mujoco.MjModel.from_xml_string` 建链 + 两个 mocap 钉 +
connect + 防甩钳制），半小时能重写。

---

## 附：我（Claude）对"抽象"批评的看法

用户的批评有理。当前版本物理上鸭子确实驱动绳、绳确实过顶、跳跃者确实在绳
经过时腾空——但"绳脚不相干"（无接触）和"鸭子甩得不显眼"让它看起来不像
真人跳绳。要我继续，我会先攻第 7.1（绳脚接触稳定性）和第 7.3（起旋鲁棒），
再把转头动作做得更夸张。祝接手顺利——所有实测结论都在上面了，别重踩坑。
