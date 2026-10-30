# 功夫鸭 Duck Fu：双鸭对决与四鸭团战

[English](README.md) · **简体中文**

Duck Fu 是基于 **Pollen Robotics 原生 Microduck** 的物理仿真游戏。两只鸭子单挑，或四只鸭子组成红蓝两队，在固定八角擂台上接近、低头蓄势、突进顶撞、侧闪、转向反顶及倒地恢复。机器人没有添加手臂；动作结果来自关节驱动、真实接触、重力与支撑变化。

![功夫鸭：实际仿真画面](media/duck_fu_cover.jpg)

[英文主片](media/duck_fu_team_hero.mp4) · [竖屏短片](media/duck_fu_team_short.mp4) · [连续正常速度、无声完整比赛](media/duck_fu_team_reference.mp4) · [技术特写与慢动作](media/duck_fu_team_technical.mp4) · [连续双鸭比赛](media/duck_fu_duel_reference.mp4)

影片为 1080p、50 fps，慢放与回放均明确标记。技术版展示实际碰撞凸包、测得的接触响应及恢复过程。英文解说、合成配乐和接触触发拟音属于后期制作，不参与仿真，也不冒充机器人麦克风录音；附英文及中文字幕。慢动作读取记录下来的真实积分状态，没有生成补帧。

## 方法

**运动控制。** 每只鸭子独立维护 61 维观测及上一帧动作历史，学习策略以 50 Hz 输出 14 个关节的位置偏移。此次针对实际部署的受限 PD 模型训练了接触推进、侧向闪避和格斗步行；站立与坐起采用上游学习策略。技能结束时使用自身训练过的零指令制动阶段，根据真实足部支撑和身体、头部稳定程度完成切换。没有对策略输出增加低通滤波，也不会在切换时清空动作历史。

**战术。** 34 维公开观测输入 `34 → 64 tanh → 64 tanh → 9 动作` 分类策略，选择接近、左右绕步、突进、左右闪避、防守、回中和后退。先用明确标注的规则轨迹蒸馏进行初始化，再通过在线 PPO 与冻结的历史策略、当前快照及规则对手对抗。半马尔可夫优势估计使用真实动作持续时间；队友提前淘汰时，不会提前获得最终团队奖励。队友共享策略参数，目标分配采用公开状态下的最短距离协调，淘汰后可更换目标。

突进有低头蓄势、锁定方向、执行和制动阶段，可能扑空。对手通过实际头部姿态及近期相对运动判断威胁，不能读取隐藏攻击阶段、动作选择、未来轨迹或随机种子。动作可行性判断使用与物理擂台一致的八角形边界。

**物理。** 所有鸭子保持相同原生质量与惯性，质量为 **0.73724318 kg**，关节力矩限制为 **±0.6405 Nm**。最终模型采用 MuJoCo 3.12、Newton 求解器 80 次迭代、0.125 毫秒物理步长；电机策略仍为 50 Hz。鸭子之间通过每只 70 个无附加质量的原生可视零件凸包接触，分离碰撞掩码，避免原生碰撞体与代理重复响应；保留原生自碰撞和落地接触，队友之间同样发生碰撞。凸包对凹面仍有近似，不能称为精确三角网格接触或已经验证的真机数字孪生。

**裁判。** 真实跌出擂台并失去支撑判出界；持续无法站稳按公开的十秒倒地读秒处理。恢复需要持续、稳定的足部支撑，不能一帧站直就算成功。不使用血条、预设赢家、根节点瞬移、额外根节点外力、临时无敌或删除淘汰身体。超时及同时淘汰允许平局；数值异常判 **NO_CONTEST_PHYSICS**，不计为任何一队获胜。

角色用队色与名称区分：ROOK／张三、BOLT／莽哥、SHADE／老六、ANCHOR／稳哥。名字服务于辨识，不表示四个独立大模型人格。

## 评估与边界

[最终结果与验收](RESULTS.md) 分开报告运动、恢复、对战、物理数值与人类观看验收。[机器可读汇总](../../artifacts/combat/complete-summary.json) 索引冻结协议及压缩原始报告。完整对战采用每个对手 **32 个基础种子 × 双向换边**，并轮换场地方向；64 局是配对观测，不能当作 64 个独立种子夸大统计强度。

主片从物理通过且结束完整的比赛中选择，目的是看清交互，不能用精选影片证明成功率。另有连续正常速度完整参考及闭环重放。有效响应事件要求实际正接触冲量和可测位移／失衡，是描述性证据，不是因果伤害模型，也不直接决定胜负。

倒地后不保证每次快速站起，尤其是持续干扰及临近边缘时。旧翻滚恢复策略因自碰撞异常被拒绝，最终比赛默认停用。没有宣称已经学会短踢或翻滚逃脱。数值无效、扑空、平局及恢复失败都保留在评估里；失败训练与求解器试验见[实验记录](03_experiments.md)和[早期实验室](EARLY_LAB.zh-CN.md)。

当前战术与运动均在本地运行，**没有把 Jev 或网络大模型放进反射或战术循环**。ROSClaw 格斗 MCP 适配器提供严格类型的 SIM 工具和只读状态检查；官方 daemon 授权执行回执的完整验证仍待完成，执行器报告不冒充授权回执。没有宣称真机部署或自主进化。影片观感验收保留为待真人观看。

成片对应种子 **50006** 与 `configs/combat/team_film.yaml`：红队使用冻结的团队 PPO 战术，蓝队使用冻结的主动进攻规则，四只鸭子都使用学习得到的关节运动策略。精确复现用该配置及 `--brain learned --seed 50006 --capture`；普通示例 `team_final.yaml` 的蓝队则使用反击规则。选片偏差见[选片记录](../../artifacts/combat/team-film-selection.json)。

## 本地运行与重放

使用 `feat/duck-fu` 分支、Python 3.13 及仓库锁定的上游资产，在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -c requirements-combat.txt -e '.[dev,combat-audit]'
export MICRODUCK_ROOT="$PWD/.assets"
.venv/bin/python scripts/bootstrap.py

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_combat.py \
  --config configs/combat/team_final.yaml --brain learned --seed 50001 \
  --capture --out artifacts/combat/local/my-team-match
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_combat.py \
  --run artifacts/combat/local/my-team-match
PYTHONPATH=src .venv/bin/python scripts/combat_director.py \
  --run artifacts/combat/local/my-team-match --edition reference --out out/my-reference.plan.json
PYTHONPATH=src .venv/bin/python scripts/render_combat_film.py \
  --plan out/my-reference.plan.json --out out/my-reference.mp4
```

双鸭使用 `configs/combat/duel_final_learned.yaml`。新种子可能平局、无法提供适合剪辑的动作，或因数值问题判无效。输出必须使用新目录。规则战术配置用 `--brain rule`；不支持的模式明确报错，不静默替换。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_combat.py \
  --protocol configs/combat/eval_motors_v5.yaml --out artifacts/combat/local/my-motor-eval
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_combat_matches.py \
  --protocol configs/combat/eval_duel_final.yaml --jobs 8 --out artifacts/combat/local/my-duel-crossplay
.venv/bin/python -m pytest tests -q
```

[训练源码、导出与检查点](../../training/combat/README.md) · [物理契约](02_physics_contract.md) · [来源审计](00_source_audit.md) · [规则与范围](01_scope_and_rules.md)

代码采用 Apache-2.0，Pollen Robotics 模型与预训练策略遵循各自上游条款，见 [THIRD_PARTY](../../THIRD_PARTY.md)。用户确认完成前不发布 Release 或 tag。

[最终源码与资产锁定](../../configs/combat/complete.source.lock.yaml) · [Delivery / 交付清单](DELIVERY.md)
