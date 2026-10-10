# 鸭鸭生存战：最后一鸭（开发中，待验收）

[English](README.md) · [验证记录](VALIDATION.md)

四只鸭在 25 块地板组成的多岛场地追逐、抢点和争位，最终只有真实存活且站稳在中心台的一只能够获胜。这是独立的新规则，不覆盖此前的计分抢岛、跳绳或跑酷实现。

| 角色 | 标记 | 控制差异 |
| --- | --- | --- |
| 张三 | 紫 | 直接抢点，拥堵后重新找路 |
| 二呆 | 黄 | 更重视地板损伤与拥挤，谨慎绕行 |
| 老六 | 蓝 | 用对手当前速度预测接近方向，尝试截路 |
| 卷王 | 珊瑚 | 追随移动目标，持续抢位 |

姓名与性格固定，种子轮换出生方位。颜色标记、姓名、小地图和字幕仅用于渲染；鸭子保留原资产材质。控制器是基于公开状态的确定性策略，底层复用已有 ONNX 行走与站立技能；这轮没有训练新策略，也没有调用语言模型控制鸭子。

## 规则与物理

地图为 5×5，每块板宽 0.44 米，板间间隙 8 毫米，外缘总宽 2.232 米。前半段连接板根据实际脚底向上支撑力积累损伤；共同承重可加快损坏。目标在中心与四方外岛间移动，连续独占且实际承重 0.75 秒记录一次抢岛，**抢岛次数不判胜**。

达到五次抢岛、36 秒，或十秒后只剩两鸭时，公开进入终局：目标返回中心，外圈在八秒预警后解除支撑约束，内圈再延后 2.5 秒。终局收圈是明确的游戏规则，不能称为全程由接触触发；此前的负载损坏仍然生效。释放后的地板由重力与真实碰撞决定运动，不直接写坠落轨迹。

中心台经两个真实铰链连接到受世界约束支撑的底座。终局收圈后，两个限力矩位置电机逐步驱动摇摆，最大目标倾角 0.14 弧度，电机各限 ±4 Nm；这些是环境执行器，与鸭子关节各自 ±0.6405 Nm 的限制分开统计。仅剩一鸭时，电机保持当时实际角度，给幸存者稳定站立的机会，不重置平台或鸭子的姿态。

只有裁判确认另外三鸭真实跌落淘汰，且剩余鸭直立、在中心台实际承重至少 0.75 秒（窗口内承重比例 ≥90%，最长失载间隔 ≤30 ms），才判唯一幸存者。两鸭仍活着不能宣布冠军；全部跌落为平局；超时不按名字或得分指定冠军。下方宽六米、顶面 -0.81 米的接落平台只接住已跌落物体，不能用于获胜。

不设置鸭子根部轨迹、不传送、不施加外力推鸭、不隐藏失败者。碰撞使用原生机器人碰撞模型，并给躯干和头部增加保守包围盒；并非逐渲染三角形精确碰撞。仿真 4 kHz、控制 50 Hz，所有接触都纳入穿透审计，包括接落平台与已淘汰身体。

## 解说与影片

同一局按时间顺序展示，碰撞、失足和抢点使用对应真实事件的特写与慢动作，终局定格明确标注。中文体育解说由 Yunjian Neural 合成；事件音效和节拍为合成音轨，不是机器人实录。

解说从该局审计事件及公开决策生成，并保留证据索引、仿真时间、视频时间和音频哈希。“老六尝试截路”只描述已记录的尝试；不会根据策略意图宣称成功封堵，也不会把一次碰撞直接说成推倒对手。配音片段不重叠，背景音在说话时自动降低；字幕与配音共用时间轴。在线语音服务输出可能变化，缓存的音频字节和哈希用于核验本次影片。

## 复现

在仓库根目录安装仿真依赖及机器人资产，输出目录须不存在：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_survival_rumble.py \
  --seed 61204 --capture --out artifacts/survival-rumble/local
PYTHONPATH=src .venv/bin/python scripts/render_survival_rumble.py \
  --run artifacts/survival-rumble/local --plan-only --out out/survival-rumble-plan.json
uv venv .tts-venv
uv pip install --python .tts-venv/bin/python -r requirements-commentary.txt
PYTHONPATH=src .venv/bin/python scripts/commentate_survival_rumble.py \
  --run artifacts/survival-rumble/local --plan out/survival-rumble-plan.json \
  --out out/survival-rumble-speech.json
PYTHONPATH=src .venv/bin/python scripts/render_survival_rumble.py \
  --run artifacts/survival-rumble/local --speech out/survival-rumble-speech.json \
  --out out/survival-rumble-silent.mp4
PYTHONPATH=src .venv/bin/python scripts/sonify_survival_rumble.py \
  --run artifacts/survival-rumble/local --video out/survival-rumble-silent.mp4 \
  --speech out/survival-rumble-speech.json --out out/survival-rumble-commentary-zh.mp4
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_survival_rumble.py \
  --run artifacts/survival-rumble/local
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_survival_rumble.py \
  --start 62000 --count 32 --workers 3 --out artifacts/survival-rumble/local-heldout
```

闭环核验会重新生成整局状态、控制、模型、接触及决策并精确比较；源代码、策略或证据变化会拒绝核验。代码在采样期间须保持冻结。测试范围只有当前地图、出生方位轮换与 ±0.005 rad 关节扰动，不能推广为任意场景或真机能力。

## 发布状态

此版仍是开发成果，未获用户最终验收。不创建 Release 或 Tag，不将它加入只展示三个已确认作品的主首页。此前计分版的证据与复现入口保留在 [relay-rumble](../relay-rumble/README.zh-CN.md)。
