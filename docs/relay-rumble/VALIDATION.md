# 本地验证 / Local validation — awaiting user acceptance

完整最终审计 JSON 以原始字节压缩保存在 `artifacts/relay-rumble/audits/`，`index.json` 记录原路径和压缩前后 SHA256。下文的 `audit.json` 路径为本地录制路径；GitHub 分支提供对应 `.json.gz` 以减少巨量逐步记分行。大型模型、轨迹、原始接触与视频保留本地，待用户确认后再考虑分发；现在可通过方法文档重新运行生成。

## 冻结的最终规则

5×5，上方可玩宽度 2.032m；五座固定安全岛、承重消耗连接板、0.105m 金圈、累计 0.75s 真实独占承重得到一分、先到三分获胜。4kHz 物理、50Hz 学习运动策略；下方接落平台顶面 -0.81m、半宽 3m。完整源代码与策略哈希见 `artifacts/relay-rumble/qa-summary.json`。

| 检查 | 结果 |
| --- | --- |
| 全套测试 | 127 passed |
| 修改的 Python 文件 Ruff | passed |
| 新种子 53000–53031 | 32/32 冠军；32/32 物理质量通过 |
| 所有接触最大穿透 | 1.568425mm，门槛 3mm |
| 首次身体接触 | 3.1625–3.3135s |
| 每场实际抢点次数 | 3–8 次 |
| 平均静止占比 | 6.2052%，只统计活动期间决策，包含开局站稳；排除淘汰和终态之后 |
| 源代码一致性 | 32 场与选定录像使用相同冻结源代码哈希 |
| 选定比赛闭环重跑 | 模型、逐步状态、控制、接触、决策一致；状态/控制最大误差 0 |

物理质量同时要求：有限状态、无时间重置、零人工施加广义/身体外力、无求解器警告、最大电机力矩不超过 0.640501Nm、全部接触穿透不超过 3mm。头部和躯干是保守包围盒，不是逐渲染三角形碰撞；不声称零软接触穿透。

这 32 个种子覆盖一张固定地图、四方角色出生轮换与关节 ±0.005rad 初值扰动，不代表任意地图的泛化能力。冠军分布：张三 3、二呆 11、老六 7、卷王 11；不据此宣称角色公平或统计上的能力排名。没有真实硬件或独立观众验收。

## 中文录像：seed 51003

本地 `out/relay-rumble-zh.mp4`：1280×720、50fps、约 48 秒，完整同局时间顺序、事件慢动作、合成音效、全场小地图和中文角色名。第三分出现时冻结胜负；末尾明确提示比赛结束后继续展示自然余势，不将赛后跌落当作夺冠依据。

- 四只鸭均有实际得分，七次抢点；卷王 3、老六 2、张三 1、二呆 1。
- 实测 31 个分段身体接触事件，涉及五组不同鸭子配对；这不是 31 次已验证的有效拦截。
- 各鸭实际承重落脚覆盖 10–18 块不同地板，活动期间累计位移约 7.78–8.02m。
- 活动期间静止占比 4.2479%，最大接触穿透 0.912222mm。
- 地板 7 在 35.3588s释放，之后二呆和老六真实跌落；第三分和跌落先后顺序以审计日志为准。

完整审计 `selected-four-wide-receiver/audit.json`；闭环验证 `replay-selected-wide-receiver.json`；逐帧状态索引、视频/轨迹/渲染源哈希及全片音视频解码检查见 `media/` 和 `media-checks.json`。

## 行为因果对照

同一 seed、同一场地和权重，让四鸭全部原地站立：65 秒没有身体接触、没有地板释放、全员零分、没有冠军。主动版本有换岛、接触、地板破坏与得分。此对照证明控制决策影响世界与结果，不单独证明每一次跌落都由对手推挤造成。原始对照见 `agency-wide-receiver.json`。

## 保留失败与修复

窄接落平台基线（种子 52000–52031）：32/32 冠军，但仅 31/32 通过物理门槛。失败的 52013 在 24.49125s 出现 6.922346mm 穿透：跌落鸭的左脚撞上接落平台外侧棱角。接触记录诊断保留在 `diagnose-52013/penetration-diagnosis.json`，不把该局算作通过。

修复只扩大下方接落平台，不关闭碰撞、不改变门槛、不增加人工外力。最终源代码回归相同失败种子，最大穿透为 0.718038mm；再以全新 53000–53031 验证上述结果。完整 MuJoCo 世界包含下方接触，因此几何修改可能改变最终胜负，不声称与旧世界逐帧相同。

早期 trial、窄平台录像与中间源代码只用于开发诊断，不能替代冻结后最终证据。旧基线源代码检查点为 `036cc96`；最终核验以逐文件 SHA256 为准。

## 发布状态

未验收、不创建 Release 或 Tag。此前 11 个未经确认或已被最终版取代的 Releases 和 Tags 已移除；公开展示保留三鸭跳绳、四鸭跳绳、最终跑酷。历史代码和本地证据保留。

English: The final frozen 5×5 rules pass 32/32 physical and outcome checks on one map; maximum penetration is 1.568425mm. All 127 tests pass. The selected Chinese film regenerates every state, control, contact and decision exactly. The failed narrow-receiver baseline is retained separately. These are simulation results, not hardware validation, successful-blocking proof or viewer acceptance. No new release or tag is authorized.
