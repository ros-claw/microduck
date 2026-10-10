> Planning document supplied by the project owner. Proposed capabilities are not delivery claims. Current implementation: [stage board](03-implementation-board.md).

我把第一期的范围定为 DUCK-GAME-01《最后一块地板》（Last Duck Standing）：先做可玩的真实物理淘汰赛，再把其中的场景、道具、技能与拍摄系统沉淀成后续 Duckverse 系列的公共基础。

核对现有代码后，有三个关键实施决定：

1. 不改坏 Neon Escape 和跳绳。 当前跑酷世界还是单鸭、连续地板结构；新建独立的竞技场模块，复用已有多鸭 MuJoCo 装配器、ONNX PolicyBank 和物理回放机制。

2. 地板必须真的塌陷。 每块地板作为独立物理刚体，先由约束支撑，触发时释放约束，在重力下掉落。MuJoCo 支持通过 `mjData.eq_active` 在运行时切换等式约束。

   [image](https://www.google.com/s2/favicons?domain=https://mujoco.readthedocs.io\&sz=32)

   MuJoCo Documentation

   +1

3. 先证明游戏，再接 AI，再拍大片。 两鸭灰盒能够稳定比赛是第一道门槛；四鸭对抗、ROSClaw 自然语言开赛、Jev 战术决策、电影化视频按后续阶段逐步加入，避免把所有不确定性堆在一起。

下面这份任务书会要求 Codex 每个阶段留下运行结果、失败证据、回归测试和可复现命令，而不是仅汇报“功能已实现”。

## 详细实施文档已整理完成

Duckverse 第一季｜Codex 完整实施任务书

Markdown · 约 37 KB · 16 个章节 · 逐 PR 验收

下载完整实施文档（.md）

文件已检查，内容完整、Markdown 代码块配对正常。

这份任务书覆盖了从物理引擎到公开视频的完整工程链路：

| 阶段       | 工作目标                                  |
| -------- | ------------------------------------- |
| DG-00～02 | 现状审计、真实塌陷地板、比赛裁判和重放                   |
| DG-03～05 | 四鸭同场、自治避险、可玩关卡与角色性格                   |
| DG-06～07 | ROSClaw Chat 真贯通、Practice、Jev 可选决策和评测 |
| DG-08～09 | 电影化拍摄、短视频、封面、多种子验收及发布                 |
| DG-10    | 后续真实学习与 Darwin 自进化                    |

特别要求 第一轮先交付一只 Duck 在真实塌陷地板上的物理验证视频，再扩展到四鸭，防止复杂功能堆叠掩盖基础问题。

## 可以直接发送给 Codex 的开工指令

请完整阅读随附的《ROSClaw × Microduck｜Duckverse 第一季工程实施任务书》，并按照文档实施 `DUCK-GAME-01：Last Duck Standing（最后一块地板）`。

具体是：~/workspace/microduck/ROSClaw_Microduck_Duckverse_DG01_Codex_Implementation_2026-10-08.md

项目继续使用现有仓库 `ros-claw/microduck`，复用已经完成的 Microduck、MuJoCo 和 ROSClaw 相关能力，不破坏跳绳和 Neon Escape 已有成果。

**总目标**：四只 Microduck 在同一个物理仿真世界中自主竞技，地板逐渐塌陷，机器人通过真实关节控制移动、避险、跌落，最终产生真实比赛结果，并制作高质量宣传视频。

实施要求：

1. 首先完整审计现有代码、资产、策略和 ROSClaw 接口，不重复开发已有组件。
2. 按文档 DG-00 至 DG-09 分阶段实施，每阶段运行真实实验并提交证据。
3. 坚持物理真实性：不修改机器人根节点制造动作，不预设赢家，不用动画冒充地板塌陷。
4. 以可玩性、动态效果、观众是否看得懂作为重要验收标准，不以单纯测试通过代替视频质量。
5. 必须区分真正接通的 ROSClaw、Jev、Practice/Darwin 能力与尚未实现的规划。
6. 每个 PR 提交代码、测试、实测报告、失败原因、截图或视频和下一阶段计划。

**第一轮优先完成 DG-00 和 DG-01**：现状审计，以及单只 Microduck 在 3×3 真实可塌陷地板上的物理实验。确认基础可行后，再继续四鸭竞技和最终作品制作。

请在实施过程中主动解决问题，复用成熟项目，不为了快速交差降低真实性要求。最终目标不仅是完成一个仿真任务，而是制作一个普通观众觉得有趣、技术人员能够复核、适合 B 站传播的机器人游戏作品。

我建议第一期先牢牢抓住一个画面：四只小鸭在不断掉落的地板上争夺最后一块安全区域。 只要这个物理场景真正做得精彩，就有机会成为 Duckverse 系列一个辨识度很高的开端。
