# Validation / 验证记录

Development evidence, 2026-10-09. No Release, Tag, hardware claim or population success-rate claim.

## Selected live match / 成片对局

Source: `artifacts/jev-rumble/phase-aware-live-70002`, schema `microduck.jev-rumble.development.v4`.

| Measurement / 项目 | Result / 结果 |
| --- | --- |
| Physical duration / 仿真时长 | 100.2815 s, including 1 s after terminal |
| Actual winner / 唯一幸存者 | 老六 / `sky`; other three physically eliminated |
| Post-terminal check / 终局后检查 | Still upright with actual centre-foot support |
| Beacon captures / 抢点 | 7; never used to select winner |
| Floor releases / 地板脱落 | 23; first at 10 s, minimum spacing 4 s |
| Warning / 预警 | At most one pending tile; ≥4 s per tile |
| Remaining map / 地图 | Connected after every release; centre never released |
| Maximum penetration / 所有接触最大穿透 | 0.0016444241809237651 m |
| Duck torque / 鸭子电机 | Maximum 0.6405 Nm |
| Platform torque / 环境平台电机 | Maximum 4 Nm, counted separately |
| Jev / 实际服务 | `jev-1.13.0`, 50 real shared-batch requests |
| Applied Jev commands / 执行记录 | 13,433 individual 50 Hz decisions citing actual request IDs |
| HTTP latency / 网络耗时 | Median 720.51 ms; maximum 801.61 ms |
| Returned API usage / API 返回用量 | 263,958 input + 10,799 output tokens; not project-development tokens |
| Solver / 求解器 | No warnings, finite state, no time reset, no externally applied forces |
| Regeneration / 闭环核验 | 401,126 steps; exact states/controls/contacts/decisions, maximum error 0 |
| Film / 影片 | 110.88 s, 5,544 frames at 50 fps; 20 Mandarin speech cues |
| Video construction / 制作 | One chronological recorded match, labelled slow motion and final freeze |

[Game checks](../../artifacts/jev-rumble/final-checks-70002.json) · [Recorded-response replay](../../artifacts/jev-rumble/final-replay-70002.json) · [Media checks](../../artifacts/jev-rumble/media/final-media-checks.json) · [API/audit archives](../../artifacts/jev-rumble/audits/index.json).

The replay regenerates public request states and revalidates raw answers at their **recorded delivery times**. It does not call the API again. Raw API usage is from returned responses. Full-rate capture ran slower than wall time: physical delivery age is not HTTP latency. The review MP4 is a CRF23 re-encode with the same frame mapping and copied AAC audio; the local high-quality film has its own manifest/hash.

闭环回放使用真实网络返回的交付时间带，不把回放冒充新在线请求。输入输出 token 仅指这局 API 实际返回的用量，不是开发项目花费。GitHub 审阅影片重编码降低体积，保留相同帧映射并复制原 AAC 音轨；本机高画质版另有哈希。配音及音效均为合成，不是实录。

## Recovery experiments / 恢复实验

Initialized poses use the actual arena collision profile, four fall directions, joint/angle perturbations and one initialization boundary. A trial first passes when 50 Hz samples show up-cosine >0.95, real foot support and full velocity norm <0.12 m/s for at least 0.15 s. This measures first recovery within 3.2 s, not permanent immunity to further falls.

| Experiment / 实验 | First recovery within 3.2 s / 首次恢复 |
| --- | --- |
| Fixed walking/turning commands / 固定行走转向对照 | 0/12 |
| Zero-command standing / 零速度站立 | 11/12 |
| Roll insertion / 插入翻滚，拒绝 | 5/12 |
| Sitting/standing burst, speed-triggered / 瞬时速度姿态触发，拒绝 | 19/24 |
| Final flat-entry phase / 最终平躺阶段，校准集 | 24/24; median 1.77 s, maximum 3.14 s |
| Fresh frozen-controller seeds 20/21/22 / 冻结后新种子留出 | 12/12; maximum 2.82 s |

Seeds 0/1/2/10/11/12 are calibration data, not a strict holdout after inspection during iteration. Only the final seeds 20/21/22 were tested after freezing and received no subsequent tuning. [Final calibration](../../artifacts/jev-rumble/phase-aware-flat-calibration.json) · [Fresh holdout](../../artifacts/jev-rumble/final-flat-heldout.json) · [Rejected skill comparisons](../../artifacts/jev-rumble/recovery-skills-calibration.json).

校准样本 0/1/2/10/11/12 在迭代时已被查看，不能宣称全部是严格留出。最终新种子 20/21/22 在控制器冻结后检验，之后未再调参。固定行走对照不是完整旧战术控制器的整局对照，不能拿它计算整局提升比例。

Three controlled rollouts start from actual recorded moving-fall states, reconstruct the previous motor action from position targets, replay opponent controls and public floor-release times, and keep all physical bodies/collisions live. Natural standing/phase-aware control recovered in 1.38/1.66/1.76 s; forcing a switch at 0.8 s took 3.28/2.18/4.06 s. Repeated bursts regressed one case, so they were rejected. These are **counterfactual fixed-input experiments**, not independent online matches or exact replay of the original motor runtime.

[Original comparison](../../artifacts/jev-rumble/recorded-recovery-comparison.json) · [Phase-aware comparison](../../artifacts/jev-rumble/recorded-phase-aware-recovery.json).

**Unresolved live limitation:** selected-match completed recoveries took 4.9, 4.8, 10.74, 3.72 and 14.46 s. The longest overlapped 36 body-contact episodes; some other falls ended in elimination before recovery. Initialized tests do not establish fast recovery under continuing impacts. Root teleportation, collision disabling and artificial protective forces are not used.

**实战慢恢复尚未彻底解决：**本局五次完成恢复耗时 4.9、4.8、10.74、3.72、14.46 秒，最长一次期间有 36 段身体接触；另有倒地后未完成恢复就跌落的情况。静态校准通过不能说成实战都能三秒起身。受控重放改变恢复控制后，其他鸭的反馈控制没有重新规划，故只用于选择运动过渡，不能当作在线胜率。

## Coverage and preservation / 覆盖与保留

The final additional online run, seed 61204, produced supported sole survivor 张三 at 82.25825 s and passed all physical/provenance gates. These two final online runs are selected, previously used seeds, not a randomized benchmark or a 100% success-rate claim. Earlier live versions include draws, two 90 s timeouts and slow recoveries; all ten run audits/API transcripts are retained in the [archive index](../../artifacts/jev-rumble/audits/index.json). Historical actor snapshots are required for historical source hashes; they must not be silently replayed with changed sources.

最终另跑的种子 61204 在 82.25825 秒产生唯一幸存者张三，并通过物理与来源检查。这两个最终在线样本都是本项目已用过的种子，不是随机化胜率评测。早期版本的全灭、两局 90 秒超时、慢恢复均有保留。不能把不同规则版本混算一个成功率。

150 full-suite tests passed locally; 14 relevant tests passed again after removing a frozen-source test's dependency on local Git history. Changed Python files passed Ruff. Serial release spacing, connectivity, identity-independent bounded grace, recovery priority, typed-answer validation, low-confidence answer preservation, actual Jev goal influence and request-state tamper rejection have focused checks.

Recovery grace is implemented and tested but **did not trigger in the selected match**. It must not be advertised as a demonstrated save in this film. Collision shapes are native robot geometry plus conservative head/torso boxes, not visual triangles. All lower-receiver and eliminated-body contacts remain in physical gates.

恢复宽限已实现且有测试，但**成片这局未触发**，不能宣传为本局的一次成功救场。完整场景、轨迹和全量接触大文件保留本机；仓库提供压缩审计、API 返回、源快照、校验结果及审阅视频。开发分支可审阅，不创建 Release 或 Tag，不加入仅展示三个已确认作品的主分支成品画廊。
