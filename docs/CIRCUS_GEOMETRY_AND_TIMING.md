# Four-duck collision diagnosis / 四鸭碰撞与节奏分析

This follow-up separates formation geometry, rope length, timing feedback, and moving entry. All trials retain real robot–robot, rope–jumper, and floor contacts. Rope–turner exclusions are inherited from the baseline and remain unchanged. No new neural policy was trained.

这轮先分析“为什么撞在一起”，再做对照实验。**匹配队形和绳长后，原地双跳通过四组独立 30 秒验证，共 258/260 个共同合格圈；完整接力和无碰绳入场仍未完成。** 测试从头运行物理仿真；没有瞬移、删除碰撞或重放关节动画。

## 1. What actually collided / 碰撞先后顺序

原并排配置中，转绳鸭根位置间距 0.448 m、跳跃鸭间距 0.15 m、绳长 0.58 m。根据初始化后的真实碰撞网格，跳跃鸭之间的水平空隙只有约 **0.027 m**；最紧的外侧空隙约 **0.012 m**。这些是静态包围盒间隔，并非动态安全距离。

重新逐步记录接触后发现：

- 0.162 s：Graphite 的脚碰到 Cream 的髋部。
- 0.203 s：Sky 的脚碰到 Graphite 躯干。
- 只把绳长增加到 0.73 m、保持站位不变，仍在约 0.204 s 出现两跳跃鸭接触。

所以“绳短”不能解释全部问题。加长绳子并不会增加两只鸭之间的距离。

上一版入场失败录像则是另一条因果链：Graphite 11.080 s 启动，约 11.50 s 先失去直立姿态，11.608 s 才碰绳，11.867 s 再碰 Sky。该次**姿态失稳先于碰绳与鸭间碰撞**；第一轮录像不足以单凭视觉判断谁撞倒了谁。

Evidence: [startup contact geometry](../artifacts/circus-analysis/startup-contact-detail/original.json), [entry chronology](../artifacts/circus-analysis/entry-controls/old_entry_chronology/old_entry_chronology.json).

## 2. Controlled layout and rhythm tests / 空间与节奏对照

All rows use seed 0, 18 seconds, and the original 9-second startup exclusion. “Shared clean cycles” requires both jumpers to pass the same complete cycle. A rate above 80% is insufficient if robot collisions or other gates fail. Dimensions are metres; span means **turner root separation**, not the distance between moving mouth handles.

| Variant | Turner span | Rope length | Jumper spacing | Shared clean cycles | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Original / 原站位 | 0.448 | 0.58 | 0.15 | 0/0 | Fail |
| Longer rope only / 只加长绳 | 0.448 | 0.73 | 0.15 | 0/0 | Fail |
| Wider turners only / 只拉开转绳鸭 | 0.600 | 0.58 | 0.15 | 0/29 | Fail |
| Wider + longer / 加宽并加长 | 0.600 | 0.73 | 0.15 | 0/27 | Fail |
| Roomy + jumper spacing / 扩大双跳间距 | 0.800 | 0.93 | 0.28 | 27/27 | Pass |
| Roomy, 2.8 Hz / 宽布局降速 | 0.800 | 0.93 | 0.28 | 1/25 | Fail |
| Roomy, feedback off / 宽布局关闭反馈 | 0.800 | 0.93 | 0.28 | 16/27 | Fail |
| Tandem / 前后站位 | 0.600 | 0.73 | 0.24 | 24/27 | Fail |

宽布局（0.80 m 转绳鸭间距、0.93 m 绳、0.28 m 跳跃鸭间距）在该次试验中得到 **27/27**，全过程没有鸭间接触。启动窗口仍有碰绳，所以它不是“从第零秒起完全无接触”的结果。

关闭反馈的对照同时关闭了原有频率自适应与相位锁定，因此只能说明**整体反馈有用**，不能从这一组实验单独归因于其中某一项。降低频率至 2.8 Hz 并未改善合格率。

[Full raw layout results](../artifacts/circus-analysis/geometry-cadence/summary.json) · [Upright/contact timelines](../artifacts/circus-analysis/geometry-cadence/upright-contact-timeline.png)

## 3. Independent longer runs / 独立种子长时验证

锁定上述宽布局后，在未用于挑选该配置的 1001–1004 四个种子上各运行 30 秒，仍排除前 9 秒启动窗口。

| Seed | Shared clean cycles | Verdict |
| --- | ---: | --- |
| 1001 | 53/65 | Pass |
| 1002 | 60/65 | Pass |
| 1003 | 37/65 | Fail |
| 1004 | 0/19 | Fail |

Only **2/4 runs passed**. The fourth run also recorded a Lavender–Sky contact at 15.660 s. The candidate is **not promoted to a robust skill**. Do not substitute the selected 27/27 clip for this validation or combine this score with the existing three-duck benchmark.

这说明扩大空间解决了某些站位下的拥挤，但不能保证所有种子的起旋、足底净空和长时队形稳定。原三鸭的 85.7% 基准与这里是不同任务，不应混用。

[Holdout analysis](../artifacts/circus-analysis/roomy-holdout/analysis.json)

## 4. Moving entry / 入场控制

原实现直接给出远处目标，中心位置控制很快饱和到 0.25 m/s 的速度命令；实际身体速度并不等于命令，在旧失败记录中入场后 1.5 秒内的水平峰值约 1.14 m/s。

新增了有界移动目标：以 0.12 m/s 推进目标点，而不是一步把目标移到绳中央。还分别测试了绳外预先跳跃两秒，以及共享相位保持。这些是控制器组合，没有移动自由关节或修改实际位姿。

在相同的 0.60 m 转绳间距、0.73 m 绳、0.20 m 双跳间距下：

- 直接给目标：入场后发生倾倒，没有共同合格圈。
- 缓慢推进目标：两只跳跃鸭保持直立，末段连续 15 个共同合格圈。
- 预跳并改变启动相位：末段连续 20 个共同合格圈。

**这些入场全部失败，因为跨越绳圈边界时仍有碰绳。** 站稳和后续恢复不能替代无碰绳入场。更宽布局下的入场也单独测试，不能从原地双跳成功推断接力成功。

[Entry evidence](../artifacts/circus-analysis/entry-controls/analysis.json)

## 5. Timing experiments / 相位实验

原驱动时钟采用 `frequency * elapsed_time`，在线改变频率时会重算整个历史相位。新增可选的连续积分时钟，单元测试覆盖调频不突跳。但是，宽布局独立验证中的请求频率始终为 3.1 Hz，**这个数学隐患并未在那些试验中触发，不能拿它解释本轮所有失败**。相位反馈仍会调整时钟偏移。

另一个实验使用每只鸭足部横向位置处的实际绳段来测相位，替代共同的材料中点参考。它使用真实绳段几何，不能直接改变绳或机器人运动。下面保留未成功的对照，不把接口改进当作物理成功。

| Phase reference, 20 s | Seed 0 shared clean | Seed 1 shared clean |
| --- | ---: | ---: |
| midpoint | 33/33 | 33/34 |
| local | 15/21 | 33/34 |
| integrated | 33/33 | 33/34 |

Local-segment feedback did not improve both seeds and remains experimental. The integrated clock matched the selected success counts; it is not evidence that a clock change solved the task.

| Rope length, span 0.80 m, 20 s | Seed 0 shared clean | Seed 1 shared clean |
| --- | ---: | ---: |
| 0.82 m | 7/33 | 33/33 |
| 0.86 m | 33/33 | 33/33 |
| 0.90 m | 10/22 | 32/33 |

这些对照不支持“越长越好”或“把所有鸭子固定到同一节拍即可”。0.86 m 是本轮两个探索种子都达到 33/33 的配置；其独立验证另列，不与探索种子混算。

### Matched-length holdout / 匹配绳长后的独立验证

选定 **0.80 m 转绳鸭根间距、0.86 m 绳长、0.28 m 双跳间距** 后，使用新的 1101–1104 种子，每组运行 30 秒。旧验证种子 1001–1004 已用于诊断，因此不重复用作这次晋级证据。

| Seed | Shared clean cycles | Verdict |
| --- | ---: | --- |
| 1101 | 65/65 | Pass |
| 1102 | 64/65 | Pass |
| 1103 | 65/65 | Pass |
| 1104 | 64/65 | Pass |

**258/260 (99.23%) shared clean cycles; all four runs passed.** Each run retains the 9-second startup exclusion. This validates this static-duo configuration on these four seeds, not universal robustness, clean startup, moving entry, or a complete relay. All four runs had no inter-robot contacts after warmup. The two failed cycles remain in the denominator and their reasons remain in the raw evidence.

[Independent evidence and promotion gate](../artifacts/circus-analysis/matched-holdout/holdout-verdict.json) · [Per-seed analysis](../artifacts/circus-analysis/matched-holdout/analysis.json) · [Replay configuration](../configs/circus/duo_matched.json)

新的 27 秒完整接力尝试也保留了全过程：Sky 在 10.74 秒完成连续 5 圈；Graphite 预跳约 2.22 秒后开始缓慢移动，18.22 秒到达目标区域。两只跳跃鸭没有倾倒，四鸭之间没有接触；后半段出现连续 28 个共同合格圈。**但是 15.595 秒发生碰绳，接力状态机正确锁定失败，未计为完整接力成功。**

[Improved relay diagnostic video](../out/circus_relay_roomy_attempt.mp4) · [Relay evidence](../artifacts/circus-analysis/relay-roomy/relay-seed0.json)


## 6. Reproduce / 复现

```bash
# Replay the selected static-duo configuration, including its complete startup.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/circus_director.py rehearse \
  --config configs/circus/duo_matched.json \
  --video out/matched-duo-local.mp4 --output-dir artifacts/local/matched-duo

# Factor comparisons. Each output directory must be new.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/circus_failure_study.py \
  --group geometry --seconds 18 --workers 3 --output-dir artifacts/local/geometry
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/circus_failure_study.py \
  --group entry --seconds 23 --workers 3 --output-dir artifacts/local/entry
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/circus_failure_study.py \
  --group matched-holdout --seconds 30 --workers 4 --output-dir artifacts/local/matched-holdout

# Reporting requires NumPy and matplotlib; simulation does not require matplotlib.
python scripts/analyze_circus_study.py artifacts/local/geometry
.venv/bin/python -m pytest tests -q
```

Trials include exact configuration, source snapshots, policy hashes, per-cycle verdicts, collision chronology, and sampled poses. Current auditing also marks contact between **any two robots**, including turner–jumper contact, as a cycle fault. Source snapshots distinguish early probes from the stricter final audit. Diagnostics are tested not to alter simulated trajectories.

Selected full video: [20-second matched-length static double skipping](../out/circus_matched_duo.mp4). The earlier [18-second longer-rope clip](../out/circus_roomy_duo.mp4) remains available for comparison. It is explicitly labeled as a static-duo test with an excluded startup window, not a completed relay.

Validation: 48 tests passed; the original three-duck 20-second replay remains 33/33, with every cycle record identical to the reference.

Machine-readable index: [complete trial and video manifest](../artifacts/circus-analysis/summary.json). The rendered matched-length video has the same per-jumper cycle records as its non-rendered trial.
