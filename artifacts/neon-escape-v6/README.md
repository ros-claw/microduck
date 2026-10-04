# Contact detail evidence / 接触细节证据

[Methods / 方法](../../docs/CONTACT_DETAILS.md) · [Training / 训练](../../training/README.md)

The selected **62.66 s** English cut has exactly three slow-motion sections
(gap landing, rod brush, bowling). The short finish remains at normal speed.
Capture `joysmooth300-704` replays historical live-Jev seed 704 exactly through
26.32 s (1,316 control ticks / 5,264 saved states), then continues physical
simulation. All six encounters remain verified; finish is 29.12 s,
HP 1/3. This is not an additional live API or unseen-seed trial.

最终英文版 62.66 秒，仅三段慢动作，完整闯关保留。前 26.32 秒精确重放已有
704 号 Jev 记录，后面从同一动态状态继续仿真。六段挑战通过，29.12 秒冲线，HP 1/3。
它不是新的在线 API 测试或未见种子测试；之前默认冻结测试的 2/6 严格通过率不变。

## Physical verification / 物理验证

- `audit.json`: motor-policy continuation, exact-prefix equality, warnings and HP.
- `replay.json`: **7,028 states**, max qpos/qvel error **0**.
- `continuation-reproduction.json`: using the release capture as parent and the archived policy reproduces every input/state array exactly.
- `all-contacts.json`: all force-bearing contacts at 5 kHz, all categories pass;
  maximum category overlap **0.915 mm** (duck/floor).
- `actuation-check.json`: 14 motors limited to ±0.6405 N m, zero applied robot wrench throughout, no robot/environment body weld or connect.
- `causality.json`: measured duck→ball→pins contact unlocks the gate; bounded guides.
- `sweeper-clearance.json`: rod/body minimum **+0.164 mm**,
  normal impulse **0.04294 N s**, peak **116.55 N**, no rod/body geometric overlap.
- `sweeper-highrate.npz`: hashed 5 kHz pre-integration contact-frame states (release).
- `sweeper-frame-check.json`: independent forward recomputation reproduces peak force.
- `source.tar.gz` / `parent-source.tar.gz`: selected producer/runtime and historical source.
- `previous-film-sweeper-clearance.json`: V5 had no rod force, **17.123 mm** clearance.
- `parent-704-sweeper-clearance.json`: original prefix's independent rod check.

No solver warning. The broader full-stunt gate remains false: this selected
record does not demonstrate knockdown and recovery. Hazard/body contact uses
conservative visual-enclosing boxes; floor and self-contact use native geometry.
Positive-margin force starts before geometric overlap. Other contact categories
have small compliant overlaps; the zero-overlap statement is specific to the rod.

## Continuous-policy results / 连续策略结果

Selected `continuous_joy.onnx`: checkpoint 300 of the matched SmoothPD run,
initialized from PD checkpoint 200. The matched stage performs 101 updates
through that selected checkpoint; later logged updates are unused. The
normalizer is baked through `scripts/export.py`; maximum Torch/ONNX numerical
error is 1.10e-5 over 128 inputs. Exact run/source hashes and flags: `training/run.json`.
The 50 Hz target conditioner is explicitly trained and deployed with alpha .25
and maximum step .08 rad. Target-rate limits are controller constraints, not
an independently learned achievement; physical pose checks are separate.

| Selected finish metric / 指标 | Value / 数值 |
| --- | ---: |
| Continuous policy time / 持续时间 | 4.40 s |
| Minimum upright projection / 最小直立投影 | 0.9941 |
| Maximum drift / 最大漂移 | 33.2 mm |
| Left/right foot peak / 双脚峰值 | 18.5 / 20.0 mm |
| Target RMS rate / 目标变化 RMS | 2.515 rad/s |
| Maximum target step / 最大目标单步变化 | 0.080 rad |
| Actual pose energy above 8 Hz / 实际姿态高频能量 | 0.0296% |

`probes/smooth300.json`: **4/4** small initial joint-perturbation deployment
checks pass. These are development tuning checks, not an unseen-game success
rate. The complete selected course additionally passes replay/contact/causality.

## Failed candidates / 失败候选

| Candidate | Result |
| --- | --- |
| `joy100-704` / BAM100 | Position-servo transfer failed; target RMS 58.97 rad/s, step 3.502 rad, drift 95.8 mm. |
| `joypd100-704`, `probes/pd100.json` | Upright stepping, but excessive target changes (~10.87 rad/s / 1.280 rad in course). |
| `joypd200-704`, `probes/pd200.json` | Improved, still fails unfiltered target continuity. |
| `probes/smooth200.json` | First matched-stage update: rate bound holds, but robot falls. Smoothing old weights alone does not qualify. |
| `joysmooth300-704`, `probes/smooth300.json` | Selected after matched fine-tuning; physical and continuity gates pass. |

The V5 hop/stand program was physically simulated but user review found it
visually twitchy. V5 stays archived and is superseded rather than counted as a
successful continuous gait. Statistical gates cannot certify subjective joy.

旧版庆祝经用户观看后被否定；新动作使用连续训练策略。失败记录与成功记录一起保留，
不能把“目标限速通过”替代为身体平衡通过，也不能把这几个调参初态当作未见测试。

Large scene/input/trajectory files and training checkpoints are distributed in
the release bundle for the selected capture only. Failed capture JSON/source
records do not imply every failed binary is downloadable. The archive contains
five deployed policies, selected/seed checkpoints, exact selected training source,
logs, patch, observer tools and checksums. No API key is needed for replay.
