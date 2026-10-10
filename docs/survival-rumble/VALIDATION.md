# 本地验证 / Local validation — not accepted for release

本模式的**物理检查通过，稳定决胜检查未通过**。这份记录区分选定录像与批量玩法表现，不用一局好录像代表所有运行。完整审计 JSON 原始字节压缩在 `artifacts/survival-rumble/audits/`，索引包含原路径及压缩前后 SHA256。模型、逐步轨迹、接触、语音缓存与视频保留本地；不创建 Release 或 Tag。

**Physical checks pass; reliable survivor outcomes do not.** A selected successful film is not an all-seed robustness claim. Exact compressed audits and their hashes are tracked; large captures and the unaccepted video remain local.

## 冻结源代码与规则 / Frozen rules

Actor checkpoint: `14d6e63`. The final presentation adds speaker-targeted cameras and retains the same captured physics. 5×5 arena, 0.44 m tiles, 8 mm gaps, 4 kHz physics, 50 Hz motor control; real support-force damage and moving beacons; announced return-to-centre and two finale rings; two-axis physical gimbal with ±4 Nm environment servos. Robot joints remain limited to ±0.6405 Nm. Captures do not select a champion.

| Check / 检查 | Result / 结果 |
| --- | --- |
| Full test suite / 全套测试 | 136 passed |
| Changed Python files / 修改文件 Ruff | passed |
| Fresh seeds / 新种子 62000–62031 | 32/32 physical checks pass |
| All-contact maximum penetration / 全部接触最大穿透 | 1.644425 mm; threshold 3 mm |
| Declared sole-survivor winners / 宣布单鸭站稳获胜 | 13/32 |
| Strict final survivors / 赛后一秒仍直立承重、仅一鸭存活 | **10/32 (31.25%)** |
| Other outcomes / 其他结果 | 18 all-out draws; 1 timeout with a toppled live body; 3 declared winners fail post-match stability |
| Frozen actor source / 源代码一致 | all 32 evaluations and selected capture match |
| Selected full closed-loop replay / 选定录像完整闭环重跑 | model, states, controls, contacts and decisions exactly equal; maximum state/control error 0 |
| Final film / 最终影片 | 73.34 s, 1280×720, 50 fps; one chronological match, event slow motion, Mandarin synthetic sports voice and captions |

The physical gate requires finite states, no reset, no manually applied generalized/body forces, no solver warnings, penetration ≤3 mm, robot actuator force ≤0.640501 Nm and environment actuator force ≤4.000001 Nm. Every contact is included, including already eliminated bodies and the lower catcher. Conservative collision envelopes do not imply exact rendered-triangle collision or zero soft-contact penetration.

严格决胜门槛要求终态和继续仿真一秒后的存活列表均为同一只鸭，且最后一帧仍直立、脚底实际承重中心台。三局赛后失稳不计入成功，18 局全灭不强行指定赢家；因此预设 ≥75% 决胜目标明确未通过。尚缺失倒地后可靠起身和多鸭拥挤平衡能力，不能声称每一局都能稳定决出幸存者。当前范围仅为一张地图、四方身份出生轮换及 ±0.005 rad 关节扰动，不是任意地图或真机验证。

The strict outcome gate checks the same unique live duck both at terminal and one second later, still upright and physically supported. Three post-match instability cases are excluded. The ≥75% outcome target is explicitly **not met**. Recovery from toppling and balance in crowded contacts remain limitations. Validation covers one map, four spawn rotations and small joint perturbations, not hardware or arbitrary scenes.

## 选定影片 / Selected match: seed 61204

Source: `artifacts/survival-rumble/selected-61204/`; actor source hashes match the repository. Full-rate replay: `replay-selected.json`. The exact archived audit is `audits/unlocked-selected-61204.json.gz`.

- Five real captures: 张三 at 7.60575 s, 11.80625 s and 31.85550 s; 老六 at 14.99825 s; 二呆 at 21.58825 s. Exact times in the audit take precedence over rounded text.
- 38 segmented body-contact episodes; these are not 38 independently proven successful interceptions.
- 卷王 eliminated at 15.36650 s; 张三 at 33.27775 s; 老六 at 42.84200 s.
- 二呆 wins at 43.59200 s after 0.75 s of supported stability, 99.9% loaded duty and a maximum unsupported gap of 0.25 ms. At the final recorded state, 44.59175 s, only 二呆 remains alive, upright and supported on the centre tile.
- Selected maximum penetration: 1.644424 mm; active stationary decisions: 6.6311%.
- Finale warning: 31.85575 s; outer-ring release: 39.85575 s; inner-ring release: 42.35575 s. These scheduled releases are disclosed game rules, rather than attributed to foot load.

The film is a selected successful run. All three opponents actually fall and are eliminated; no bodies are hidden or deleted. Slow motion and camera changes only read captured states. The final five-second hold is labelled as a freeze. Speech cues refer to audit events or actual public decisions; interception is described as an attempt. Speaker-targeted close-ups show the named character. Retrospective capture calls explicitly use past tense. Speech and Foley are synthesized, not robot recordings. The final film contains 16 spoken cues; full audio/video decode and speaker-camera provenance checks pass. Measured final integrated loudness is −17.00 LUFS, audio peak 0.882, with no samples near full scale. Exact frame mapping, cue sheet, SRT and media hashes are in `artifacts/survival-rumble/media/`; decoder checks are in `media-checks.json`.

## 因果对照 / Passive counterfactual

Same seed, geometry and motor weights; all four controllers remain standing. There are zero captures and zero body-contact episodes; the announced deadline still starts the finale at 36 s. Four passive ducks physically fall after their spawn islands release, producing a draw at 44.5225 s. The active match produces travel, five captures, contacts and a supported sole survivor. This supports control-to-outcome agency; it does not establish that every elimination was caused by an opponent's push. See `passive-summary.json` and the compressed passive audit.

## 失败修复试验 / Rejected early-lock experiment

The initial gimbal can deflect under crowded loads even before the finale. We tested real joint equality locks that hold it level during roaming, then release after the inner ring. This changes actual constraints, without modifying duck poses.

Paired seeds 62000–62007: all 8 physical checks pass; strict survivors **decline from 3/8 to 2/8**. The modification is rejected, rather than silently claimed as an improvement. Source diff: `early-lock-experiment.patch`, applied to actor checkpoint `14d6e63`; raw-byte audit archives: `audits/early-lock-*.json.gz`; summary: `locked-calibration/summary.json`. Final actor source is restored exactly to the selected capture hashes.

The simpler diagnosis “locking the starting platform will fix survival” is unsupported by this experiment. More crowded-contact recovery and tactical work is still needed; presentation quality does not remove this limitation.
