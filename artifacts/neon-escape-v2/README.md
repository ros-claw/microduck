# V2 evidence index / 证据索引

These records contain development failures as well as successes. They span
multiple physical configurations and must not be pooled into one success rate.
See [methods and current blockers](../../docs/NEON_ESCAPE_V2.md).

这些文件保留开发中的失败；不同地面参数、轨道宽度、控制器版本不能混算成功率。

| Evidence | Meaning / 含义 |
|---|---|
| `v1-freeze.json` | 51 unchanged V1 implementation/evidence files |
| `velocity-envelope.json` | Initial walking command envelope; actual speed, not requested speed |
| `jump-gap-sweep.json` | Failed reuse of the old jump policy |
| `gap500-handoff.json` | 40 native-ground gap/handoff trials, selected normalized ONNX |
| `gap500-launch-window.json` | Native-ground launch-window trials (when present; see file list) |
| `impact-rotating.json` | Isolated rotating impacts; no whole-course robustness claim |
| `trip-recovery-rail-overlap.json` | Invalid earlier geometry: rail/sweeper overlap at initialization |
| `trip-recovery-battery-native.json` | Revised physical running-contact battery |
| `live-development-8/audit.json` | Selected native-ground development run; the `passed` field is the older encounter/stunt gate |
| `live-development-8/replay.json` | Recorded-input replay, zero state error |
| `live-development-8/all-contacts.json` | **Publication rejected:** native ground/self penetration exceeds the threshold |
| `unseen/summary.json` | Fixed live Jev seeds 101–106; no retries, 4/6 escapes, 1/6 earlier stunt gate |
| `unseen/frozen-source.tar.gz` | Exact source/policy snapshot for that historical cohort |
| `floor-transfer-visual-footprint.json` | 18 controlled hard-ground roll comparisons on 20 cm lanes |
| `hard-floor/` | Integrated hard-ground transfer attempts, including failures |
| `quick-dodge-hard-search.json` | Hard-contact 20 cm lane-change screening, including unstable commands |
| `training/` | Training logs, failed initialization evidence and prior reward snapshots |

## Hard-contact evidence / 硬接触证据

| Evidence | Meaning |
|---|---|
| `hard-floor/full-stunts/` | Selected **rule** baseline; all contact gates + exact replay pass; 1.08 s real recovery |
| `live-hard-5/` | Live Jev full-stunt success rejected for self-contact penetration |
| `live-hard-7/` through `live-hard-13/` | Approach/launch development failures and clean escapes without knockdown; not pooled |
| `hard-launch-self12.json` | 40 launch-distance trials on current self-contact profile |
| `gap500-hard-self12-edge06.json` | 40 width trials at a settled 6 cm launch distance |
| `hard-route-self12/`, `hard-route-middle-speed/` | 162 entry-state route calibration trials |
| `hard-roll-transfer-100.json`, `hard-roll-margin-alignment.json` | Failed deployment transfer of new roll checkpoints |
| `unseen-hard-201-206/` | Frozen current-profile live cohort, no reruns; see summary for outcomes |

Large captures are distributed separately from Git. The selected baseline's
`scene.mjb`, `inputs.npz`, `trajectory.npz`, source archive, audits and policy
files are in the [physical baseline release](https://github.com/ros-claw/microduck/releases/tag/neon-escape-v2-physical-baseline-2026-09-27).
Historical failed-run hashes do not imply their large binary captures are online;
those captures remain local. A new capture can be created with `--capture`.

完整基线证据包随实验预发布提供。其余历史失败在 Git 中保留审计、源码归档与哈希，
大体积状态仍在本机；不能把哈希说成可下载轨迹。API 凭据不进入代码或证据包。
