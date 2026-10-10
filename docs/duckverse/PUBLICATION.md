# Videos, evidence and publication copy / 发布素材

Release: [duckverse-game-2026-10-08-r1](https://github.com/ros-claw/microduck/releases/tag/duckverse-game-2026-10-08-r1). The r1 tag contains the fresh-install asset-path fix. Films and heavy evidence are unchanged assets of the [original physics freeze](https://github.com/ros-claw/microduck/releases/tag/duckverse-game-2026-10-08).

| Asset | Length / format | Purpose |
| --- | --- | --- |
| [English hero](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_en.mp4) | 45.58 s / 1280×720 / 50 fps | Chronological seed-101 match, close-ups, slow motion, original synthesized soundtrack |
| [Vertical short](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_short_en.mp4) | 17.04 s / 720×1280 / 50 fps | Same match, final contest and real contact |
| [Technical film](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_technical_en.mp4) | 2:24.32 / 1280×720 / 50 fps | Three labelled matches, head-follow POV, foot crossing, collision geometry, physical fall and method limitations |
| [One-shot reference](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_reference.mp4) | 34.84 s / 1280×720 / 50 fps | Entire seed-101 match at 1×, fixed camera, no audio |
| [Cover](../media/duckverse-game.jpg) | 1280×720 | Actual recorded scene; native model materials |

English HUD throughout. Each film has separate `.en.srt` and `.zh-CN.srt` subtitles in the release and `out/`. The hero stays below one minute; the longer technical film is an optional methods companion. Hero/vertical music is original synthesized sound design, not measured physical audio.

## Source correspondence

[Film manifests](../../out/duckverse_game_en.json) and compressed per-frame maps record the exact state index, simulation time, playback speed and camera shot. No pose interpolation, root edit or actor intervention occurs during filming. Hero and vertical use **one** chronological seed-101 match. The technical film explicitly labels seeds **101 (square), 20008 (ring), 20017 (cross)** and every detail replay; it does not present different matches as one continuous contest. End/method cards carry no simulated-state claim.

Selected source runs and both actual native ROSClaw runs pass strict closed-loop regeneration. Video decode, dimensions, FPS, durations, asset hashes and subtitle hashes were checked; [media checks](../../artifacts/duckverse-game/media-checks.json) are preserved. Numerical contact quality does not imply zero mesh-level overlap; conservative collision boxes and soft-contact thresholds are disclosed in [methods](GAME.md).

## Evidence downloads

Each archive is below GitHub's individual release-asset limit. Extract at the repository root of this release checkout:

- [Seed 101 full-rate evidence](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_seed101_evidence.tar.gz)
- [Seed 20008 full-rate evidence](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_seed20008_evidence.tar.gz)
- [Seed 20017 full-rate evidence](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_seed20017_evidence.tar.gz)
- [Native ROSClaw two/four-robot captures and original Practice catalog](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_native_rosclaw_evidence.tar.gz)
- [All qualification/development metadata, decisions, subtitles and frame maps](https://github.com/ros-claw/microduck/releases/download/duckverse-game-2026-10-08/duckverse_game_metadata.tar.gz)

[Bundle checksums](../../artifacts/duckverse-game/release-bundles.json). Full movie and release checksums are also available as `duckverse_game_release_provenance.json`. Original Practice absolute references are preserved; see the [integration portability note](ROSCLAW_INTEGRATION.md).

```bash
.venv/bin/python scripts/replay_duckverse.py artifacts/duckverse-game/selected/101
.venv/bin/python scripts/replay_duckverse.py \
  artifacts/duckverse-game/rosclaw/native-runs/duckverse_530dc5de0b51451a
```

## Copy

**中文标题：最后一块地板：四只机器鸭的物理生存赛**

**一句话说明：四只 Microduck 依据公开预警，用学习步态换格避险；地板解除约束后真实下坠，重力、碰撞与脚底承重接触决定淘汰和冠军，ROSClaw 原生 Agent 负责开赛与核验。**

**English title:** Last Duck Standing | Four Robots, One Shrinking Arena — ROSClaw × Microduck

**English description:** Four native Microducks share a MuJoCo arena whose tiles physically fall. Current warnings guide independent controllers over existing learned walking policies; torque-limited actuation, contacts and gravity determine the result. GPT-6 Astra (medium), through the native ROSClaw SIM action channel, starts and inspects the match. Reproducible code and evidence: https://github.com/ros-claw/microduck. Simulation only; no new training or hardware claim.

Suggested tags: `Microduck`, `ROSClaw`, `MuJoCo`, `Robotics`, `PhysicalAI`, `EmbodiedAI`, `GPT6Astra`.
