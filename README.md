# ROSClaw × Microduck — Duckverse

**English** | [简体中文](README.zh-CN.md)

A series of reproducible robot games in **MuJoCo simulation**: contact-aware rope skipping, cooperative double jumping, reactive pursuit, and physical skill composition. Built with Pollen Robotics’ Microduck assets, learned ONNX motor policies, explicit controllers and contact audits. This repository contains code, weights, methods, failed experiments and replay evidence. Native ROSClaw chat launches Duckverse through a simulation-only candidate kit; hardware deployment and autonomous evolution are not demonstrated.

## Confirmed demos

Click each preview to watch or download its video. Confirmed demos are three-duck skipping, four-duck skipping, and parkour. Island Rumble remains in development; no release or tag is published until the user confirms completion.

### Cooperative Rope Skipping

[![Cooperative Rope Skipping](https://raw.githubusercontent.com/ros-claw/microduck/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/media/skipping.gif)](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/out/microduck_youtube_en.mp4)

Two ducks rotate a mouth-held flexible rope; a third jumps. Learned motor policies use measured phase feedback. Contact-based evaluation: **442/516 clean cycles (85.7%), 5/8 runs pass**. Rope–turner and rope self-collisions are excluded.

[20 s continuous](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/out/microduck_studio_full.mp4) · [Slow-motion details](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/out/microduck_closeups.mp4) · [Methods, training & reproduction](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/COOPERATIVE_ROPE_SKIPPING.md)

### Four-Duck Circus

[![Four-Duck Circus](https://raw.githubusercontent.com/ros-claw/microduck/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/media/circus_duo.gif)](https://github.com/ros-claw/microduck/releases/download/circus-pov-2026-09-17/microduck_circus_pov_en.mp4)

Two turners and two simultaneous jumpers share one world. Matching rope length and formation yields **258/260 shared clean cycles across four static runs**. The 108 s English film includes head-follow views and a failed moving entry. Impact penetration reaches **23–37 mm**; relay entry remains unsuccessful.

[Full static rollout](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/out/circus_matched_duo.mp4) · [Methods & reproduction](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/CIRCUS_GEOMETRY_AND_TIMING.md) · [Contact limitations](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/PHYSICAL_FIDELITY_AUDIT.md)

### Strike & Escape — final contact-detail edition

[![Strike & Escape — final contact-detail edition](https://raw.githubusercontent.com/ros-claw/microduck/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/out/microduck_neon_escape_v6_details_en.jpg)](https://github.com/ros-claw/microduck/releases/download/neon-escape-contact-details-v6/microduck_neon_escape_v6_details_en.mp4)

Six encounters combine rolls, a real gap, rotating-arm contact and duck–ball–pin interactions that unlock the exit. The **63 s English final cut** replays valuable details in slow motion; a 5 kHz freeze shows the actual rod brush. Its selected 35.14 s simulation replays exactly; conservative capsule/box collisions are not visual-triangle collisions. Jev decisions come from the archived live run, with a new learned finish suffix; no new success-rate claim.

[Final method & reproduction](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/CONTACT_DETAILS.md) · [Evidence bundle](https://github.com/ros-claw/microduck/releases/download/neon-escape-contact-details-v6/microduck_neon_escape_v6_evidence.tar.gz)

## How it works

1. **Motor skills:** 61D observations feed 14D learned motor actions at 50 Hz; rope turners use a separate extended contract. Joint servos have explicit torque limits.
2. **Coordination and tactics:** deterministic feedback supplies goals and phase timing. Some Neon Escape runs use live Jev for finite skill choices and physical previews for transitions; these are separate from motor-policy inference.
3. **Physics:** robots and props share one MuJoCo world. Contacts, equality forces, gravity and actuation determine outcomes. Collision profiles and exclusions are disclosed per demo.
4. **Evidence and filming:** full-rate contact checks, recorded inputs and trajectory replay verify selected results. Cameras, slow motion and synthesized Foley are presentation only.

```mermaid
flowchart LR
    Goal[Task / tactic] --> Skill[Feedback skill controller]
    Skill --> Policy[Learned motor policy / 50 Hz]
    Policy --> Motor[Torque-limited joint servos]
    Motor --> World[Shared MuJoCo world]
    World --> Skill
    World --> Audit[Contact audit and replay]
    Audit --> Video[Read-only video renderer]
```

## Run locally

Validated: Linux, Python 3.13, MuJoCo 3.12. Policy inference runs on CPU; EGL accelerates video rendering.

```bash
git clone https://github.com/ros-claw/microduck.git
cd microduck
git checkout 8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666
python3 -m venv .venv
.venv/bin/python -m pip install "mujoco==3.12.0" -e ".[dev,rosclaw]"
export MICRODUCK_ROOT="$PWD/.assets"
.venv/bin/python scripts/bootstrap.py
scripts/contact_skip.sh
.venv/bin/python -m pytest tests -q
```

For each game’s exact command, policy and asset requirements, follow its method link above. [Pinned upstream assets](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/upstream.lock.yaml) · [Training source and exports](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/training/README.md). Model/trajectory binaries and larger movies are distributed as release assets.

## Code and evidence

| Path | Purpose |
| --- | --- |
| [`src/microduck_lab/sim`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/src/microduck_lab/sim) | Robot runtime, composition and rope physics |
| [`src/microduck_lab/parkour`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/src/microduck_lab/parkour) | Physical skill combinations and props |
| [`src/microduck_lab/arena`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/src/microduck_lab/arena) | Shared-world Duckverse simulation, tactics and referee |
| [`scripts`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/scripts) | Run, evaluate, replay and render |
| [`policies`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/policies) / [`training`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/training) | Exported policies and training source |
| [`artifacts`](https://github.com/ros-claw/microduck/tree/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/artifacts) / [`docs`](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/docs/README.md) | Results, failures, manifests and methods |

## Contributing and attribution

Include reproducible commands, seed/model hashes, contact evidence and limitations with behavior changes; preserve frozen references. Project code is Apache-2.0. Robot assets and upstream policies come from **Pollen Robotics**; upstream 3D models have their separate CC BY-SA-NC terms. See [LICENSE](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/LICENSE), [THIRD_PARTY](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/THIRD_PARTY.md), and [upstream lock](https://github.com/ros-claw/microduck/blob/8ce0daab08cd4f0e3d25cb5b15e91167dc8aa666/upstream.lock.yaml). ROSClaw/Jev integrations are credited only where actually exercised; Duckverse native-agent execution is demonstrated with the linked candidate core extension; Practice imports are retrospective simulation records, not online learning.
