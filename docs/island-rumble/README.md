# Island Rumble: contact-driven tiles and a contested crown

[简体中文](README.zh-CN.md) · [First prototype and failed rules](PROTOTYPE.md)

Four Microducks compete in one MuJoCo world. **Stand inside the white circle to
score; first to accumulate 1.5 seconds of exclusive, upright, loaded control wins.**
A rival entering the circle stops scoring. Losing ground contact stops scoring.
Moving away pauses your score; it does not erase it. Falling below the arena
eliminates a duck. An unclaimed game can still draw or time out: no forced winner.

This explicitly named **CrownRace** mode is a capture-point game, not the archived
Last Duck Standing elimination rule or the stricter continuous **IslandClaim**.
A champion can win while other ducks remain alive. The selected four-duck game
ends with three living ducks; it is not presented as a fabricated 1-v-1 final.

## Gameplay causes the map

Each normal tile has an independent free joint, supported initially by a world
weld. Actual loaded upward foot contacts (normal force >0.05N, upward normal >0.5)
feed a 120ms low-pass filter; shock peaks are capped at three body weights.

Damage integrates `filtered_load / body_weight / 3.2s`. Two fully loaded ducks
consume a tile approximately twice as fast. Damage persists, but unvisited tiles
are not released by a seeded countdown. At damage 1.0, the weld is disabled;
gravity and collisions determine the tile's descent. Cracks/colour only display
recorded damage and are depth-tested scene decorations, not extra supports.

The gold centre tile is permanent and public from the start. On the 3x3 square,
all four cardinal spawns have equal geometric distance to it. Scoring opens at
2s. The public circle radius is 0.08m, measured against the trunk XY position;
the duck must simultaneously be upright and have a loaded foot on the gold tile.
Each eligible 4kHz step earns exactly one timestep. Two bodies in the zone earn
no points, including an airborne rival contesting the zone. No unsupported
micro-gap earns points. Scores in the terminal result are frozen; post-terminal
physical settling records are not additional competition results.

Survivor, Rusher and Blocker are deterministic target-ranking roles over public
current poses, damage and nearby velocities. Rusher approaches the goal early;
Survivor favours uncrowded paths; Blocker attempts a short-horizon intercept.
All commands use the existing 50Hz ONNX stand/walk policies with ±0.6405Nm motor
limits. **Successful tactical blocking is not independently established.** There
is no Jev, new RL training, scripted fall, root teleport or external assist force.
The new mode was run locally; native ROSClaw receipts from the old game do not
certify this mode. New-mode agent integration remains unqualified.

## What changed and what passed

| Check | Result |
| --- | --- |
| Tests | 119 passed; changed-source Ruff passed |
| Frozen CrownRace seeds | 44000–44031: 32/32 explicit champions, 32/32 physics passes |
| First body contact | 1.861–1.976s |
| First physical elimination | 4.410–4.629s |
| Mean stationary fraction | 31.91% at sampled trunk speed <0.03m/s |
| Maximum signed contact depth | 1.678mm, declared limit 3mm |
| Source-bound closed-loop replay | Selected two/four matches: bitwise equal states and controls; every contact/decision regenerated |
| World-agency pair, seed 44016 | Active: 1 released tile, crown winner; passive: 4 released tiles, all eliminated |
| Winner counts | Lavender 7, Cream 8, Sky 7, Graphite 10 |

These are **32 small joint perturbations and identity rotations on the same
3x3 / 0.36m square arena**, not unseen-layout or new-skill generalization. The
sample does not establish personality balance. Stationary statistics include
settling, eliminated ducks and terminal holds; comparing them to the old seed
101's 80.28% is descriptive, not a controlled treatment estimate.

Physics passes require finite states, no time reset, no solver warnings, zero
applied external forces, torque <=0.640501Nm and penetration <=3mm. Collision
proxies are conservative massless head/torso boxes plus native convex meshes;
**not exact visual triangles or zero penetration**. Simulation is offline, not
real hardware validation. Read [all results](../../artifacts/island-rumble/qa-summary.json)
and [agency pair](../../artifacts/island-rumble/agency-four.json).

The five-person entertainment panel has **not** been conducted. Engineering
checks and improved event timing are not a substitute for independent viewers.

## Films and evidence

- [English four-duck close-ups and event slow motion](https://github.com/ros-claw/microduck/releases/download/island-rumble-prototype-2026-10-09/island-rumble-four-en.mp4)
- [English vertical cut](https://github.com/ros-claw/microduck/releases/download/island-rumble-prototype-2026-10-09/island-rumble-four-vertical-en.mp4)
- [Two-duck nine-tile greybox](https://github.com/ros-claw/microduck/releases/download/island-rumble-prototype-2026-10-09/island-rumble-two-en.mp4)
- [Uninterrupted four-duck 1x reference](https://github.com/ros-claw/microduck/releases/download/island-rumble-prototype-2026-10-09/island-rumble-four-reference-raw.mp4)

Every film uses one capture. The hero/vertical cuts begin with a **labelled later
moment of the same seed**, explicitly return to the start, then show the whole
match with event-based slow motion and a paused result. No different seeds are
stitched into one execution. Body-anchored names, colour badges, score bars and
crack marks are render-only; original robot materials are retained. Sound cues
are original synthesis aligned to events, **not physical sound recordings**.
Manifests and frame maps bind output frames to exact recorded state indices.

## Reproduce

Use the release tag rather than main: main is the series gallery, while game
code lives in the release and feature branch. Install assets as in the root
README (`scripts/bootstrap.py`, `MICRODUCK_ROOT=$PWD/.assets`). MuJoCo must be
3.12.0. Policy inference runs on CPU; EGL is used only for rendering.

```bash
git checkout island-rumble-prototype-2026-10-09
python3 -m venv .venv
.venv/bin/python -m pip install 'mujoco==3.12.0' -e '.[dev]'
export MICRODUCK_ROOT="$PWD/.assets"
.venv/bin/python scripts/bootstrap.py

# The requested small prototype; all output directories must be new.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_island_rumble.py \
  --seed 31001 --capture --out artifacts/island-rumble/my-two

# Four ducks: explicit cumulative CrownRace mode.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_island_rumble.py \
  --seed 44016 --players 4 --final-at 2 --claim-radius .08 \
  --claim-time 1.5 --score-mode cumulative --capture \
  --out artifacts/island-rumble/my-four

.venv/bin/python scripts/replay_island_rumble.py \
  --run artifacts/island-rumble/my-four --out replay.json
.venv/bin/python scripts/render_island_rumble.py \
  --run artifacts/island-rumble/my-four --mode hero --out out/my-four-raw.mp4
.venv/bin/python scripts/sonify_island_rumble.py \
  --run artifacts/island-rumble/my-four --video out/my-four-raw.mp4 --out out/my-four.mp4
.venv/bin/python -m pytest -q
```

The final two-duck capture is `selected-two-current`; the four-duck capture is
`selected-four`. Full scene/trajectory/contact binaries are in release evidence
archives. Strict replay verifies required hashes, source, policies, model bytes,
all state/control values, every semantic contact/decision row and outcome.
Actors reject source edits after import or during execution; restart the worker
before running a changed revision. Earlier two-duck `selected-two` is pinned to
checkpoint `c77113b`; it is preserved, not silently re-labelled as current code.

## Development honesty

The first four-player continuous-island rule produced 0/32 winners. A later
continuous-centre trial produced 14/32 observed winners, still below the target;
one late development row had an invalid file-read source hash after an edit
while its worker retained already-imported code. Raw audits are retained and
marked in `development-provenance-notes.json`; they are not final qualification.
This prompted import/start/end source guards. All 32 final CrownRace source
hashes match the selected capture and the selected seed reproduces its
qualification final-state hash.

CrownRace is a **published rule change**, not a claim that the earlier continuous
rule passed. The strict continuous modes remain available, and all old release
files remain unchanged. Compact locomotion measured about 0.92–1.26s to turn
before forward travel and 1.82–2.20s from first walk command to first new loaded
tile. Hence simply shortening floor endurance to 0.8s would have been unsolvable.
No unqualified celebration, jumping, skating or aggressive learned push skill
has been added to disguise limitations.
