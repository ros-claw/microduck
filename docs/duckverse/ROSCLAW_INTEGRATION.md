# Native ROSClaw execution / 原生执行与 lineage

This integration was exercised with the actual native Pi agent, **openai-codex / gpt-6-astra / medium**, in an isolated `ROSCLAW_HOME`. The agent discovered capabilities, proposed `microduck.start_game`, waited for the native SIM action channel and observed the resulting status. No scripted chat transcript or shell shortcut substitutes for action admission.

It requires the **unmerged candidate [ROSClaw PR #632](https://github.com/ros-claw/rosclaw/pull/632)** on top of upstream `21838614bb14c39599b8acef731b2b64dad3b92a`. The extension adds the native 14-DoF Microduck simulation profile, a typed external kit, its offline timeout and declared environment forwarding. It does not weaken admission rules or enable REAL execution. The external game worker remains pinned to **MuJoCo 3.12.0**; the core environment uses 3.13.0. This is not a qualification claim for the core 3.13 Harness.

## Actual runs

| Run | Native execution receipt | Outcome | Replay |
| --- | --- | --- | --- |
| Four robots, seed 101 | `rcpt_0f235ab91ded4dd78f05eb46` | lavender wins | Exact state/control/decisions/contacts |
| Two robots, seed 510 | `rcpt_6f98e6d996ce4085b2220f42` | lavender wins | Exact state/control/decisions/contacts |

The first run's task is `task_a9da0dc2bfd34e84ba7ca55c`, mission `mis_05f48fb5a2954f2786764004`; the native task finished **SUCCEEDED / verification_passed**, without setting user acceptance. The additional two-robot action has its own consumed grant and terminal receipt in the same session/mission; it is not advertised as a separately created root task. Registered participants are `microduck-lavender`, `microduck-cream`, `microduck-sky`, `microduck-graphite`.

[Four-robot evidence](../../artifacts/duckverse-game/rosclaw/native-evidence.json) · [Two-robot evidence](../../artifacts/duckverse-game/rosclaw/native-two-evidence.json). These contain allowlisted tool results and native transaction metadata, not credentials, model private reasoning, a full session dump or a copied memory database. The four-robot native scene and trajectory digests match the selected seed-101 film source exactly. Gzip timestamps differ, so contact/decision content is verified by separate full semantic replays.

The LLM selects the mission-level start/inspect actions. It does **not** control the duck motors every frame or train the survivor. Tactical feedback and existing ONNX policies produce the actual motion.

## Installation

Use separate environments to retain the game's recorded MuJoCo version. Prepare game assets using the root README first.

```bash
# In a clone of ros-claw/rosclaw
# Check out the exact candidate PR before installing/building.
git fetch origin codex/microduck-simulation-kit
git checkout codex/microduck-simulation-kit
uv venv
uv pip install -e .
npm ci --prefix packages/rosclaw-agent
npm run build --prefix packages/rosclaw-agent

# Install the external MCP adapter without replacing core MuJoCo 3.13.
uv pip install --python .venv/bin/python --no-deps -e /absolute/path/to/microduck
export MICRODUCK_ROOT=/absolute/path/to/microduck/.assets
export MICRODUCK_PYTHON=/absolute/path/to/microduck/.venv/bin/python
export ROSCLAW_HOME=/absolute/path/to/an/isolated/rosclaw-home

.venv/bin/python -m rosclaw.entrypoint body create --robot microduck --name microduck-lavender
.venv/bin/python -m rosclaw.entrypoint body create --robot microduck --name microduck-cream
.venv/bin/python -m rosclaw.entrypoint body create --robot microduck --name microduck-sky
.venv/bin/python -m rosclaw.entrypoint body create --robot microduck --name microduck-graphite
```

Set these fields in this isolated home's `config.yaml` (retain any other configuration):

```yaml
agent:
  body_id: microduck-lavender
  engine: pi
```

Configure a supported native provider using ROSClaw's normal setup, then launch:

```bash
.venv/bin/python -m rosclaw.entrypoint chat \
  --workspace /absolute/path/to/microduck --mode SIMULATION --basic
```

Send: “Run one recorded Last Duck Standing simulation: seed 101, players 4, layout square, cadence steady. Use the registered native simulation action channel, wait for its receipt, then inspect the result.” Do not invoke the MCP side-effect tool as an observation or via shell. The kit's SIM action uses the existing DEV_SIM_ONLY policy; hardware remains forbidden.

The adapter enforces a strict bounded argument schema, single-flight execution and 600 s worker timeout. Cancellation kills and waits for the child worker. Tests cover read-only status, rejection of undeclared/invalid arguments, duplicate concurrent start, cancellation cleanup and timeout cleanup. Core native grant/idempotency semantics remain authoritative; network-disconnect endurance has not been independently qualified.

## Practice lineage

The actual four-robot audit was imported through ROSClaw's **existing RuntimeBus + PracticeRecorder** into its ordinary Practice catalog:

`mission → task → consumed grant → execution receipt → game run → prac_20261008T092059Z_b66dff → native stand/walk policy SHA`.

The official `practice verify --strict --json` command passes with no issues. [Lineage](../../artifacts/duckverse-game/rosclaw/practice-lineage.json) · [Verification](../../artifacts/duckverse-game/rosclaw/practice-verification.json).

This is explicitly a **retrospective recorded-simulation import**, not live hardware telemetry. Event timestamps map import UTC plus recorded simulation seconds. SUCCESS means the evidence import completed; individual ducks can lose. No new policy, Jev call, Darwin promotion or learning improvement is claimed. There is no parallel private Practice/Memory implementation.

```bash
# Run with the candidate core environment, which has rosclaw installed.
/path/to/core/.venv/bin/python scripts/import_duckverse_practice.py \
  --native-evidence artifacts/duckverse-game/rosclaw/native-evidence.json \
  --audit /path/to/the/native-run/audit.json \
  --data-root /path/to/new-practice-root --out /path/to/new-lineage.json
```

The released original catalog preserves original absolute artifact references. Its archived strict result pertains to the original capture location; relocating the raw catalog is not claimed to rebase those paths. A fresh import uses the supported API, creates new IDs and retains the original native receipt/audit references.

中文：这里的“接通”是实际原生 Agent 经受控 SIM 动作通道开赛、等待真实回执并读取结果。核心扩展尚待合并；模型不直接控制每个关节，也没有本期新训练。Practice 是事后导入的真实仿真事件记录，官方严格检查通过，但不代表在线学习或真机数据。
