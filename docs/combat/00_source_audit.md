> Historical first-round record. Subsequent training, complete matches and current qualification are in [RESULTS.md](RESULTS.md) and the [current methods](README.md). Earlier unqualified statements below describe their named versions.

# CF-00 source audit — 2026-10-30

Development branch `feat/duck-fu` was created in a separate worktree from
`1034ed000c352476cf213ebe1544eb8c7db8b0e7`. The accepted demos and their actors
are unchanged. No Release, tag, Champion promotion or hardware action is made.

`feat/jev-dynamic-getup` at `5db0586ca451454c09796f79a985ef924b0d6c4b`
was reviewed, including its controller, tests and inclusive recovery statistics.
It is **not merged wholesale**. Its recovery controller depends on survival
observations and tile rules, so it is not directly installed into combat.
The duel currently has a conservative stand/get-up guard; dynamic contact
recovery remains unqualified. The getup branch's 7.72 s longest completed
recovery and its three direct-fall unfinished episodes do not prove fast
recovery in combat. Its inherited nominal skill-start event can differ from
the actual applied policy; combat records the applied motor policy every tick.

The existing training worktree is dirty; it was left untouched. Its committed
base is `18052aaf7d98942921a8d57319f40d3692f96bcd`. A separate
`feat/duck-fu-motor` training worktree was created at that base. The local
training source differs from the older `upstream.lock.yaml` pin, and both
ancestry and actual asset hashes are recorded rather than silently relabelled.

## Reusable contracts

- Legacy motor observation: gyro 3, projected gravity 3, relative joints 14,
  joint velocity 14, previous motor action 14, twist/head/body commands 13.
  Float32 61D input, 14D output; joint order from `sim/runtime.py`.
- Position target = HOME + raw action; no action filter or joint override.
  Action history is retained across policy switches.
- CPU inference is explicitly one thread per shared ONNX session; histories
  are per duck. Legacy `SUBSTEPS=4` is never used by combat.
- Native body mass is **0.73724318 kg per duck**, measured from the actual
  model. Torque cap is **±0.6405 Nm**, not the XML's higher ±0.96 Nm.
- Current upstream motor weights are used as **transferred priors** in a
  simulation-only capped position-servo model. This does not establish BAM
  equivalence or hardware transfer. New combat training uses position servos.

## Local resources and versions

CPU runtime: Python 3.13.12, MuJoCo 3.12.0, ONNX Runtime 1.29.0,
NumPy 2.5.2. GPU training environment: Python 3.12, PyTorch 2.9.1+cu129,
MuJoCo 3.10.0, mjlab 1.3.0, Warp 1.12.0, rsl-rl 5.0.1.
GPU: one NVIDIA GB10; driver 580.159.03. PyTorch reports its compiled
architecture support warning for this device; actual smoke execution is the
compatibility evidence. Other workloads were not stopped.

Only about 41 GB disk space was free at start. Full 4 kHz captures are selected
runs, not every training/evaluation environment. Training has a finite local
budget, checkpoint/disk limits and a recorded cancellation path.

The original training logs contain 438 substantive `.pt` files. None has been
verified as the PPO checkpoint that produced the native alpha walking/standing
weights. Tiny Wandb `.pt` pointer files are not checkpoints. The motor candidate
is initialized by importing the normalized walking actor into a new PPO
checkpoint; its optimizer and critic are not restored from the original run.
Teacher numerical parity is measured separately from new training quality.

## Regression actually executed

- Before combat implementation: **150 passed**, no skipped tests.
- With first-round combat implementation: **161 passed**, no skipped tests.
- Accepted survival, full **401,126 steps**: rules, motor histories, contact
  stream and referee regenerated, integration state/control error **0**.
  Real network answers came from the archived delivery tape, not new HTTP.

[Regression replay](../../artifacts/combat/baseline-regression/survival-replay.json)
· [Actual source/asset lock](../../configs/combat/source.lock.yaml)
· [Checkpoint inventory](../../training/combat/checkpoints_manifest.json).

The separate ROSClaw SIM boundary was reviewed: start is an authorized action,
status is read-only. No combat natural-language dispatch or receipt is claimed
in this first round. Jev is not used for the local motor/reflex loop.
