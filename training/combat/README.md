# Duck Fu — motor training and tactical self-play

Three new motor skills address measured deployment failures: weak contact drive, insufficient lateral motion, and unstable reverse/orbit control. These are simulation policies for the fixed, torque-limited position-servo model. They are not hardware-validated BAM policies.

## Rebuild the training source

Base training commit: `18052aaf7d98942921a8d57319f40d3692f96bcd`.
Completed training source: `1bb01ff9c1780c13c753f660894f30cf96c30a18` in the local `feat/duck-fu-motor` branch.

The [complete source patch](duckfu-complete.patch) applies directly to that base. It includes all combat factories, registered tasks, custom MDP functions, the official exporter metadata and contract tests. It **replaces**, rather than stacks on, the historical R1/R2/contact patches. Set both this delivery repository and the patched training source on `PYTHONPATH`: the collision-part builder is shared with deployment.

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git duckfu-training
cd duckfu-training
git checkout 18052aaf7d98942921a8d57319f40d3692f96bcd
git apply /path/to/microduck/training/combat/duckfu-complete.patch
uv sync
export PYTHONPATH="$PWD/src:/path/to/microduck/src"
uv run python -m pytest tests/test_duckfu_cfg.py tests/test_duckfu_contact_cfg.py \
  tests/test_duckfu_skill_transitions.py -q
```

## Frozen physical and observation contracts

All tasks use the standard **61D actor observation → 14 joint offsets**, a baked observation normalizer and 50 Hz unfiltered output. Physical twist and head-pose command slots keep their original meanings. Each task keeps all native self contacts, the same robot mass/inertia and the 70 external component hulls, ±0.6405 Nm torque caps and the deployed PD gains. Reset randomization is initialization-only; there are no timed root pushes.

GPU training uses MuJoCo 3.10/Warp at 0.5 ms / 40 substeps. The final CPU game uses MuJoCo 3.12/libccd at 0.125 ms / 160 substeps. This backend/timestep difference is explicit: a GPU reward curve alone never qualifies a motor. All deployed candidates are exported by `scripts/export.py`, checked numerically against the normalized Torch actor, then exercised in the actual CPU model.

| Task | Curriculum | Selected export |
| --- | --- | --- |
| `Mjlab-DuckFu-Contact-MicroDuck-v1` | Native head tracking; drive a free full-mass 0.73724318 kg box for 1.8 s, then brake; bounded progress reward requires support | `duckfu_contact_v1_100.onnx` |
| `Mjlab-DuckFu-EvadeStep-MicroDuck-v1` | Random left/right physical lateral command for 1 s, then 0.8 s braking; reward supported lateral progress | `duckfu_evade_v1_200.onnx` |
| `Mjlab-DuckFu-Locomotion-MicroDuck-v1` | Two held physical command bursts with zero intervals; forward, reverse, lateral and combined orbit/turn commands; meaningful head tracking | `duckfu_locomotion_v1_200.onnx` |

Each long run followed a **64-environment × 5-iteration smoke**. The three bounded runs used 512 environments and a maximum of 501 iterations each; contact training stopped after 212 iterations after CPU evidence established the earlier checkpoint, while evasion and locomotion completed 501. Selected checkpoints are chosen on development tests, not held-out game outcomes. One reward version was used for each of these three corrected tasks. Historical failed R1/R2 versions remain separate.

[Selected full checkpoints](selected-checkpoints.json) include actor, critic, normalizer and optimizer, distributed as lossless `.pt.gz` archives in [checkpoints](checkpoints). The [full local checkpoint/log inventory](complete-checkpoints-manifest.json) records the remaining run artifacts. [Contact log](contact-v1.log.gz), [evade log](evade-v1.log.gz), [locomotion log](locomotion-v1.log.gz), parity reports and resource budgets are retained here. GPU retraining is not promised to be bit-identical across machines.

```bash
# Repeat the smoke before extending any experiment.
uv run train Mjlab-DuckFu-Contact-MicroDuck-v1 \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard
# Official exporter; do not hand-export an actor without its normalizer.
uv run scripts/export.py Mjlab-DuckFu-Contact-MicroDuck-v1 \
  --checkpoint-file /path/to/model_100.pt --device cpu --num-envs 1 \
  --onnx-file /tmp/contact.onnx
```

The upstream walking actor supplied an explicitly verified initialization, not a faithful resume of its unavailable original PPO checkpoint. The critic/optimizer were initialized afresh for that seed. Training uses an all-joint servo-order mapping; no head-command flag is hidden in a velocity slot and no action EMA is added at deployment. Head-bias EMA is a training reward statistic only.

## Tactical learning

High-level policies use fixed physical feature scales, 34 public inputs, two 64-unit tanh layers and nine categorical maneuvers. Initial rule-trajectory distillation is labelled separately from PPO. The first 1v1 run made twelve PPO updates with sixteen real games per update after twelve teacher games. Team training resumes an actor/critic/optimizer checkpoint and fine-tunes the shared red-team actor with frozen body policies and a frozen opponent in every episode.

Opponent sampling is 50% historical, 25% latest snapshot and 25% rules. Online training samples categorical actions; final evaluation uses the frozen actor's masked argmax. Semi-Markov discounts use actual maneuver seconds (`gamma_per_second=0.985`, `lambda_per_second=0.95`); eliminated teammates retain the real absorbing wait until the team result. Physics faults abort a batch **before its PPO update**. A run that stopped after a rejected batch is not labelled complete.

```bash
# Torch may use a different environment; rollouts must use CPU MuJoCo 3.12.
MICRODUCK_COMBAT_PYTHON=/path/to/delivery/.venv/bin/python \
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  /path/to/torch/python scripts/train_combat_tactics.py \
  --config configs/combat/match_motors_v5.yaml --updates 12 \
  --games-per-update 16 --jobs 8 --seed 100001 \
  --out training/combat/local/new-solo-training
```

[All tactical checkpoints and frozen opponent pools](tactical-checkpoints-manifest.json) are archived with update metrics, including the rejected earlier team run. Full CPU rollout manifests and transition traces are retained for [solo training](tactics-solo-v1-rollout-audit.json.gz) and [final team training](tactics-team-v5-rollout-audit.json.gz). The [distillation audit](solo-distillation-audit.json) distinguishes actual action counts from the minimum-one floor used for weighted loss.

The actual final training counts, selected weights and held-out comparison are in [results](../../docs/combat/RESULTS.md). Learned tactics are not automatically promoted because PPO ran: compare the rule, distilled and PPO actors on the same body/physics package and disclose regression or no gain. No network LLM/Jev is called by this trainer or the game loop.

[Earlier failed experiments](EARLY_EXPERIMENTS.md) · [Methods and films](../../docs/combat/README.md) · [中文项目说明](../../docs/combat/README.zh-CN.md)
