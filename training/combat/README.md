# Duck Fu motor experiments — candidates, not accepted skills

The first duel lab exposed weak attack effects. These experiments test a
simulation-only capped position-servo dash against a free **0.73724318 kg**
box. A free box is a curriculum target, not proof of defeating another robot.
No force, support impulse or root velocity is injected during an episode.

## Source and contracts

Training base: `18052aaf7d98942921a8d57319f40d3692f96bcd` in a new
`feat/duck-fu-motor` worktree. The old dirty training worktree was preserved.

- [R1 source patch](duckfu-r1.patch): completed first reward version, checkpoint
  50/149 candidates. [Current R2 patch](duckfu.patch): supported delayed impact
  payout, stopped after an unsuccessful early run.
- The delivery checkout is a training dependency for the shared
  `microduck_lab.combat.world.external_parts` geometry builder. Set both source
  directories on `PYTHONPATH`; do not quietly train with native-only geometry.
- 61D standard observation semantics, 14 standard motor outputs, 50 Hz;
  normalization included by the official exporter. There is no combat flag
  hidden inside a legacy velocity/head/body command slot.
- Native mass/inertia, position-servo gains and ±0.6405 Nm torque cap.
  GPU training: 0.5 ms / 40 substeps, MuJoCo 3.10 + Warp; CPU duel: 0.25 ms /
  80 substeps, MuJoCo 3.12/libccd. Backend/timestep differences are declared;
  CPU qualification is mandatory and failed for R1/149.
- Training keeps startup domain randomization and finite-state termination;
  runtime velocity pushes are removed. Reset is the only pose/velocity write
  boundary. Foot/head/body conventions come from the inherited task factory.

## Actual runs

1. Initial build smoke exposed a wrong reset API name; fixed to the installed
   `write_root_link_*_to_sim` methods. Failed log retained.
2. A 64-env/5-iteration build/export passed, but inspection then found inherited
   random spawn directions and a doubled height offset in the new fixed-target
   setting. That training attempt was stopped at iteration 12; it is not a
   valid combat result.
3. Corrected spawn/floor: 64 envs × 5 iterations, no NaN termination; normalized
   exporter parity passed. Walking actor import matched ONNX within about
   3.10e-6 on 128 observations. Its critic/optimizer are new initialization,
   **not the original walking PPO state**.
4. R1: 256 envs × 150 iterations, roughly 12 minutes of learning-loop wall time.
   Checkpoints 0/50/100/149. Official export parity: about 3.58e-6 max error.
   The last checkpoint failed CPU qualification, **2/40** effect-and-stable
   attacks. Earlier checkpoint 50 is a developmental film candidate only.
5. R2: delayed stable-support target payout, fresh 64-env/5-iteration smoke and
   export parity. Warm start from R1/50 with a fresh critic/optimizer. Early
   run stopped because the target-effect signal remained absent. No R2 motor
   is presented as a finished skill. See the raw stopped log for exact last
   iteration and interruption.
6. CPU target tests then located a short contact window in the original task;
   [results](target-reachability.json) and [friction variations](target-friction-reachability.json)
   are retained before any further task changes.

The initial/corrected [budgets](budget.json) cap local GPU time at 0.67 h,
checkpoints and disk use; no paid cloud compute is used. Both reward revisions
and failed configurations remain inspectable. Learned skill qualification has
not passed, and no model is promoted to Champion.

## Rebuild and smoke

Use a clean checkout of the pinned training base, its own AGENTS.md and a
working architecture-appropriate GPU environment. Install the delivery
repository and its dependencies too. Choose **one** patch from the same base;
R2 is not applied on top of R1.

```bash
git apply /path/to/delivery/training/combat/duckfu-r1.patch
export PYTHONPATH="$PWD/src:/path/to/delivery/src"
train Mjlab-DuckFu-DashBump-MicroDuck-v0 \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.logger tensorboard --agent.run-name smoke
python -m pytest tests/test_duckfu_cfg.py -q
```

Use `scripts/seed_duckfu_from_onnx.py --template <new-model_0.pt> --policy
<alpha_walking.onnx> --output <seed/model_0.pt>` for the numerically checked
actor import. It creates a candidate checkpoint, not a faithful original PPO
resume. Run the official `scripts/export.py` with `--checkpoint-file`,
`--device cpu --num-envs 1 --onnx-file`, then
`scripts/check_duckfu_export.py --checkpoint ... --onnx ... --out ...`.

Training checkpoints, normalizers (inside checkpoints), TensorBoard files and
full local logs remain under the dedicated training worktree / local evidence
directory. Compact logs, source patches, ONNX candidates and CPU reports are
tracked. There is no promise of bit-identical GPU retraining across machines.
An exact actor is identified by its weight digest, never just `best.onnx`.

[Checkpoint audit](checkpoints_manifest.json) · [Parity](candidate-149-parity.json)
· [CPU result](../../artifacts/combat/rejected-candidate-149-validation.json)
· [Runtime methods and films](../../docs/combat/README.md).
