# Recorded-state recovery training

[Results and limitations](../../../docs/combat/v2/body-recovery/README.md) · [中文结果](../../../docs/combat/v2/body-recovery/README.zh-CN.md)

The new registered task is `Mjlab-DuckFu-Recovery-MicroDuck-v1`. The [patch](recovery.patch) applies **after** the existing `duckfu-complete.patch`, against training commit `1bb01ff9c1780c13c753f660894f30cf96c30a18`. Actual frozen training source is `a4a22e0e9e30133bb9e81610fd254d97a5ca25a0`; frozen delivery actor source is `02301aa`. This is simulation motor training, not tactical PPO or hardware BAM training.

## Dataset and transfer checks

[Development](states/development.npz) contains 92 snapshots from four contributing trajectories and twelve recovery events. Five development games were captured; one produced no samples. [Validation](states/validation.npz) contains 80 snapshots from four whole, different trajectories and ten events. Snapshot offsets are −0.16, 0, 0.12, 0.30, 0.60, 1, 1.60 and 2.20 seconds around recovery starts. These are correlated cohorts. Validation was not used for training or selection; the training loader rejects a validation-labelled file.

Records retain native root/joint poses, velocities, previous executed actions, source times, native collider heights and source hashes. Unsupported edge-hanging initial states are excluded from the flat-floor pool by geometry, never raised into a valid pose. Raw world states remain in the NPZ. Yaw/XY normalization happens only at training initialization, rotating both velocity vectors consistently.

- `mj_objectVelocity(BODY)` returns COM-centered linear velocity. The reset requires **link-origin free-joint linear velocity**, and world angular velocity is obtained by rotating body-local angular qvel. The [independent tape comparison](development-state-provenance.json) checks all 92 records with zero extraction error; COM versus link velocity differs by as much as **0.302 m/s**.
- mjlab resets its action manager after reset events. The first observation explicitly restores the preceding executed 14 offsets, and the first action-rate cost compares against that history. Initial episode clocks remain zero. The [actual 64-environment GPU reset check](reset-state-history.json) has maximum state error **1.91e−6**.
- Final-substep contact sampling aliases stiff foot support. At the logged 3.52-second sample, the native standing actor completed **0/64** continuous-support checks with that reward, versus **60/64** after averaging actual solved foot load over the complete motor period. Both diagnostic logs and the exact diagnostic script are retained; its asset path records this machine's frozen input.

## Bounded experiments and rejected candidates

Every long run followed a 64-environment × 5-iteration smoke. Local training used 256 environments. The first sitstand-initialized run was planned for 301 iterations and stopped after last logged iteration 109, at 8:55 elapsed. Checkpoints 50 and 100 recovered **0/23** development states in the initial CPU check. It is rejected, not called a completed 301-iteration run.

The corrected standing-initialized run was planned for 251 iterations and stopped after last logged iteration 169, at 13:38 elapsed. No NaN termination was logged. [Budgets](budget.json), [all local checkpoint hashes](local-run-inventory.json), [training log](train.log.gz), [smoke](smoke.log.gz) and the [earlier rejected log](../recovery-v1/train.log.gz) remain inspectable. No cloud service was used, and training has stopped.

| Development variant | Within 3 s / 92 | Recovery → walking / 92 | Recovery median |
| --- | ---: | ---: | ---: |
| Original stand actor alone | 91 | 92 | 1.26 s |
| Corrected checkpoint 50, continuous | 89 | 92 | 1.57 s |
| Checkpoint 100, continuous | 76 | 87 | 1.57 s |
| Checkpoint 150, continuous | 92 | 90 | 1.17 s |
| **Checkpoint 150 + measured handoff** | **92** | **92** | **1.09 s** |

A contact-only checkpoint-50 guard also failed to generalize across swapped development games. It remains an explicitly optional experiment. The final [selection record](selection.json) was written before the held-out results were viewed. Later qualification exposed remaining ring-out and timestep-sensitivity limitations; weights and guard were not tuned on that audit.

The initial actor was imported from `alpha_stand.onnx`, numerically checked, paired with a fresh deterministic critic, and given cleared optimizer state. This is **not** a faithful resume of that actor's unavailable original PPO training. The exact [imported seed](checkpoints/native-stand-seed.pt.gz) and [selected full checkpoint](checkpoints/model_150.pt.gz) include their normalizers and training state. [Selected hashes](checkpoint-150.json) and [official normalized ONNX parity](export-150-parity.json) are supplied; maximum Torch/ONNX error is **3.25e−5** over 128 probes.

## Rebuild

```bash
# In the upstream training checkout at 18052aaf7d98942921a8d57319f40d3692f96bcd:
git apply /path/to/delivery/training/combat/duckfu-complete.patch
git apply /path/to/delivery/training/combat/recovery-v2/recovery.patch
uv sync
export PYTHONPATH="$PWD/src:/path/to/delivery/src"
export MICRODUCK_RECOVERY_STATES=/path/to/delivery/training/combat/recovery-v2/states/development.npz
mkdir -p logs/rsl_rl/duckfu_recovery/native-stand-clean-seed
gzip -dc /path/to/delivery/training/combat/recovery-v2/checkpoints/native-stand-seed.pt.gz \
  > logs/rsl_rl/duckfu_recovery/native-stand-clean-seed/model_0.pt

# Repeat smoke before extending training; the real work used this imported seed.
uv run python -m mjlab_microduck.train_cli Mjlab-DuckFu-Recovery-MicroDuck-v1 \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard \
  --agent.resume True --agent.load-run native-stand-clean-seed \
  --agent.load-checkpoint model_0.pt --agent.seed 1031

gzip -dc /path/to/delivery/training/combat/recovery-v2/checkpoints/model_150.pt.gz \
  > /tmp/model_150.pt
uv run scripts/export.py Mjlab-DuckFu-Recovery-MicroDuck-v1 \
  --checkpoint-file /path/to/model_150.pt --device cuda:0 --num-envs 1 \
  --onnx-file /tmp/recovery.onnx
uv run python scripts/check_duckfu_export.py \
  --checkpoint /path/to/model_150.pt --onnx /tmp/recovery.onnx --out /tmp/parity.json
```

The long run used the same command with 256 environments, `--agent.max-iterations 251`, run name `period-support-stand-v2`, and the predeclared early-stop conditions. Repeated GPU training is not promised to be bit-identical. CPU qualification uses a different backend and smaller timestep; GPU reward improvement alone never promotes a model. All fifteen targeted training tests pass. The patch applies cleanly to the prior frozen training source.
