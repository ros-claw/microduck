# Native-opponent finishing and braking

This experiment fine-tunes the existing contact motor against a complete native Microduck on a finite octagonal platform. It is motor PPO against frozen opponent policies. Team coordination in the demonstration is a separate public-state rule controller.

The [source patch](native-finish.patch) applies to recovery training commit `a4a22e0e9e30133bb9e81610fd254d97a5ca25a0`, after the existing combat and recovery patches. Final training source is `7f96987`; the shared collision builder is supplied by this delivery repository. [Results and film](../../../docs/combat/v2/finishing/README.md) · [中文结果](../../../docs/combat/v2/finishing/README.zh-CN.md).

## Task and interface

`Mjlab-DuckFu-FinishStable-MicroDuck-v1` retains the **61D actor → 14 joint offsets**, native physical command slots, baked observation normalization and unfiltered 50 Hz output. Only its 85D critic receives privileged opponent geometry, boundary margins and episode phase. The actor has no arena coordinates: a public-state coach supplies the drive/zero-command decision.

Both robots retain their native 0.73724318 kg bodies, 14 servos, self/floor contacts and 70 massless external component hulls, with ±0.6405 Nm position-servo torque limits. Root/joint-state initialization happens only at reset. The opponent's stand and walking networks remain frozen; its joints continue driving a freely simulated body. No external push or scripted hit is applied.

Target initialization varies its distance to the edge and four facing directions. After 0.60 s settling, the student receives a bounded forward drive with live relative-bearing steering. Short drives explicitly practice stopping; longer drives practice sustained contact. A latched zero command starts at a 0.14 m inner boundary margin, at the drive limit, or after a real target exit. The walking-opponent fraction increases from 25% to 50% after 3,600 environment steps; short drives decrease from 50% to 25% after 2,400 steps. One PPO iteration collects 24 steps per environment.

Rewards pay bounded new outward target progress only with recent solved contact, upright posture, foot load and an inner margin. A one-shot finish requires the opponent's root to fall below −0.10 m, prior physical contact, and another **0.30 s** with upright >0.94, solved foot load >0.5 N, speed <0.16 m/s and own margin >0.06 m. Own falls and exits are penalized. Holding an already paid frontier or rereading termination cannot repeatedly earn a finish reward.

GPU training uses MuJoCo 3.10/Warp at 0.5 ms ×40; CPU deployment qualification uses MuJoCo 3.12/libccd at 0.125 ms ×160. These are different backends. GPU rewards and first-episode results alone cannot qualify a motor.

## Why the actor's normalizer is fixed

Early fine-tuning changed both weights and running input statistics. A diagnostic on 688 observations reconstructed from recorded physical states found a 0.132 rad RMS action change from the normalizer alone, versus 0.054 rad from weight changes alone. This is an output diagnostic, not a counterfactual proof of why any particular fall occurred. Those versions also regressed actual CPU courses; two checkpoint cohorts contained numerical rejections and are not promoted.

`DuckFuFinishStableRunner` sets the imported actor normalizer's `until=0`. Every actor weight still trains, and the privileged critic keeps learning its own statistics. No artificial sample count is added. The 64-environment ×5-iteration smoke verifies that actor statistics stay bit-identical while actor weights actually change. Official ONNX export still bakes those statistics.

The seed imports the already qualified contact ONNX actor, with a fresh 85D critic, cleared optimizer state and initial exploration standard deviation 0.06. Its 100,000-count prior is an explicit import initialization, not a claimed count of original training samples or a faithful resume of an unavailable original PPO checkpoint.

## Rebuild and train

```bash
# Begin in the upstream training checkout after applying the existing
# duckfu-complete.patch and recovery-v2/recovery.patch.
git apply /path/to/delivery/training/combat/finish-native/native-finish.patch
uv sync
export PYTHONPATH="$PWD/src:/path/to/delivery/src"
export DUCKFU_STAND_TEACHER=/path/to/assets/microduck/policies/alpha_stand.onnx
export DUCKFU_WALK_TEACHER=/path/to/delivery/policies/candidates/duckfu_locomotion_v1_200.onnx
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

uv run python -m pytest -q tests/test_duckfu_finish_cfg.py \
  tests/test_duckfu_finish_normalization.py tests/test_duckfu_contact_cfg.py \
  tests/test_duckfu_skill_transitions.py

mkdir -p logs/rsl_rl/duckfu_finish_stable/contact-seed-low-noise
gzip -dc /path/to/delivery/training/combat/finish-native/checkpoints/contact-seed.pt.gz \
  > logs/rsl_rl/duckfu_finish_stable/contact-seed-low-noise/model_0.pt

# Always repeat the small smoke before extending training.
uv run python -m mjlab_microduck.train_cli Mjlab-DuckFu-FinishStable-MicroDuck-v1 \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard \
  --agent.resume True --agent.load-run contact-seed-low-noise \
  --agent.load-checkpoint model_0.pt --agent.seed 1103

uv run scripts/export.py Mjlab-DuckFu-FinishStable-MicroDuck-v1 \
  --checkpoint-file /path/to/checkpoint.pt --device cuda:0 --num-envs 1 \
  --onnx-file /tmp/finish.onnx
uv run python scripts/check_duckfu_export.py --checkpoint /path/to/checkpoint.pt \
  --onnx /tmp/finish.onnx --out /tmp/finish-parity.json
```

Both opponent paths must be explicit; there is no silent substitution. The bounded long-run configuration uses 256 environments, a maximum of 251 iterations and a 45-minute wall limit. Actual completed iterations, retained checkpoints, rejected variants and selection are recorded in the accompanying audit files. Retraining is not promised to be bit-identical across GPUs.

| Long run | Last logged iteration (zero based) | Logged transitions | Status |
| --- | ---: | ---: | --- |
| V1, immediate exit | 18 / planned 401 | 116,736 | Stopped; did not teach post-exit braking |
| V2, prolonged drive + supported finish | 150 / planned 401 | 927,744 | Stopped; CPU support/falls regressed |
| V3, bounded coach + adaptive actor statistics | 100 / planned 301 | 620,544 | Stopped; drift and numerical rejection |
| V4, bounded coach + fixed actor statistics | 250 / planned 251 | 1,542,144 | Completed 251 iterations |

The [checkpoint manifest](checkpoints-manifest.json) distinguishes the imported seed, selected exhibition checkpoint 50, final training state 250 and rejected diagnostic checkpoint. The selected [ONNX](../../../policies/candidates/duckfu_finish_v4_050.onnx) has SHA-256 `d661bba179ced26ede2c697e01e119e23c9700c94f20cd8e55f7c683ca976475`. A later checkpoint is not automatically a better deployment choice. V2/V3 patches in this directory are alternative historical versions against the same recovery base; do not stack them with the final patch.

[Normalizer preservation at every saved checkpoint](normalizer-preservation.json) and [reconstructed diagnostic inputs](normalization-inputs.npz) make the coordinate-system change inspectable. Run `check_normalization_diagnostic.py` with the archived seed, rejected V3 checkpoint and those inputs to reproduce the output diagnostic.

## CPU qualification

`scripts/qualify_combat_finish_motor.py` tests four opponent orientations against standing, inward-walking and pressure defenses. `--bounded` supplies a 2.2 s drive budget and a 0.14 m public edge-braking signal, matching the coached motor course. Full games additionally extrapolate measured velocity 0.35 s ahead and brake before the predicted inner margin drops below 0.12 m. The earlier unbounded courses remain separately labelled stress tests; their outcomes are not mixed with bounded results. GPU first-episode probes start at the initial curriculum mix; they are not the harder, later training mix.

`scripts/eval_combat_finisher.py --finish-only` deploys a candidate only during `finish_drive` and its measured braking. Short bumps, counter drives, evasion, walking and learned recovery retain their separate motors. Swapped/rotated development games choose the assembly before held-out trials. A film additionally requires a real free 2v2 win, recent contact with a surviving winner, the complete 1× match and a separate three-second supported winner-hold audit.

The numerical gates stay at ≤3 mm maximum penetration, ≤1.5 mm penetrating-contact P99, finite states, no solver warnings, zero applied external forces, unchanged physical model and the same torque cap. These are simulation checks, not hardware validation or proof of exact mesh contact.
