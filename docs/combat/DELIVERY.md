# Duck Fu delivery — 2026-10-30

- Main English film: 42.00 s, 1920×1080, 50 fps.
- Vertical short: 18.60 s, 1080×1920, 50 fps.
- Technical close-ups: 100.22 s, 1920×1080, 50 fps.
- Continuous silent 2v2 reference: 49.92 s. Continuous 1v1: 45.00 s.
- English and Chinese SRT, actual-frame cover, looped normal-speed README preview.
- New drive, sidestep and walking policies; full selected motor/tactical checkpoints and training source patch.
- 832 scheduled held-out games / 64 distinct base seeds across duel/team groups, paired sides and frozen opponents. 831 numerical passes; one explicit no-contest.
- 185 delivery tests; five clean training-patch contract tests; three additional adapter checks.
- Final selected closed-loop replay: 360,000 duel and 399,360 team states, zero error.

[Results and actual limitations](RESULTS.md) · [Machine-readable evidence](../../artifacts/combat/complete-summary.json) · [Film/code hashes](../../artifacts/combat/media-manifest.json).

PPO tactics remain candidates; a stable advantage was not demonstrated. Full closed-loop outcomes depend on the frozen timestep. ROSClaw official daemon-authorized combat receipt validation remains pending; the typed SIM executor is not an authorization authority. Human viewing acceptance is pending. No Release, tag or merge is authorized by this technical delivery.

## Source and replay

Frozen final actor code: `c6e0dec`; training source: `1bb01ff9c1780c13c753f660894f30cf96c30a18`. Runtime/config/model digests: [complete source lock](../../configs/combat/complete.source.lock.yaml). Full selected capture arrays remain local under `artifacts/combat/local/team-final-reference-v2-50006` and `duel-final-reference-v2-32001`; standalone manifests and replay checks are committed. Recreate with the documented exact config/seed, then use `replay_combat.py` with the frozen runtime. Raw capture arrays are large and are not GitHub media assets. Cross-machine bit identity is not promised.

The four twelve-second motor clips and numerical-rejection diagnostic were captured under `ebc6d26`. Their actor source is preserved in `artifacts/combat/source-snapshots/competition-ebc6d26.tar.gz`; historical replay needs that source on `PYTHONPATH` and the pinned assets via `MICRODUCK_ROOT`. The later change only corrected post-match event classification, and full physical parity is recorded for both selected final matches.
