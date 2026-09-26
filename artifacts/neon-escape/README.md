# Neon Escape evidence

- `jev-first-call.json`: first real request; private authentication is never recorded.
- `jev-spike-strict-validation.json`: first 1,000-request batch; 27 rounded distributions rejected locally, not HTTP failures.
- `probability-rounding.json`: direct check of real API probability sums.
- `jev-spike.json`: corrected 1,000-request benchmark, four concurrent workers, 1,000 valid responses. This fixed hypothetical skill-ready state predates the refined game prompt.
- `probe-*.json`, `skill-sweep.json`, `roll-*-probe.json`: motor transfer and physical clearance experiments, including failed heights and unstable forward hopping.
- `baseline-*`, `brake-align-probe`, `jev-seed0`, `jev-v*`: exploratory versions, not repeated trials of one frozen controller. Do not aggregate their pass rates. They preserve stationary failures, collisions, and control-interface changes.
- `final-baseline-*`, `final-jev-*`: final scene/control settings on development seeds 0/101/102. Jev is paced in wall time; the baseline uses the same physics time but can execute faster than wall time.
- `holdout-jev-*`: first tests on fresh seeds 9001/9002/9003 after controller freeze. No tuning based on these outcomes.
- `film-jev-seed102`: deterministic re-simulation of the recorded decisions in `final-jev-seed102`; exact position/upright/action trace, contact audit and skill-event equality verified. Not an additional live-model evaluation.
- `verified-inversion-seed102`: the same replay also passes the strengthened roll gate requiring actual body inversion; identical original outcomes.
- `evaluation.json`: development and fresh-seed outcomes, including every failure and measured body inversion.

MJB/NPZ caches are local, ignored, and regenerated with `scripts/replay_neon_run.py`. The video is a Release asset. Early exploratory versions did not save complete source snapshots; use final recorded decisions plus equality-checked replay for the reproducible demonstration. These small sample sets do not establish general reliability or model superiority.
