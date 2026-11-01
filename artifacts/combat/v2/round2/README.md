# Round-two evidence

The delivered actor is frozen at `693f662`; later changes add measurements, documentation and media. `final-evaluation.json` is the authoritative paired 80-game result. Its `.raw-report.json.gz` preserves the original batch report. Four base seeds are shared across five groups, revisions and sides.

- `verification.json`, `tests.log`: environment, source equality, 215-test result and evidence hashes.
- `capture-parity.json`, `replay-r1.json`, `replay-r2.json`: selected capture/noncapture equality and complete 480,000-step closed-loop replay for both revisions.
- `film-*-check.json`, `media-manifest.json`: decoded video dimensions/rate, physical frame mappings and delivered file hashes.
- `support-*-dt-check.json`: local fixed-input timestep comparisons around the two shown interventions.
- `stage-edge-dt-check.json`: a retained numerical rejection; smaller timestep did not fix it.
- `pivot-component-91003.json`: five local walking-command alternatives initialized from an actual evade exit, including root velocity and the previous executed motor action. Other motors use a recorded target tape. This is not a free-match or new training result.

Development archives A–D correspond to source patches in `source-snapshots/` applied to `3585414`. E is a separate, unmerged exploratory patch against `693f662`, with no independent qualification. These patches describe actor source; use the archived protocol configs and pinned assets to reconstruct runs. A/B forward or lateral turn variants and D's earlier projected-clearance handoff were not promoted. C supplies the retained turn behavior. E's friendly-lane changes remain local experiments.

Patches are losslessly stored as `.patch.gz`; `source-snapshots/manifest.json` records their base commits and both compressed and original SHA-256 digests. For example, on an isolated checkout of the stated base, use `gzip -dc source-snapshots/development-a.patch.gz | git apply`.

Development runs include rejections and regressions. They are not added to the frozen evaluation's success count. Large raw state/contact captures remain under `artifacts/combat/local/` in the local worktree and are intentionally excluded from Git; selected manifest hashes and frame mappings are carried with the media.

Qualified intervention definitions are unchanged from the first-round strict audit. Same-target role changes are excluded, and proximity changes are observational rather than causal saved-win claims. See [English methods](../../../../docs/combat/v2/round2/README.md) or [中文方法](../../../../docs/combat/v2/round2/README.zh-CN.md).
