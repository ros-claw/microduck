# Strike & Escape evidence / 证据索引

[Methods and reproduction](../../docs/STRIKE_ESCAPE.md). Six encounters combine
existing walking, standing, rolling and jumping policies with a guided physical
bowling puzzle. The gate unlocks only after a verified duck → ball → targets
contact chain topples all three targets. This is privileged-state simulation
planning with live Jev encounter selection, not vision or general game intelligence.

| Frozen cohort | Controller | Escape | All publication gates |
|---|---|---:|---:|
| `unseen-401-406` | Initial six-encounter integration | 2/6 | 1/6 |
| `unseen-501-506` | Expanded gap/exit transition planning; original first-roll trigger bypassed the experimental entry preview | 3/6 | 3/6 |
| `unseen-601-606` | Experimental entry preview actually enabled | 0/6 | 0/6 |
| `unseen-701-706` | Final default: moving entry, expanded gap/exit previews | 2/6 | 2/6 |

Each cohort freezes source and policies and attempts six new seeds once. Do not
pool versions or claim a controlled improvement: seed sets and API timing differ.
The experimental first-roll preview introduced late braking and bar brushes; its
forecast success did not satisfy the contact-free first-encounter criterion.
The default now retains moving foot-phase entry, with entry preview opt-in.
Contact limits and encounter criteria have not been relaxed to rescue failures.

`prototype-*`, `probe-*`, `realign-*`, `guides-*`, `shot-*`, `causal-*`,
`sequence-*`, `live-dev-*`, `chain-live-*`, `rubber-live-*`, `combo-live-*`
preserve development attempts and their own source snapshots. Examples include
pins sliding without toppling, direct duck hits invalidating a shot, a boss
trapped by fixed guides, failed roll transitions and self-contact failures.
`rubber-live-14` finished but failed self-contact P99 (1.260 mm); it is not a
publication-qualified video. `film-402` is a copy of cohort seed 402, not a new
attempt. Its historical controller is preserved separately from current code.

All contact categories are checked at 5 kHz: maximum penetration <1.5 mm,
P99 <1 mm, no solver warnings. Exact recorded-input replay and independent
unlock/guide-release causality are additional requirements. Planned forecasts
never substitute for executed contact evidence. Ball/target touches have no HP
penalty but remain audited. First roll and crate dodge require no corresponding
hazard contact; sweeper survival and exit roll permit brushes with HP damage.
The separate full-stunt goal still requires knockdown and recovery and is not
claimed by these escape films.

## Published capture / 精选记录

Both videos use `film-402`, identical to initial cohort seed 402. This is explicitly
an earlier controller record; its historical source is included, and current live
runs need not reproduce it. The final default cohort has two strict passes (704,
706); seed 706 also meets the separate full-stunt gate after recovery. Seeds 701
and 705 abort the exit-roll transition, 702 misses the strike, and 703 falls into
the void. These failures are not edited out of the evaluation.

The filmed seed finishes at 27.08 simulation seconds with HP 2/3. Its complete
35 s capture takes 77.407 wall seconds. Accepted Jev calls take 708–788 ms;
selected gap/sweeper/exit previews cost 0.641/3.852/1.797 wall seconds. Exact input
replay covers 7,000 states with zero position and velocity error. Causal target
parents are pin0 ← ball, pin1 ← pin0, pin2 ← pin1. Duck push impulse: 0.03740 N·s.

| Contact category | Max penetration, mm | P99, mm |
|---|---:|---:|
| Duck–floor | 0.8407 | 0.0304 |
| Duck–self | 0.2137 | 0.1816 |
| Props–floor | 0.8667 | 0.0000 |
| Props–props | 0.8470 | 0.0756 |
| Duck–hazard | 0.0000 | 0.0000 |

Zero reported penetration does not mean no force-bearing contact; soft-contact
margins can produce force before geometric overlap. Native materials and camera
close-ups show the same captured states. The action/technical cuts last 35.26 /
54.44 s. Render JSONs include capture audit, replay, contact and causal hashes.

Only the selected filmed capture is distributed as a binary release bundle.
Other JSONs and historical source archives do not imply that all scene/trajectory
files are downloadable. Credentials are excluded. Published videos use native
duck materials, simulation-time playback, three slow-motion segments each and
synthesized Foley. Camera editing never changes the simulated trajectory.

每组只跑一次；失败和旧控制器保留。新增入场预测实测退步，因此默认保留运动中的
入场方式。撞杆会扣血，接触和穿透检查照常记录；不能用预测通过来替代真实执行验收。
精选视频不代表普遍成功率，也不代表实时或硬件性能。
