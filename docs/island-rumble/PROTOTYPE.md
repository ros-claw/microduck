# Island Rumble — contact-driven gameplay experiment

This is a separate mode, based on `duckverse-game-2026-10-08-r1`. Old scenes, rules,
releases and evidence are preserved. PR #4 and core ROSClaw PR #632 were OPEN at
inspection on 2026-10-09; neither is assumed merged.

## First experiment: two ducks, nine tiles

- Native ONNX stand/walk control, native visual materials, calibrated conservative
  head/torso collision proxies; no teleportation, external forces or new training.
- Filter actual upward normal foot load (120 ms); cap shocks at three body weights.
  Integrate load in body-weight seconds; 3.2 sustained single-duck seconds releases
  a normal tile. Two ducks bearing full weight consume it approximately twice as fast.
- Only disable the tile weld; gravity and contact solve the descent. No scheduled
  collapse permutation exists. Cosmetic cracks represent recorded damage.
- Gold centre tile is public and permanent from the start. For the first IslandClaim
  rule, from t=6s, one duck must exclusively bear weight there upright for 1.5s.
  A rival's loaded feet invalidate the timer even if the rival is toppled.
- Physical elimination retains the fall-height and unsupported debounce checks.
  No tie-breaking by name or colour. Sole survival alone is not an IslandClaim win.
- Names and coloured render-only badges follow each body. Original mesh materials,
  robot masses and collision geometries remain unchanged.

Seed 31001: contact at 6.030s, shared tile 3 releases at 7.84525s, Cream physically
eliminated at 10.28425s, Lavender wins IslandClaim at 11.0565s. Full capture 12.0565s;
chronological English greybox film 15.98s (slow actual contact/fall only).
Strict closed-loop regeneration matches every state, motor target, contact and
public decision exactly (zero error). Max contact depth 0.768mm.

Stationary fraction (sampled speed < 0.03m/s): old selected seed 101 **80.28%**, first
prototype **21.31%**. These are different matches, not a controlled treatment effect;
samples include settling, post-elimination and terminal hold. See baseline diagnosis
for ring/rapid seed 20008 (84.28%, first loss 15.221s) and cross/steady 20017
(75.69%, first loss 27.3205s). Baseline seed 101 first loss 30.62025s.

## Failed tests are evidence too

Four ducks on 3x3 / 0.36m with the same 6s opening phase failed the gameplay gate:
**0 winners / 32 matches**, although all 32 met the separately defined physical
quality limits (finite state, no reset, no external force, no solver warning,
max depth <=3mm, actuator torque <=0.640501Nm). Early crowding + slow turning
consumed the escape tiles. A 1s opening avoided that failure but four ducks could
share the gold tile indefinitely. This is why the original exclusivity rule is
not presented as a finished four-player game.

The measured locomotion sweep is retained in `artifacts/island-rumble/calibration`.
Compact dimensions 0.32/0.36/0.38m and 2.5–3.5s endurance include failed, all-dead,
and stalemated runs. A 4x4 calibration run is not a fair IslandClaim tournament:
there is no unique geometric centre, so its candidate goal has unequal spawn distances.

`prototype-a` contains an early contact-normal indexing bug (Y used instead of Z)
and is **invalid as contact-damage evidence**; b/c are development only. Only
`selected-two` plus `replay-two.json` is the verified first-prototype capture.
Source revisions must match for replay. Gzip timestamps may vary; compare semantic rows.

The 5-person entertainment panel has not been conducted. The prototype has no
ROSClaw native receipt yet; earlier Last Duck Standing receipts are not evidence
that this new mode ran through the agent. No Jev/Darwin or new RL policy is claimed.
