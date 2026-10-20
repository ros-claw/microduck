# Recovery-control improvement — development review

[简体中文](README.zh-CN.md) · [English review film](media/jev-getup-en.mp4) · [English subtitles](media/jev-getup-en.srt) · [Evidence](../../artifacts/jev-getup)

This iteration responds to a visible failed recovery in the accepted English Jev survival film. It is a **new online match with revised motor recovery**, awaiting user review; no new Release or tag is published, and the accepted film is preserved.

## What was wrong

The original controller tried one bounded sitting/standing burst, then stayed with the standing policy indefinitely. In the original selected match, Dash entered recovery at 49.16 s and never completed it before elimination at 91.673 s: **42.513 s unfinished**. Reporting only its five completed recoveries (maximum 14.46 s) missed this failure. The inclusive summary now records completions, eliminations during recovery and unfinished episodes at match end.

## What changed

- Give a moving fall its original two-second natural standing opportunity. If it is still substantially toppled, try the existing learned roll for at most 0.5 s; stop early once orientation returns toward upright. Initially flat poses keep the previously calibrated sitting/standing attempt.
- If recovery remains incomplete, retry bounded sitting/standing and roll skills, separated by standing phases. A short history of improving posture postpones interruptions. The controller no longer exhausts its only attempt and waits indefinitely.
- Require actual foot support, upright posture and low horizontal speed before resuming pursuit. Hold for 0.3 s after recovery and ramp walking commands over the next 0.5 s.

These are explicit local motor reflexes, labelled separately from Jev tactics. They reuse upstream `alpha_stand`, `alpha_sitstand` and `roulade` policies; **no new policy was trained**. Contacts, native/conservative collision geometry, root dynamics, 4 kHz integration, ±0.6405 Nm duck motor limits, floor rules and elimination rules are unchanged. There is no pose reset, external lift or collision immunity.

The inherited `RECOVERY_SKILL_STARTED` event records the original scheduler's nominal sitting/standing phase; `RECOVERY_ROLL_STARTED` records its moving-fall override. **The 50 Hz decision log's `policy` field is authoritative for the actually applied policy**, and the film uses those decisions for the status cards. Both event stages are retained in exact replay.

## Evidence and limitations

| Measurement | Original film | Final new match, seed 31261 |
| --- | --- | --- |
| Completed recovery episodes | 5 | 6 |
| Completed recovery durations, seconds | 4.9, 4.8, 10.74, 3.72, 14.46 | 1.00, 3.84, 3.48, 2.62, 3.52, 7.72 |
| Longest unfinished recovery | 42.513 s, then fell | 0.22625 s, during a direct fall |
| Longest continuously low-motion topple over a remaining tile | 8.48 s | 0.04 s |
| Outcome | Sly sole survivor | Sly sole survivor, supported on tile 7 |
| Maximum all-contact penetration | 1.644 mm | 1.644 mm |

The stall metric samples at 50 Hz: local recovery, up-cosine <0.6, torso height >−0.05 m, horizontal speed <0.04 m/s, and body projection over a remaining tile. It is not a claim that every moving recovery is fast. **One new recovery still took 7.72 s under ongoing interaction.** The three unfinished episodes in the new match lasted 0.22625/0.185/0.2225 s and ended in physical elimination; the controller cannot rescue a duck already falling off the deck.

Twelve initialized-pose cases using fresh seeds 40/41/42 recovered within **3.14 s**. These are calibration poses, not population game success rates. Three controlled counterfactuals from the first development match recovered in 2.66/4.04/4.84 s with the final controller; stand-only failed within 8 s in two cases and took 4.04 s in the third. Opponent targets and hazard times follow the original recording; these interventions are not independent online games.

Both online trials and rejected alternatives are retained. Continuous sitting/standing, long rolls and a posture-terminated roll variant did not reliably recover the original stuck states. The first watchdog prototype removed the long unfinished recovery but still had completed recoveries near ten seconds. `watchdog-counterfactual.json` is approximate prototype evidence; its `graphite:52` restored a not-yet-started skill incorrectly and is excluded from final comparisons. The initializer was corrected; `v2-moving-counterfactual.json` uses fresh controller entry instead.

## Video and reproduction

The new English film is **108.10 s, 1280×720, 50 fps**, with 23 event-grounded English sports-commentary cues, native materials and the existing Dash/Waddles/Sly/Ace identities. It shows one chronological new match, including observed falls and roll-to-stand close-ups. Its labelled ending is a frozen recorded final state. Commentary, Foley, badges and slow motion do not participate in control.

Install the repository dependencies/assets and privately configure Jev credentials, then create a fresh output directory:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_jev_game.py \
  --seed 31261 --capture --out artifacts/jev-getup/local
PYTHONPATH=src .venv/bin/python scripts/check_jev_rumble.py \
  --run artifacts/jev-getup/local --out artifacts/jev-getup/local-checks.json
PYTHONPATH=src .venv/bin/python scripts/summarize_jev_recovery.py \
  --run artifacts/jev-getup/local --out artifacts/jev-getup/local-recovery.json
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_jev_rumble.py \
  --run artifacts/jev-getup/local --out artifacts/jev-getup/local-replay.json
```

Fresh live Jev results are nondeterministic. Full captures remain local; compressed raw audits/API responses, frozen actor sources, failure records and validation summaries are tracked. New captures use schema `microduck.jev-rumble.development.v5`. Reproduce older accepted v4 captures at their pinned source revision.

For film production, generate a plan with `render_jev_rumble.py --plan-only --language en`, generate speech with `commentate_jev_rumble.py --language en`, render with `--speech ... --language en`, mix with `sonify_survival_rumble.py`, and validate with `check_jev_media.py`. The generic English speech generator reads the new match's events and applied decisions; the archived fixed-English-dub script is only for the original accepted match.
