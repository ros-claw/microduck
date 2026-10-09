# Last Duck Alive — development preview

[简体中文](README.zh-CN.md) · [Validation](VALIDATION.md)

Four native Microduck robots pursue moving objectives across a 25-tile arena. Captures create encounters; the only victory condition is one physically surviving, upright, supported duck after the other three have actually fallen and been eliminated. This separate ruleset preserves the earlier score-based relay and all confirmed demos.

| Character | Badge | Public-state tactic |
| --- | --- | --- |
| 张三 / Zhang San | Purple | Rush objectives; replan after congestion |
| 二呆 / Er Dai | Yellow | Prefer less damaged, less crowded routes |
| 老六 / Lao Liu | Blue | Attempt interception using a rival's current position and velocity |
| 卷王 / Juan Wang | Coral | Chase successive objectives |

Identity and personality stay bound; seeds rotate spawn directions. Names, badges, minimap and subtitles are presentation overlays. Native robot materials remain intact. Deterministic tactical controllers select existing ONNX walking and standing skills; no new policy training or language-model robot control is claimed here.

## Mechanics and physical implementation

The 5×5 arena uses 0.44 m tiles and 8 mm gaps, spanning 2.232 m. In the roaming phase, actual upward foot support forces accumulate bridge damage. Multiple loaded ducks accelerate deterioration. A public beacon cycles between the centre and four cardinal outposts. An upright, exclusively occupying, physically supported duck earns a capture after 0.75 s. **Captures never declare a winner.**

Five captures, 36 seconds, or two remaining ducks after ten seconds trigger a public finale. The objective returns to the centre; the outer ring loses its support constraints after eight seconds, followed by the inner ring 2.5 seconds later. Earlier load-driven failures remain active. This announced finale is a timed game rule, not a claim that every release is caused by contact. Released tiles move through gravity and contact, without prescribed falling trajectories.

The centre tile is mounted on two real hinges above a welded base. After the rings release, two position servos gradually command a rocking motion, with a maximum target amplitude of 0.14 rad and a ±4 Nm limit per environment motor. Robot joints retain their separate ±0.6405 Nm limit. When one duck remains, the servos hold the measured platform angles so it can settle; no body or platform pose is reset.

Victory requires three recorded physical eliminations and one upright survivor with actual centre-tile foot support for at least 0.75 s, ≥90% support duty, and no unsupported gap over 30 ms. Two live ducks cannot produce a champion. All eliminated means a draw; timeout does not select a winner by score or identity. The six-metre catcher below the arena, whose top is at −0.81 m, only catches fallen bodies and cannot support a winning claim.

There are no duck root trajectories, teleports, external pushing forces, predetermined winners or hidden losers. Native collision geometry is augmented with conservative torso/head envelopes, rather than exact rendered-triangle collision. Simulation runs at 4 kHz with 50 Hz control. Penetration audits include every contact, including catcher contacts and eliminated bodies.

## Film and commentary

The film follows one match chronologically, using event-aligned collision and fall close-ups, slow motion and a labelled final freeze. Mandarin sports commentary uses the synthetic Yunjian Neural voice. Foley and the rhythmic bed are synthesized, rather than measured robot sound.

Each spoken cue references the recorded event or public decision, simulation time, video time and audio hash. An interception intention is described as an *attempt*, never assumed successful. A collision alone is not evidence that one duck pushed another off. Speech does not overlap, the background ducks during narration, and captions share the speech timeline. Online synthesis is not bit-deterministic; cached bytes and hashes identify the actual delivered audio.

## Reproduce

Install simulator dependencies and native robot assets first. Run from the repository root; simulation output directories must not already exist. The [Chinese guide](README.zh-CN.md#复现) contains the complete commands.

1. `scripts/run_survival_rumble.py --seed 61204 --capture --out ...` records the match.
2. `scripts/render_survival_rumble.py --plan-only ...` creates the chronological edit plan.
3. Install the optional voice dependency in `.tts-venv` using `requirements-commentary.txt`; `scripts/commentate_survival_rumble.py` creates cached speech, an evidence-linked cue sheet and SRT subtitles.
4. `scripts/render_survival_rumble.py --speech ...` renders native recorded states and captions.
5. `scripts/sonify_survival_rumble.py --speech ...` mixes commentary and event sound.
6. `scripts/replay_survival_rumble.py` regenerates and exactly compares the complete closed loop.
7. `scripts/eval_survival_rumble.py --start 62000 --count 32 --workers 3 --out ...` evaluates fresh seeds.

Set `PYTHONPATH=src OPENBLAS_NUM_THREADS=1` when running simulation or verification. Actor sources must stay frozen while workers run. Replay rejects changed source, weights, models or evidence. Validation covers one layout, cardinal spawn rotations and ±0.005 rad joint perturbations; it is not hardware validation or a general arbitrary-map success claim.

## Publication status

This remains unaccepted development work. It creates no Releases or Tags and does not add an unconfirmed game to the public gallery of three accepted demos. Historical score-based results remain in [relay-rumble](../relay-rumble/README.md).
