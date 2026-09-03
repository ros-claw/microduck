"""Record the Microduck Circus hero video, scene by scene (with caching)."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import numpy as np

from microduck_lab.demos import circus
from microduck_lab.skills.skip import SkipConfig
from microduck_lab.video.overlay import title_card
from microduck_lab.video.writer import VideoWriter

OUT = pathlib.Path("~/workspace/microduck/rosclaw-microduck/out").expanduser()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--fast", action="store_true", help="skip practice re-shoot (reuse)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seg = OUT / "segments"
    seg.mkdir(exist_ok=True)

    champion = json.loads((OUT.parent / "practice/champion.json").read_text())
    champion_cfg = SkipConfig(**{k: v for k, v in champion["params"].items()
                                 if k in SkipConfig.__dataclass_fields__})

    segs = []

    # S1: title
    p = seg / "s1_title.mp4"
    if not p.exists():
        wr = VideoWriter(str(p), circus.FPS)
        wr.write_n(title_card("MICRODUCK CIRCUS", "ROSClaw × Pollen Microduck — 三只鸭子跳长绳",
                              (circus.W, circus.H)), int(2.4 * circus.FPS))
        wr.close()
    segs.append(p)

    # S2: intro
    p = seg / "s2_intro.mp4"
    if not p.exists() or "intro" in args.scenes:
        print("shooting intro...", flush=True)
        circus.scene_intro(p)
    segs.append(p)

    # S3: first attempt
    p = seg / "s3_first.mp4"
    if not p.exists() or "first" in args.scenes:
        print("shooting first attempt...", flush=True)
        m = circus.scene_first_attempt(p)
        print("  first attempt:", m.successful_skips, "/", m.crossings, "skips", flush=True)
    segs.append(p)

    # S35: teach a duck to jump (the RL payoff)
    p = seg / "s35_jump_evolution.mp4"
    if not p.exists() or "jump" in args.scenes:
        print("shooting jump evolution...", flush=True)
        from microduck_lab.demos.jump_scene import scene_jump_evolution
        scene_jump_evolution(p)
    segs.append(p)

    # S4: practice montage
    p = seg / "s4_practice.mp4"
    if not p.exists() or "practice" in args.scenes:
        print("shooting practice montage...", flush=True)
        variants = [
            ("lead=0.45", SkipConfig(trigger_lead_s=0.45, seed=304)),
            ("lead=0.55", SkipConfig(trigger_lead_s=0.55, seed=311)),
            ("lead=0.62", SkipConfig(trigger_lead_s=0.62, seed=314)),
            ("lead=0.70", SkipConfig(trigger_lead_s=0.70, seed=305)),
            ("f=0.95", SkipConfig(frequency=0.95, seed=306)),
            ("f=1.15", SkipConfig(frequency=1.15, seed=307)),
            ("f=1.30", SkipConfig(frequency=1.30, seed=308)),
            ("blind", SkipConfig(seed=309, blind=True)),
        ]
        labels = circus.scene_practice(p, variants)
        for l in labels:
            print("  tile:", l, flush=True)
    segs.append(p)

    # S5: evolution report
    p = seg / "s5_evolution.mp4"
    if not p.exists() or "evolution" in args.scenes:
        print("shooting evolution report...", flush=True)
        circus.scene_evolution(p, OUT.parent / "practice/rope_skip.jsonl", champion)
    segs.append(p)

    # S6: performance (champion)
    p = seg / "s6_performance.mp4"
    if not p.exists() or "perf" in args.scenes:
        print("shooting performance...", flush=True)
        m, consec = circus.scene_performance(p, champion_cfg)
        print(f"  performance: {m.successful_skips}/{m.crossings} skips, streak {consec}", flush=True)
    segs.append(p)

    # S7: celebration
    p = seg / "s7_celebration.mp4"
    if not p.exists() or "celebration" in args.scenes:
        print("shooting celebration...", flush=True)
        circus.scene_celebration(p)
    segs.append(p)

    # S8: end card
    p = seg / "s8_end.mp4"
    if not p.exists():
        wr = VideoWriter(str(p), circus.FPS)
        wr.write_n(title_card("Chat → Act → Practice → Learn → Evolve",
                              "ROSClaw — Tell them what you want.", (circus.W, circus.H)),
                   int(3.0 * circus.FPS))
        wr.close()
    segs.append(p)

    final = OUT / "microduck_circus.mp4"
    print("assembling...", flush=True)
    circus.assemble(segs, final)
    print("DONE:", final, flush=True)


if __name__ == "__main__":
    main()
