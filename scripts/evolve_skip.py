"""Run a real Failure → Practice → Adapt → Darwin → Champion loop for rope skipping."""
import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from dataclasses import replace

from microduck_lab.skills.skip import RopeSkipSession, SkipConfig
from microduck_lab.learning.practice import PracticeLog, PracticeRecord
from microduck_lab.learning import adapt

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"
BANK = {"stand": str(POL / "alpha_stand.onnx"), "walk": str(POL / "alpha_walking.onnx"),
        "sitstand": str(POL / "alpha_sitstand.onnx")}

TRAIN_SEEDS = [101, 202, 303, 404, 505, 606]
HOLDOUT_SEEDS = [777, 888, 999, 111]   # Darwin only; the search never sees these

log = PracticeLog(ROOT / "rosclaw-microduck/practice/rope_skip.jsonl")


def evaluate(cfg: SkipConfig, seeds, tag: str, version: str) -> adapt.EvalResult:
    eps = []
    for sd in seeds:
        c = replace(cfg, seed=sd)
        s = RopeSkipSession(str(ROBOT), BANK, c)
        s.settle()
        s.start_rope()
        m = s.run_episode(16.0)
        eps.append(m)
        log.record(PracticeRecord(
            skill_id="microduck.rope_skip", skill_version=version,
            params={k: getattr(c, k) for k in ("trigger_lead_s", "jump_crouch", "frequency")},
            seed=sd, success=m.success_rate > 0.5,
            metrics={"crossings": m.crossings, "success": m.successful_skips,
                     "trips": m.trips, "consec": m.consecutive_best,
                     "turner_up": m.turner_upright_frac, "jumper_up": m.jumper_upright_frac,
                     "jumper_max_down_s": m.jumper_max_down_s,
                     "turner_max_down_s": m.turner_max_down_s},
            failure=None if m.successful_skips else "no_success",
            duration_s=m.duration_s))
    r = adapt.EvalResult.of(eps)
    print(f"  [{tag}] rate={r.success_rate:.0%} consec={r.consecutive_best} "
          f"up(t/j)={r.turner_upright:.0%}/{r.jumper_upright:.0%} "
          f"down(j)={r.jumper_max_down_s:.1f}s", flush=True)
    return r


def main(rounds: int = 4):
    print("=== ROSClaw evolution loop: microduck.rope_skip ===", flush=True)
    champion_cfg = SkipConfig(trigger_lead_s=0.40)   # v1.0: untrained guess — jumps too early
    champion_ver = "1.0"
    champion_h = evaluate(champion_cfg, HOLDOUT_SEEDS, "baseline v1.0 HOLDOUT", "1.0")
    print(f"champion v1.0 holdout: {champion_h.success_rate:.0%}", flush=True)

    for rnd in range(1, rounds + 1):
        print(f"\n===== ROUND {rnd} (champion v{champion_ver}) =====", flush=True)
        print("-- candidates on train seeds --", flush=True)
        cands = adapt.generate_candidates(champion_cfg, n_per_axis=5)
        results = []
        for i, cand in enumerate(cands):
            cfg = replace(champion_cfg, **cand.params)
            r = evaluate(cfg, TRAIN_SEEDS, f"cand {i+1}/{len(cands)} {cand.tag}",
                         f"1.{rnd}-cand")
            results.append((r, cand))
        best_r, best_c = max(results, key=lambda x: x[0].success_rate)
        print(f"best: {best_c.tag} train={best_r.success_rate:.0%}", flush=True)
        print("-- Darwin holdout --", flush=True)
        cand_h = evaluate(replace(champion_cfg, **best_c.params), HOLDOUT_SEEDS,
                          f"candidate {best_c.tag} HOLDOUT", f"1.{rnd}")
        verdict = adapt.promotion_gate(champion_h, cand_h, min_delta=0.05)
        print(f"GATE: {verdict.reason} ({verdict.baseline_rate:.0%} → "
              f"{verdict.candidate_rate:.0%})", flush=True)
        if verdict.promoted:
            champion_cfg = replace(champion_cfg, **best_c.params)
            champion_ver = f"1.{rnd}"
            champion_h = cand_h
            print(f"*** CHAMPION v{champion_ver}: {best_c.params} ***", flush=True)
        else:
            print("no promotion; search converges", flush=True)
            break

    champion = {"skill_id": "microduck.rope_skip", "champion": champion_ver,
                "params": {k: getattr(champion_cfg, k) for k in adapt.TUNABLE},
                "holdout_success": champion_h.success_rate,
                "consecutive_best": champion_h.consecutive_best}
    out = ROOT / "rosclaw-microduck/practice/champion.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(champion, indent=2))
    print("FINAL CHAMPION:", json.dumps(champion, indent=2), flush=True)


if __name__ == "__main__":
    main()
