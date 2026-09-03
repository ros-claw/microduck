"""Procedural-side grid evolution: frequency × lead × rope length."""
import sys, pathlib, itertools, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dataclasses import replace
from microduck_lab.skills.skip import RopeSkipSession, SkipConfig

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"
BANK = {"stand": str(POL / "alpha_stand.onnx"), "walk": str(POL / "alpha_walking.onnx"),
        "sitstand": str(POL / "alpha_sitstand.onnx")}

SEEDS = [101, 202, 303, 404]
results = []
for freq, lead, L in itertools.product([0.85, 1.0, 1.15], [0.55, 0.62, 0.70], [0.55, 0.58]):
    ts = tc = tt = 0
    worst_up = 1.0
    for seed in SEEDS:
        s = RopeSkipSession(str(ROBOT), BANK,
                            SkipConfig(seed=seed, frequency=freq, trigger_lead_s=lead, rope_length=L))
        s.settle(); s.start_rope()
        m = s.run_episode(14.0)
        ts += m.successful_skips; tc += m.crossings; tt += m.trips
        worst_up = min(worst_up, m.jumper_upright_frac)
    rate = ts / max(1, tc)
    print(f"f={freq} lead={lead} L={L}: skips={ts}/{tc} ({rate:.0%}) trips={tt} worst_up={worst_up:.0%}", flush=True)
    results.append({"frequency": freq, "trigger_lead_s": lead, "rope_length": L,
                    "skips": ts, "crossings": tc, "trips": tt, "rate": rate,
                    "worst_upright": worst_up})

results.sort(key=lambda r: -r["rate"])
out = ROOT / "rosclaw-microduck/practice/grid_evolution.json"
out.write_text(json.dumps(results, indent=2))
print("\nTOP 3:")
for r in results[:3]:
    print(f"  f={r['frequency']} lead={r['trigger_lead_s']} L={r['rope_length']}: {r['rate']:.0%} ({r['skips']}/{r['crossings']})")
