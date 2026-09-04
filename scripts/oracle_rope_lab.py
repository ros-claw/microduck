"""Oracle Rope Feasibility Lab — the go/no-go experiment from the 0904 doc.

No ducks. Two ideal endpoint carriers at mouth height, an elastic-cable rope.
The endpoints are driven by a PhaseTrackingTurner (swing-up: pump the measured
belly angle with a +90° lead; this is the drive that inflates the loop).

GO criterion (per the 0904 doc §36): sustained full rotation with
  top_z > 27 cm (clears the 25 cm duck) AND bottom_z < 2 cm (floor-grazing)
  for 8+ revolutions.

Usage:
    python scripts/oracle_rope_lab.py sustain --length 0.82 --density 400
    python scripts/oracle_rope_lab.py sweep
    python scripts/oracle_rope_lab.py render --length 0.82 --density 400
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from microduck_lab.sim.oracle_rope import (
    build_oracle_world, rope_bodies, RopeMorphology, PhaseTrackingTurner)


def run_oracle(length=0.82, density=400.0, bend=1e3, sep=0.50, height=0.18,
               radius=0.003, damping=1e-4, sustain_amp=0.045, dphi_deg=0.0,
               seconds=20.0, render_path=None):
    """Swing up + sustain a rope loop; measure morphology. Returns dict."""
    m, d = build_oracle_world(length=length, sep=sep, density=density, bend=bend,
                              radius=radius, damping=damping, height=height)
    ids = rope_bodies(m)
    turner = PhaseTrackingTurner(m, d, ids, axis_z=height)
    cA = np.array([-sep / 2, 0.0, height])
    cB = np.array([sep / 2, 0.0, height])
    for _ in range(2000):        # let the rope hang and settle
        mujoco.mj_step(m, d)

    morph = RopeMorphology()
    hist = []
    N = int(seconds * 1000)
    renderer = None
    frames = []
    if render_path:
        renderer = mujoco.Renderer(m, height=460, width=460)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = [0, 0, 0.16]; cam.distance = 0.8; cam.azimuth = 0; cam.elevation = 0
    for i in range(N):
        cont = turner.step(i, 0.001, cA, cB, sustain_amp=sustain_amp,
                           dphi=math.radians(dphi_deg))
        mujoco.mj_step(m, d)
        if i % 50 == 0:
            morph.update(d, ids, dt=0.05, jumper_x=0.0, axis_z=height)
            hist.append((morph.revolutions, morph.omega, morph.belly_radius,
                         morph.top_z, morph.bottom_z, morph.plane_error))
        if render_path and i >= int(0.8 * N) and i % 60 == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

    hist = np.array(hist)
    # tail window (last 40%): the steady state
    tail = hist[int(len(hist) * 0.6):]
    revs_tail = abs(tail[-1, 0] - tail[0, 0])
    hz = abs(tail[-1, 0] - tail[0, 0]) / (len(tail) * 0.05) if len(tail) else 0
    top_max = float(np.max(tail[:, 3]))
    bot_min = float(np.min(tail[:, 4]))
    plane = float(np.mean(tail[:, 5]))
    feasible = bool(revs_tail >= 8 and top_max > 0.27 and bot_min < 0.02)

    if render_path and frames:
        import imageio
        cols = 5
        rows = (len(frames) + cols - 1) // cols
        h, w = frames[0].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, f in enumerate(frames):
            grid[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = f
        imageio.imwrite(str(render_path), grid)

    return dict(revolutions=revs_tail, rot_hz=hz, top_z=top_max, bottom_z=bot_min,
                plane_error=plane, feasible=feasible)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sustain", "sweep", "render"])
    ap.add_argument("--length", type=float, default=0.82)
    ap.add_argument("--density", type=float, default=400.0)
    ap.add_argument("--bend", type=float, default=1e3)
    ap.add_argument("--sep", type=float, default=0.50)
    ap.add_argument("--height", type=float, default=0.18)
    ap.add_argument("--radius", type=float, default=0.003)
    ap.add_argument("--damping", type=float, default=1e-4)
    ap.add_argument("--sustain-amp", type=float, default=0.045)
    ap.add_argument("--dphi", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", type=str, default="/tmp/oracle_render.png")
    args = ap.parse_args()

    if args.mode in ("sustain", "render"):
        r = run_oracle(length=args.length, density=args.density, bend=args.bend,
                       sep=args.sep, height=args.height, radius=args.radius,
                       damping=args.damping, sustain_amp=args.sustain_amp,
                       dphi_deg=args.dphi, seconds=args.seconds,
                       render_path=args.out if args.mode == "render" else None)
        print(f"\n=== ORACLE (L={args.length} density={args.density} bend={args.bend:.0e} "
              f"sep={args.sep} h={args.height} amp={args.sustain_amp} dphi={args.dphi}) ===")
        print(f"  sustained revolutions: {r['revolutions']:.1f}  at {r['rot_hz']:.2f} Hz")
        print(f"  top_z max:    {r['top_z']*100:.1f} cm   (need > 27)")
        print(f"  bottom_z min: {r['bottom_z']*100:.1f} cm   (need < 2)")
        print(f"  plane_error:  {r['plane_error']*1000:.1f} mm")
        print(f"  FEASIBLE: {'YES ✓' if r['feasible'] else 'no ✗'}")
        if args.mode == "render":
            print(f"  render → {args.out}")
    else:
        print(f"{'L':>5} {'dens':>5} {'revs':>7} {'Hz':>6} {'top':>7} {'bot':>7}  FEASIBLE")
        for L in (0.68, 0.74, 0.78, 0.82, 0.86):
            for dens in (100, 200, 400):
                r = run_oracle(length=L, density=dens, seconds=args.seconds)
                print(f"{L:5.2f} {dens:5.0f} {r['revolutions']:+7.1f} {r['rot_hz']:6.2f} "
                      f"{r['top_z']*100:6.1f}cm {r['bottom_z']*100:6.1f}cm  "
                      f"{'YES ✓' if r['feasible'] else 'no'}", flush=True)


if __name__ == "__main__":
    main()
