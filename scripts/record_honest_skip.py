"""Record the HONEST classic rope-skip: the rope's ends are connected to the
two turner ducks' beak-held handles and the rope is swung BY THE DUCKS — the
rope-turner policy (CR-05, trained with the real chain rope in the Warp loop)
circles the mouth-held handle at 3.1 Hz while balancing; the jumper runs the
rope-hop policy. No mocap carriers anywhere.

Usage:
  MUJOCO_GL=osmesa .venv/bin/python scripts/record_honest_skip.py [turner.onnx]

Writes honest_skip_full.mp4 to the repo parent's parent (~/workspace/microduck).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import imageio
from PIL import Image, ImageDraw

from microduck_lab.demos.honest_skip import run_honest_classic_skip

ROOT = pathlib.Path(__file__).resolve().parents[1]
MD = ROOT.parent                                # ~/workspace/microduck
TURN = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "policies/turner_rope.onnx"
HOP = ROOT / "policies/ropehop_classic.onnx"
OUT = MD / "honest_skip_full.mp4"


def overlay(img, t, m):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 330, 44], fill=(0, 0, 0))
    dr.text((8, 6), f"t={t:4.1f}s  passes={m['passes']}", fill=(255, 255, 255))
    dr.text((8, 24), f"SKIPS={m['skips']}  trips={m['trips']}", fill=(120, 255, 120))
    return np.array(im)


def main():
    metrics, frames = run_honest_classic_skip(
        None, seconds=50.0, turner_onnx=str(TURN), hop_onnx=str(HOP),
        overlay_fn=overlay)
    print("METRICS", metrics)
    imageio.mimsave(str(OUT), frames, fps=25)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
