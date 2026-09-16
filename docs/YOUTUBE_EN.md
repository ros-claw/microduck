# English YouTube upload package

## Title

GPT-6 Astra Helps Build Rope-Skipping Robots | ROSClaw Physical AI

## Description

Three robots. One flexible rope. Physical contact, learned motion, and closed-loop coordination.

ROSClaw's MicroDuck project brings two rope-turning robots and one jumping robot into the same MuJoCo simulation. The rope is attached to mouth-held handles and driven by the robots' motion. A clean skip requires the rope to clear both feet and the jumper to land with stable support.

Building on an existing Microduck prototype, GPT-6 Astra with medium reasoning assisted the development and refinement of the physics, training, deployment, evaluation, and presentation. This is a concrete example of AI-assisted robotics engineering: turning an incomplete prototype into a measured, reproducible physical-simulation result.

The runtime uses reinforcement-learned ONNX motor policies at 50 Hz, together with explicit rope-phase and timing feedback. The language model does not directly command the servos during the rollout.

RESULTS
442 clean cycles out of 516: 85.7% across eight seeded 30-second runs, excluding each run's first nine seconds. Five runs pass individually; three remain below 80%. This is a simulation result, not a hardware demonstration or a guarantee across arbitrary initial conditions.

The video shows edited views of one audited trajectory. The mouth and foot close-ups are labeled 0.25x slow motion. Audio is original post-produced foley, not recorded simulation sound.

CODE, METHODS, POLICIES & RESULTS
https://github.com/ros-claw/microduck

CREDITS
ROSClaw / MicroDuck project
AI-assisted development: GPT-6 Astra (medium reasoning)
Robot model and original materials: Pollen Robotics Microduck
Physics: MuJoCo
Training: mjlab / PPO

#ROSClaw #PhysicalAI #Robotics #ReinforcementLearning #MuJoCo #GPT6

## Files

- Video: [microduck_youtube_en.mp4](../out/microduck_youtube_en.mp4), 1920 x 1080, 50 fps, 32 seconds, English on-screen text, AAC foley audio.
- Thumbnail: [microduck_youtube_en_thumbnail.jpg](../out/microduck_youtube_en_thumbnail.jpg), 1920 x 1080.
- Captions: [microduck_youtube_en.srt](../out/microduck_youtube_en.srt), optional English subtitle track; the video already contains English text.
- Provenance: [youtube-en.json](../artifacts/presentation/youtube-en.json).

## Reproduce

After installing the project and configuring `MICRODUCK_ROOT` as described in the README:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/render_youtube_en.py
```

If no trajectory cache exists, the renderer captures and audits it first. No joint interpolation, action edits, or new physics settings are introduced by this edit. The local silent intermediate remains in the ignored presentation cache.
