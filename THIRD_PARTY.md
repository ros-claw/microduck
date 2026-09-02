# Third-party assets & references

| Project | Role | License | Used as |
|---|---|---|---|
| [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) | official RL training repo | code Apache-2.0 / 3D models CC BY-SA-NC | MJCF robot (`robot_allcollisions.xml` + meshes), pulled at the pinned commit in `upstream.lock.yaml` |
| [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) | official onboard runtime | Apache-2.0 | pretrained ONNX policies (`alpha_stand`, `alpha_walking`, `alpha_sitstand`, `roulade`, …) |
| [rokbenko/quackd](https://github.com/rokbenko/quackd) | community LLM brain for Microduck | Apache-2.0 | studied as reference / future A/B benchmark; not a dependency |
| [MuJoCo](https://mujoco.org) | physics engine | Apache-2.0 | simulation |
| Marope (arXiv, 2026) | cooperative long-rope skipping MARL | paper | prior art for rope-skipping metrics |

Nothing from upstream is vendored into this repository; run
`scripts/bootstrap.py` to fetch pinned assets locally.
