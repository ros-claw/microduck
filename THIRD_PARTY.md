# Third-party assets & references

| Project | Role | License | Used as |
|---|---|---|---|
| [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) | official RL training repo | code Apache-2.0 / 3D models CC BY-SA-NC | MJCF robot (`robot_allcollisions.xml` + meshes), pulled at the pinned commit in `upstream.lock.yaml` |
| [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) | official onboard runtime | Apache-2.0 | pretrained ONNX policies (`alpha_stand`, `alpha_walking`, `alpha_sitstand`, `roulade`, …) |
| [rokbenko/quackd](https://github.com/rokbenko/quackd) | community LLM brain for Microduck | Apache-2.0 | studied as reference / future A/B benchmark; not a dependency |
| [MuJoCo](https://mujoco.org) | physics engine | Apache-2.0 | simulation |
| Marope (arXiv, 2026) | cooperative long-rope skipping MARL | paper | prior art for rope-skipping metrics |

Robot source meshes are fetched locally through `scripts/bootstrap.py` rather
than stored as source assets here. Release MJB captures contain compiled robot
geometry, and evidence bundles include the upstream motor policies used by the
record. Their original attribution and asset/model license terms remain relevant.

`policies/continuous_joy.onnx` is a PPO fine-tune initialized from upstream
`alpha_walking.onnx`, trained with the project's continuous-pose task and matching
position-servo target conditioner. The source patch, seed/selected checkpoints,
source snapshot and SHA-256 provenance are in the V6 evidence bundle; see
[CONTACT_DETAILS](docs/CONTACT_DETAILS.md) and
[training/run.json](artifacts/neon-escape-v6/training/run.json). It is a simulation
policy, not an upstream release or a hardware-qualified behavior.
