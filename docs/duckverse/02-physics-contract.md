# Physics contract / 物理契约

- Arena top z=0; nine square tiles, 0.48 m sides, 0.012 m seams, 0.05 m thick, 0.8 kg each. Grid pitch 0.492 m. Dimensions remain a one-duck calibration, not four-duck sizing proof.
- Each tile is a freejoint body with box geometry and world weld. Initial relative pose is computed by MuJoCo. Only `data.eq_active` changes at release. No floor moves/vanishes by animation.
- The only fixed support is a visible receiver, top z=-0.81 m. No plane exists at arena height. Robot root and joint initialization are explicitly marked; subsequent dynamics receive motor ctrl and equality events only. No external applied wrench or root pose editing.
- Native robot collision meshes are retained. They are MuJoCo convex collision representations, not promises of exact visual triangle contact. No new collision exclusion or mass-bearing costume.
- Standing/adjacent walking ONNX at 50 Hz; no online training, Jev or motor overrides. Warning is a visual-only presentation colour; at release the equality actually changes.
- Contact stream samples every physics step: geom IDs, signed distance, 6D force/torque, point, normal and supporting tile IDs. Forces and full integration states refer to pre-integration time; root/tile diagnostics refer to t+dt. A positive-force foot contact to an active tile with near-vertical normal is counted as support. This is experimental support evidence, not a completed referee.
- Complete integration states include warmstart, ctrl, equality activity and external force channels via `mjSTATE_INTEGRATION`. Strict replay initializes once then applies ctrl/eq_active and advances physics, comparing qpos, qvel, contact identities/distances/wrenches.
- Per-run hash checks bind model, trajectory and contact stream. Rendering first requires successful replay, then loads recorded states only; camera/HUD/colours do not affect the recorded dynamics.
- Suggested acceptance: no solver warnings/NaN/time reset, locked drift <1 mm, supported neighbour transition, actual fall below z=-0.4 m; P99 penetration ≤1.5 mm and absolute max ≤3 mm. Failed configurations remain documented. No claim of zero penetration.

MuJoCo documents runtime equality activity in [mjData](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html). Exact tested settings and results are in [DG-01](LAST_DUCK_STANDING.md).
