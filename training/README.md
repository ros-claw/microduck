# 训练源码交付

`microduck_rl.patch` 包含本项目相对官方基线的全部已提交训练改动：跳跃、原地跳跃、带绳转头、真实旋转障碍训练、导出元数据和回归测试。它从训练仓库的 Git 提交生成，不含本地未提交的 `uv.lock` 修改，也不包含模型资产或大体积训练日志。

- 官方基线：`pollen-robotics/microduck_rl` @ `5946fd9cdbc58956424420153e51975af3b30d77`
- 已交付源码：`18052aaf7d98942921a8d57319f40d3692f96bcd`
- 发布策略：`../policies/ropehop_contact.onnx`
- 策略 SHA-256：`4d2e1f85bc3243dfe6d05d9221a77b855a9dbabf734d17214181dc9d032abdd4`

在一个新训练目录恢复，保持官方仓库自己的 AGENTS.md 约定：

```bash
# 从交付仓库根目录运行
DELIVERY_ROOT="$PWD"
git clone https://github.com/pollen-robotics/microduck_rl.git ../microduck-training
cd ../microduck-training
git checkout -b microduck-contact 5946fd9cdbc58956424420153e51975af3b30d77
git apply --check "$DELIVERY_ROOT/training/microduck_rl.patch"
git apply "$DELIVERY_ROOT/training/microduck_rl.patch"
uv sync
uv run --with pytest pytest tests/test_clean_ropehop_cfg.py tests/test_sweep_ropehop_cfg.py
```

训练任务 `Mjlab-SweepRopeHop-Flat-MicroDuck` 的命令、奖励、环境偏移修复和消融结果见 [SWEEP_TRAINING](../docs/SWEEP_TRAINING.md)。长训练前先运行 64 环境、5 iteration 的 smoke。

此包提供完整训练源码与部署 ONNX，**不包含用于继续训练的原始 `.pt` 检查点或全部日志**。文档中的 resume 命令需要相应检查点；没有检查点时可从头训练，但不保证得到相同权重或成绩。现成 ONNX 的评估与视频复现不依赖这些训练检查点。

## Parkour V2 / 跑酷训练

`parkour_v2.patch` 是叠加在上述 `18052aa` 交付源码上的增量，不含本地 `uv.lock` 修改。
增加真实地形缺口任务、站立 ONNX 初始化、出生姿态验证和配置测试。

```bash
git apply --check "$DELIVERY_ROOT/training/parkour_v2.patch"
git apply "$DELIVERY_ROOT/training/parkour_v2.patch"
uv run --with pytest pytest tests/test_parkour_cfg.py
uv run python scripts/verify_parkour_spawn.py
```

The incremental patch adds `Mjlab-ParkourGap-Flat-MicroDuck`. The actor retains
the shared 61-dimensional observation, 50 Hz control and normalized ONNX export.
The selected deployment model and evidence are described in
[NEON_ESCAPE_V2](../docs/NEON_ESCAPE_V2.md). Training checkpoints are separate from
the deployment ONNX; a specific weight reproduction requires its original checkpoint.

`Mjlab-ParkourRoll-Flat-MicroDuck` is an additional hard-contact transfer
experiment based on the official roulade task. It uses XML PD servos for this
simulation runtime, 0.6405 Nm limits, 1 ms training physics / 50 Hz actions,
native robot geometry and 2 ms ground/self contact references. Warp's current
mesh multi-contact implementation rejects nonzero contact margin; this training
profile therefore uses **zero margin**, which must also be used in its deployment
comparison. It is not certified by the earlier positive-margin batteries.

硬接触翻滚训练是独立实验，不能拿旧策略测试结果替代新配置验证。训练前已完成
4 个扰动初态的 3 秒站立检查、64 环境 / 5 轮短测、标准 ONNX 导出，以及官方
翻滚 ONNX 初始化的数值一致性检查。训练日志位于 V2 evidence 的 `training/`。

The hard-contact roll training was stopped after deployment tests of checkpoints
100 and 200 did not qualify. Its zero-margin profile does not match the selected
positive-margin course; these weights are not shipped as successful skills.

硬接触翻滚微调已停止：100／200 轮检查点未通过部署验证，没有替代官方策略。
失败结果与日志保留，训练奖励不作为技能成功证明。

The V2 physical-baseline experimental release separately includes the selected
long-jump `model_500.pt` checkpoint in its evidence bundle. The older rope policies'
training checkpoints are still not included.

V2 物理基线实验预发布的证据包单独附带所选长跳 `model_500.pt`；这不改变旧跳绳
训练检查点未随仓库交付的限制。

## Continuous gait / 连续步态

`continuous_joy.patch` applies after `microduck_rl.patch` and `parkour_v2.patch`.
It preserves the 61-D actor, 50 Hz controls, observation normalization and base
randomization/noise/NaN guards. `Mjlab-JoySmoothPD-Flat-MicroDuck` uses the
existing supported XML position-servo family with ±0.6405 N m limits; these
simulation weights must not be described as BAM/hardware-qualified policies.
Zero-margin, 1 ms Warp training is checked in the existing 0.2 ms CPU course.
The matched target conditioner uses alpha .25 and max step .08 rad per 20 ms;
raw previous actions remain in observations. Its metadata must accompany ONNX.

连续步态补丁叠加于已交付训练源码与跑酷补丁之上。训练和运行时采用相同目标递推，
不能只在部署端临时加滤波；ONNX 元数据记录这个接口。位置伺服分支、电流限制、
训练/部署碰撞差异与失败候选详见 [CONTACT_DETAILS](../docs/CONTACT_DETAILS.md)。

```bash
git apply --check "$DELIVERY_ROOT/training/continuous_joy.patch"
git apply "$DELIVERY_ROOT/training/continuous_joy.patch"
uv run --with pytest pytest tests/test_joy_cfg.py
uv run train Mjlab-JoySmoothPD-Flat-MicroDuck --env.scene.num-envs 64 \
  --agent.max-iterations 5 --agent.logger tensorboard
# Long run, after successful smoke. To reproduce the selected fine-tune,
# use the archived seed checkpoint and the flags in the evidence run manifest.
uv run train Mjlab-JoySmoothPD-Flat-MicroDuck --env.scene.num-envs 2048 \
  --agent.max-iterations 200 --agent.logger tensorboard
uv run scripts/export.py Mjlab-JoySmoothPD-Flat-MicroDuck \
  --checkpoint-file /path/to/checkpoint.pt --device cpu --num-envs 1 \
  --onnx-file continuous_joy.onnx
```

The selected policy is initialized from the separately trained PD candidate,
which was itself seeded from upstream `alpha_walking.onnx` with numerical parity
verification. Training from scratch follows the same task but does not reproduce
these weights. The release includes the selected actor checkpoint, its seed,
exact selected-run source snapshot, logs, standard-export parity and deployment
probes. The incremental patch omits pre-existing uncommitted `uv.lock` changes;
the full run snapshot retains the actual lockfile. Failed unfiltered candidates
and the initially unstable filtered handoff are reported rather than relabeled
as successes. Four small deployment perturbations are tuning checks, not
unseen-game trials or hardware tests.
