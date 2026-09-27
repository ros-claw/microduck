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
