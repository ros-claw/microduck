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
