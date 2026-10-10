# Victory finish / 终点庆祝

> Historical V5: physically simulated, but user review found the hop/stand switches
> twitchy. Superseded by the [continuous-motion and contact-detail revision](CONTACT_DETAILS.md).
> 旧版 V5 的动作经过物理计算，但观看反馈是抽搐；新版改用连续动作训练，详见上面的新方法文档。

The final English technical cut adds a happy finish approach, two short airborne
hops, head nods and a closer front-side camera. It uses native robot materials;
all movement remains in the contact-enabled MuJoCo simulation. Walking and
centered rope-hop policies are reused. Head gestures and skill scheduling are
explicitly scripted servo commands, not a newly trained emotional or general
reasoning capability. Current limits are unchanged. No robot/prop pose or
velocity is overwritten and no external robot force is added.

最后一版保留完整关卡，并增加冲线时的轻微摆头、刹稳后的两次小跳和点头庆祝。
行走与小跳复用已有学习策略，庆祝时序和头部动作是明确编排的电机指令。腾空和落地
由物理求解产生；没有改写鸭子根部位置、速度或给鸭子施加隐形外力。

## What this capture represents / 记录范围

`artifacts/neon-escape-v5/victory-2` replays historical live-Jev seed 402 up to
25.24 simulation seconds, immediately after its second roll. All 1,262 prefix
control ticks and 5,048 captured states match the parent trajectory exactly.
It then executes a newly simulated finish, not a pose splice. Gate machinery,
pursuit, guides, bowling contacts and damage remain active. Historical Jev
choices and forecast timings are retained; no new API choices were made for
this continuation. This selected demonstration is not an additional unseen
trial. Prior six-seed results remain 2/6 strict passes for the final default.

The duck crosses at 27.74 s with HP 2/3. Two foot-supported hop landings follow
73.2/71.2 ms flights, with peak trunk heights 14.18/13.93 cm. Minimum post-finish
upright projection is 0.908; final stance is supported and upright. Sustained
hopping and several shorter transitions failed during development; the selected
0.4 s hop-policy bursts are separated by standing. Diagnostic probes are
retained, not counted as independent reliability evidence.

精选记录的前半段是之前 Jev 实测轨迹的精确输入重放，后半段是从该动态状态继续做的
新仿真；没有将两段不同位置的鸭子拼接在一起。新的庆祝不是一次新 API 测试，也不能
因此增加未见种子的成功率。长时间连续跳会失稳，因此选用短跳与站稳交替。

Independent input replay, full-contact inspection at every 0.2 ms, and gate/guide
causality all pass. Every force-bearing contact category retains maximum
penetration <1.5 mm and P99 <1 mm; no solver warning is allowed. The largest
observed category depth is duck–floor 0.976 mm. The separate full-stunt gate
remains false, since this record does not demonstrate knockdown/recovery.
This is a privileged-state simulator example, not hardware or real-time vision.
Audio is synthesized Foley/chimes, not a simulated microphone.

## Reproduce / 复现

```bash
# Requires the earlier published seed-402 capture, extracted as capture/.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/continue_parkour_victory.py \
  --source capture --output artifacts/neon-escape-v5/local-new
.venv/bin/python scripts/replay_parkour_run.py artifacts/neon-escape-v5/local-new
.venv/bin/python scripts/audit_parkour_contacts.py artifacts/neon-escape-v5/local-new
.venv/bin/python scripts/audit_arcade_causality.py artifacts/neon-escape-v5/local-new
.venv/bin/python scripts/render_parkour.py \
  --source artifacts/neon-escape-v5/local-new --goal escape --style industrial \
  --cut technical --blur 1 --output out/local-victory.mp4
```

For a fresh arcade run, `parkour_rehearsal --victory-dance` enables the same finish
controller after the exit roll. Failure gates also require two qualified hops
and a supported final stance. Other behavior is unchanged unless enabled.

The final English cut keeps three slow-motion segments (gap, strike, celebration)
and the complete run. Rendering reads the saved states; camera motion and cuts
never change the physics. Binary release evidence includes the scene, inputs,
trajectory, five motor policies, source snapshot, audits and checksums; replay
needs no API key. Only the final English technical cut is produced this revision.
