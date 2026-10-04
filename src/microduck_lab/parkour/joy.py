"""Continuous commanded joy gait; a single learned policy, no burst switching."""

from dataclasses import dataclass, field
import math
import numpy as np
from .skills import lane_command
from .world import foot_support


def joy_pose_commands(elapsed):
    p = 2 * math.pi * 1.15 * elapsed
    e = min(1.0, max(0.0, elapsed / 0.8))
    head = e * np.array(
        [
            0.07 * math.sin(2 * p),
            0.03 * math.sin(2 * p),
            0.24 * math.sin(p),
            0.12 * math.sin(p),
        ]
    )
    body = e * np.array(
        [0.0, 0.0, 0.006 * (1 - math.cos(2 * p)), 0.07 * math.sin(p), 0.0, 0.0]
    )
    return head, body


@dataclass
class JoyDance:
    started: float
    finish_x: float = 7.95
    finished_at: float | None = None
    events: list = field(default_factory=list)
    phase: str = "APPROACH"
    mode: str = "continuous_joy_v1"
    anchor: np.ndarray | None = None
    dance_started: float | None = None
    min_upright: float = 1.0
    max_drift: float = 0.0
    feet_peak: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rates: list = field(default_factory=list)
    head_rates: list = field(default_factory=list)
    motion_samples: list = field(default_factory=list)
    last_targets: np.ndarray | None = None
    max_target_step: float = 0.0
    target_conditioner: dict | None = None
    metadata_checked: bool = False

    def update(self, m, d, r):
        t = float(d.time)
        r.head_override = None
        r.leg_override = None
        r.bank.mirror_run = False
        r.set_command()
        r.joint_vel_delay = 0
        if self.finished_at is None:
            r.active_policy = "run"
            r.command[:3] = lane_command(r, 0.0, 0.5)
            if r.trunk_pos()[0] >= self.finish_x:
                self.finished_at = t
                self.events.append(dict(type="FINISH", t=t))
            return
        q = t - self.finished_at
        if q < 0.8:
            r.active_policy = "run"
            self.phase = "BRAKE"
            return
        if q < 1.6:
            r.active_policy = "stand"
            self.phase = "SETTLE"
            return
        if self.anchor is None:
            self.anchor = np.array([*r.trunk_pos()[:2], r.trunk_yaw()])
            self.dance_started = t
            self.last_targets = d.ctrl[r.act_ids].copy()
            r._jv_prev = d.qvel[r.joint_qvel_idx].astype(np.float32).copy()
            self.events.append(dict(type="JOY_START", t=t))
        self.phase = "CONTINUOUS JOY"
        r.active_policy = "joy"
        r.joint_vel_delay = 1
        a = t - self.dance_started
        head, body = joy_pose_commands(a)
        delta = self.anchor[:2] - r.trunk_pos()[:2]
        dx, dy = np.clip(2 * delta, -0.15, 0.15)
        yaw = r.trunk_yaw()
        error = math.atan2(
            math.sin(self.anchor[2] - yaw), math.cos(self.anchor[2] - yaw)
        )
        twist = (
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
            float(np.clip(2 * error, -0.8, 0.8)),
        )
        r.set_command(twist=twist, head=head, body=body)

    def condition_targets(self, d, r):
        if self.anchor is None:
            return
        if not self.metadata_checked:
            meta = r.bank.get("joy").get_modelmeta().custom_metadata_map
            if meta.get("joy_target_conditioner") == "v1":
                alpha = float(meta["joy_target_alpha"])
                step = float(meta["joy_target_max_step_rad"])
                if alpha != 0.25 or step != 0.08:
                    raise ValueError("Unknown training/deployment target conditioner")
                self.target_conditioner = dict(
                    version="v1",
                    alpha=alpha,
                    max_step_rad=step,
                    hz=50,
                    initialization="previous actuator target",
                )
                self.mode = "continuous_joy_smooth_pd_v1"
            self.metadata_checked = True
        if self.target_conditioner:
            delta = np.clip(
                self.target_conditioner["alpha"]
                * (d.ctrl[r.act_ids] - self.last_targets),
                -self.target_conditioner["max_step_rad"],
                self.target_conditioner["max_step_rad"],
            )
            d.ctrl[r.act_ids] = self.last_targets + delta

    def sample(self, m, d, r):
        if self.finished_at is None:
            return
        self.min_upright = min(self.min_upright, float(d.xmat[r.trunk_body_id, 8]))
        if self.anchor is None:
            return
        self.max_drift = max(
            self.max_drift, float(np.linalg.norm(r.trunk_pos()[:2] - self.anchor[:2]))
        )
        for i, side in enumerate(["left", "right"]):
            self.feet_peak[i] = max(
                self.feet_peak[i], float(r.site_pos(side + "_foot")[2])
            )

    def control_sample(self, d, r):
        if self.anchor is None:
            return
        self.motion_samples.append(d.qpos[r.joint_qpos_idx].copy())
        targets = d.ctrl[r.act_ids].copy()
        if self.last_targets is not None:
            step = targets - self.last_targets
            self.max_target_step = max(
                self.max_target_step, float(np.max(np.abs(step)))
            )
            self.rates.append(float(np.mean((step / 0.02) ** 2)))
            self.head_rates.append(float(np.mean((step[5:9] / 0.02) ** 2)))
        self.last_targets = targets

    def report(self, m, d, r):
        duration = float(d.time) - (self.dance_started or float(d.time))
        rms = math.sqrt(np.mean(self.rates)) if self.rates else 1e3
        poses = np.asarray(self.motion_samples)
        if len(poses) > 20:
            centered = poses - poses.mean(0)
            spectrum = (
                np.abs(np.fft.rfft(centered * np.hanning(len(poses))[:, None], axis=0))
                ** 2
            )
            frequencies = np.fft.rfftfreq(len(poses), 0.02)
            motion_hf_ratio = float(
                spectrum[frequencies > 8].sum()
                / max(spectrum[frequencies > 0].sum(), 1e-12)
            )
        else:
            motion_hf_ratio = None
        passed = (
            self.finished_at is not None
            and duration >= 3.5
            and self.min_upright > 0.9
            and self.max_drift < 0.08
            and (self.feet_peak > 0.008).all()
            and d.xmat[r.trunk_body_id, 8] > 0.95
            and foot_support(m, d)
            and rms < 6
            and self.max_target_step < 0.2
            and motion_hf_ratio is not None
            and motion_hf_ratio < 0.01
        )
        return dict(
            mode=self.mode,
            passed=bool(passed),
            phase=self.phase,
            finished_at=self.finished_at,
            dance_started=self.dance_started,
            duration_s=duration,
            min_upright=self.min_upright,
            max_drift_m=self.max_drift,
            feet_peak_m=self.feet_peak.tolist(),
            target_rate_rms_rad_s=float(rms),
            head_target_rate_rms_rad_s=float(math.sqrt(np.mean(self.head_rates)))
            if self.head_rates
            else None,
            max_target_step_rad=self.max_target_step,
            pose_power_above_8hz_ratio=motion_hf_ratio,
            target_conditioner=self.target_conditioner,
            method="Single fine-tuned PPO policy at 50 Hz; continuous 1.15 Hz head/body pose commands and measured position/heading feedback. No servo overrides, action filtering, state edits or robot external force."
            if not self.target_conditioner
            else "Single PPO policy at 50 Hz; continuous pose commands and measured feedback; the same alpha=.25 / max-step=.08 rad target conditioner as training, initialized from the previous actuator target. Previous-action observations remain raw. No servo overrides, state edits or external robot forces.",
        )
