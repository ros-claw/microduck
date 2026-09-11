import math
from types import SimpleNamespace
import numpy as np
import pytest
from microduck_lab.sim.runtime import PolicyBank, DuckRuntime, sweep_height_command


def test_height_command_is_periodic_body_delta():
    assert sweep_height_command(0) == .06
    assert sweep_height_command(math.pi) == 0
    assert sweep_height_command(-.4) == sweep_height_command(.4)
    assert sweep_height_command(2*math.pi) == .06


def test_missing_feedback_fails_before_policy_inference():
    duck = object.__new__(DuckRuntime)
    duck.bank = SimpleNamespace(uses_hop_height_command=lambda _: True)
    duck.active_policy = 'hop'
    duck.hop_phase_source = None
    with pytest.raises(ValueError,match='phase source'):
        duck.step()


def test_metadata_unknown_version_rejected():
    bank = PolicyBank({})
    bank._sessions['bad'] = SimpleNamespace(get_modelmeta=lambda: SimpleNamespace(
        custom_metadata_map={'hop_height_command':'unknown'}))
    with pytest.raises(ValueError,match='unsupported'):
        bank.uses_hop_height_command('bad')


def test_runtime_writes_body_z_and_clears_it_on_policy_switch():
    duck = object.__new__(DuckRuntime)
    observations = []
    duck.bank = SimpleNamespace(
        uses_hop_height_command=lambda name: name == 'hop',
        uses_hop_centering=lambda _: False,
        infer=lambda name, obs: observations.append(obs.copy()) or np.zeros(14))
    duck.active_policy = 'hop'
    duck.hop_phase_source = lambda: 0.
    duck.command = np.zeros(13,dtype=np.float32)
    duck.get_obs = lambda: np.concatenate([np.zeros(48),duck.command])
    duck.default_pose = np.zeros(14)
    duck.action_scale = 1.
    duck.head_override = duck.leg_override = duck.bam_drive = None
    duck.data = SimpleNamespace(ctrl=np.zeros(14))
    duck.act_ids = np.arange(14)
    duck.step()
    expected = np.zeros(61)
    expected[57] = .06  # 48 proprio + 3 twist + 4 head + xyz[2]
    np.testing.assert_allclose(observations[-1],expected,atol=1e-8)
    duck.active_policy = 'stand'
    duck.step()
    assert not observations[-1].any()
