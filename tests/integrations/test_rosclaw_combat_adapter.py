import asyncio
import pytest
from microduck_lab.integrations import rosclaw_combat_adapter as combat


def test_combat_read_only_status_never_creates_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    before = list(tmp_path.rglob("*"))
    assert combat.get_status()["usable_for_real_execution"] is False
    assert before == list(tmp_path.rglob("*"))
    assert combat.get_status()["last_executor_report"] is None


def test_combat_catalog_strictly_separates_observation_and_action():
    catalog = asyncio.run(combat.server.list_tools())
    tools = {t.name: t for t in catalog}
    assert tools["microduck.combat.get_status"].annotations.readOnlyHint
    action = tools["microduck.combat.start_match"]
    assert not action.annotations.readOnlyHint
    assert action.inputSchema["additionalProperties"] is False
    assert action.inputSchema["properties"]["seed"]["maximum"] == 2**31 - 1
    assert not any("promote" in name or "grant" in name for name in tools)


def test_combat_unregistered_or_unknown_request_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    with pytest.raises(ValueError):
        asyncio.run(combat.start_match(1, "unknown"))
    with pytest.raises(ValueError, match="Register"):
        asyncio.run(combat.start_match(1, "duel"))
    with pytest.raises(Exception):
        asyncio.run(
            combat.server.call_tool(
                "microduck.combat.start_match",
                {"seed": 1, "preset": "duel", "force": True},
            )
        )
    assert not list(tmp_path.rglob("*"))
