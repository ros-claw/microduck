"""The external executor rejects unreviewed parameters before any side effect."""

import asyncio
import hashlib
import pytest

pytest.importorskip("mcp")
from microduck_lab.integrations import rosclaw_adapter as adapter


def test_action_catalog_is_strict_and_unknown_arguments_are_rejected():
    import jsonschema

    async def check():
        tools = {t.name: t for t in await adapter.server.list_tools()}
        schema = tools["microduck.start_game"].inputSchema
        assert schema["additionalProperties"] is False
        for bad in [
            dict(seed=1, players=True),
            dict(seed=1, layout="unknown"),
            dict(seed=1, teleport=True),
            dict(seed=-1),
        ]:
            with pytest.raises(jsonschema.ValidationError):
                await adapter.server.call_tool("microduck.start_game", bad)
        assert not adapter._busy

    asyncio.run(check())


def test_status_is_read_only_and_binds_actual_profile_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    path = tmp_path / "bodies/microduck-lavender/refs/eurdf.profile.yaml"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"identity: microduck\n")
    before = set(tmp_path.rglob("*"))
    status = adapter.get_game_status()
    assert set(tmp_path.rglob("*")) == before
    assert (
        status["body_profiles"][0]["profile_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert not status["body_profiles"][1]["registered"]
    assert status["evidence_domain"] == "simulation"


def test_single_flight_and_cancellation_stop_the_actual_worker_handle(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    for name in ("lavender", "cream", "sky", "graphite"):
        p = tmp_path / f"bodies/microduck-{name}/refs/eurdf.profile.yaml"
        p.parent.mkdir(parents=True)
        p.write_text("identity: microduck\n")

    class Worker:
        returncode = None
        killed = False

        async def communicate(self, request):
            started.set()
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True

        async def wait(self):
            self.returncode = -9

    worker = Worker()

    async def spawn(*args, **kwargs):
        return worker

    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", spawn)

    async def check():
        global started
        started = asyncio.Event()
        job = asyncio.create_task(adapter.start_game(101))
        await started.wait()
        with pytest.raises(ValueError, match="already active"):
            await adapter.start_game(102)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert worker.killed and worker.returncode == -9 and not adapter._busy

    asyncio.run(check())


def test_worker_timeout_kills_and_waits_before_clearing_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
    for name in ("lavender", "cream"):
        path = tmp_path / f"bodies/microduck-{name}/refs/eurdf.profile.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("identity: microduck\n")

    class Worker:
        returncode = None
        killed = False

        async def communicate(self, request):
            return b"", b""

        def kill(self):
            self.killed = True

        async def wait(self):
            self.returncode = -9

    worker = Worker()

    async def spawn(*args, **kwargs):
        return worker

    async def timeout(awaitable, seconds):
        awaitable.close()
        assert seconds == 600
        raise asyncio.TimeoutError

    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(adapter.asyncio, "wait_for", timeout)

    async def check():
        with pytest.raises(asyncio.TimeoutError):
            await adapter.start_game(510, players=2)
        assert worker.killed and worker.returncode == -9 and not adapter._busy

    asyncio.run(check())
