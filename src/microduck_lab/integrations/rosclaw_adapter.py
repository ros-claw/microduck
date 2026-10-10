"""Microduck SIM MCP boundary. Game mutation is PHYSICAL_ACTION, not OBSERVE.

ROSClaw's SimActionChannel owns dispatch and receipt/grant/mission linkage.
This executor never issues grants, edits agent state, or accesses hardware.
"""

import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from mcp.server.fastmcp import FastMCP


class StrictGameMCP(FastMCP):
    async def list_tools(self):
        catalog = await super().list_tools()
        for tool in catalog:
            tool.inputSchema["additionalProperties"] = False
            if tool.name == "microduck.start_game":
                props = tool.inputSchema["properties"]
                props["seed"].update(minimum=0, maximum=2**31 - 1)
                props["players"]["enum"] = [2, 4]
                props["layout"]["enum"] = ["square", "ring", "cross", "terraces"]
                props["cadence"]["enum"] = ["steady", "rapid"]
        return catalog

    async def call_tool(self, name, arguments):
        import jsonschema

        tools = {tool.name: tool for tool in await self.list_tools()}
        if name not in tools:
            raise ValueError("Unknown game tool")
        jsonschema.validate(arguments or {}, tools[name].inputSchema)
        return await super().call_tool(name, arguments)


server = StrictGameMCP("microduck-sim", log_level="WARNING")
_receipt = None
_busy = False


def body_profiles():
    home = Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    rows = []
    for name in ("lavender", "cream", "sky", "graphite"):
        path = home / "bodies" / ("microduck-" + name) / "refs/eurdf.profile.yaml"
        rows.append(
            dict(
                body_id="microduck-" + name,
                registered=path.is_file(),
                profile_sha256=sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None,
            )
        )
    return rows


@server.tool(name="microduck.get_game_status")
def get_game_status() -> dict:
    """Read the last completed physical simulation result; never start a game."""
    return dict(
        ok=True,
        evidence_domain="simulation",
        busy=_busy,
        last_game=_receipt,
        body_profiles=body_profiles(),
    )


@server.tool(name="microduck.start_game")
async def start_game(
    seed: int, players: int = 4, layout: str = "square", cadence: str = "steady"
) -> dict:
    """SIMULATION ONLY: run one shared MuJoCo world, record controls/contacts.

    Four independent robots; gravity releases free-body tiles; public warnings
    guide a deterministic survivor over official 50Hz ONNX walking policies.
    This mutates simulated state and writes evidence. It does not train robots.
    """
    global _busy, _receipt
    if _busy:
        raise ValueError("A match is already active")
    if type(seed) is not int or not 0 <= seed < 2**31 or players not in (2, 4):
        raise ValueError("Invalid seed/player count")
    if layout not in ("square", "ring", "cross", "terraces") or cadence not in (
        "steady",
        "rapid",
    ):
        raise ValueError("Unknown layout/cadence")
    home = Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    profiles = body_profiles()[:players]
    if not all(p["registered"] for p in profiles):
        raise ValueError("Register the participating Microduck simulation bodies first")
    run_id = "duckverse_" + uuid4().hex[:16]
    out = home / "artifacts" / "duckverse" / run_id
    repo = Path(__file__).resolve().parents[3]
    local_python = repo / ".venv" / "bin" / "python"
    python = os.environ.get(
        "MICRODUCK_PYTHON",
        str(local_python) if local_python.exists() else sys.executable,
    )
    request = dict(
        seed=seed, players=players, layout=layout, cadence=cadence, out=str(out)
    )
    _busy = True
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            python,
            "-m",
            "microduck_lab.integrations.rosclaw_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(request).encode()), 600.0
        )
        if proc.returncode:
            raise RuntimeError("Simulation worker failed: " + stderr.decode()[-1200:])
        result = json.loads(stdout)
        audit = out / "audit.json"
        # Participant identities are references into ROSClaw's existing registry.
        _receipt = dict(
            ok=result["finite"] and not any(result["solver_warnings"]),
            quality_pass=result["penetration_max_m"] <= 0.003,
            evidence_domain="simulation",
            usable_for_real_execution=False,
            run_id=run_id,
            participant_body_ids=[
                "microduck-" + n
                for n in ("lavender", "cream", "sky", "graphite")[:players]
            ],
            participant_profiles=profiles,
            audit_path=str(audit),
            audit_sha256=sha256(audit.read_bytes()).hexdigest(),
            **result,
        )
        return _receipt
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        _busy = False


if __name__ == "__main__":
    server.run(transport="stdio")
