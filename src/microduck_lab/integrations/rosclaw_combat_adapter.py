"""Duck Fu SIM executor boundary, separate from the frozen survival adapter.

The ROSClaw admission channel owns authorization/grants/official receipts.
This server produces an executor report, never fabricates a grant or promotes
a candidate. Merely listing tools or reading status cannot launch physics.
"""

import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import jsonschema
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .rosclaw_adapter import body_profiles

ROOT = Path(__file__).resolve().parents[3]
PRESETS = {
    "duel": "configs/combat/match_motors_v1.yaml",
    "team": "configs/combat/team_final.yaml",
}
_busy = False
_latest = None


class CombatMCP(FastMCP):
    async def list_tools(self):
        catalog = await super().list_tools()
        for tool in catalog:
            tool.inputSchema["additionalProperties"] = False
            if tool.name == "microduck.combat.start_match":
                props = tool.inputSchema["properties"]
                props["seed"].update(minimum=0, maximum=2**31 - 1)
                props["preset"]["enum"] = sorted(PRESETS)
        return catalog

    async def call_tool(self, name, arguments):
        catalog = {t.name: t for t in await self.list_tools()}
        if name not in catalog:
            raise ValueError("Unknown combat capability")
        jsonschema.validate(arguments or {}, catalog[name].inputSchema)
        return await super().call_tool(name, arguments)


server = CombatMCP("microduck-combat-sim", log_level="WARNING")


@server.tool(
    name="microduck.combat.get_status",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
def get_status() -> dict:
    """Read-only SIM status. No physics, training, grants or files are created."""
    return dict(
        ok=True,
        evidence_domain="simulation",
        busy=_busy,
        last_executor_report=_latest,
        participant_profiles=body_profiles(),
        presets={k: (ROOT / v).is_file() for k, v in PRESETS.items()},
        authorization_owner="ROSClaw admission channel, outside this executor",
        usable_for_real_execution=False,
    )


@server.tool(
    name="microduck.combat.start_match",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def start_match(seed: int, preset: str = "duel") -> dict:
    """SIMULATION PHYSICAL_ACTION: run a fixed, versioned Duck Fu match.

    Dispatch only after admission through ROSClaw's exact-action channel.
    This is not OBSERVE, real hardware, a training request or human acceptance.
    """
    global _busy, _latest
    if _busy:
        raise ValueError("A simulated match is already running")
    if type(seed) is not int or not 0 <= seed < 2**31 or preset not in PRESETS:
        raise ValueError("Invalid exact match request")
    config = ROOT / PRESETS[preset]
    if not config.is_file():
        raise ValueError("The frozen preset is not installed")
    import yaml

    settings = yaml.safe_load(config.read_text())
    profiles = body_profiles()[: settings["players"]]
    if not all(p["registered"] for p in profiles):
        raise ValueError("Register the simulation body profiles through ROSClaw first")
    run_id = "duckfu_" + uuid4().hex[:16]
    home = Path(os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw"))
    out = home / "artifacts/duckfu" / run_id
    python = os.environ.get("MICRODUCK_PYTHON", sys.executable)
    command = [
        python,
        str(ROOT / "scripts/run_combat.py"),
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--brain",
        "learned" if settings.get("tactics_path") else "rule",
        "--capture",
        "--out",
        str(out),
    ]
    _busy = True
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        _, stderr = await asyncio.wait_for(process.communicate(), 240.0)
        if process.returncode:
            raise RuntimeError("Combat worker failed: " + stderr.decode()[-1200:])
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        physical = manifest["physical"]
        quality = bool(
            physical["finite"]
            and not any(physical["solver_warnings"])
            and physical["penetration_max_m"] <= 0.003
            and physical["penetrating_p99_m"] <= 0.0015
            and physical["external_forces_zero"]
            and physical["model_unchanged"]
        )
        _latest = dict(
            schema="duckfu.executor-report.v1",
            ok=quality,
            run_id=run_id,
            preset=preset,
            seed=seed,
            evidence_domain="simulation",
            usable_for_real_execution=False,
            result=manifest["result"],
            participant_body_ids=[p["body_id"] for p in profiles],
            participant_profiles=profiles,
            manifest_path=str(manifest_path),
            manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
            physical=physical,
            authorization_receipt=None,
            boundary="Executor result only; official authorization/action receipt belongs to ROSClaw",
        )
        return _latest
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        _busy = False


if __name__ == "__main__":
    server.run(transport="stdio")
