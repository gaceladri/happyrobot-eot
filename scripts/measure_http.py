"""Actual Docker HTTP timings on the same frozen payload as the earlier measurements."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path

import httpx

from eot.serving.load_test import _payload, run_level


async def measure(url, out, container, levels, requests, root):
    audio = root / "data/mined/smart-turn-ensemble/audio/aaf8a162-468b-4af9-ac77-25803837275c_00-026a1283f637.wav"
    body = _payload(str(audio), 8)
    digest = hashlib.sha256(body).hexdigest()
    assert digest == "a0579bc7560032f6137237177f09da25b7cb0d28c0f7ccff9f2c1b4bc25908cc"
    ready = time.perf_counter()
    async with httpx.AsyncClient(timeout=15) as client:
        for _ in range(90):
            try:
                health = (await client.get(url + "/healthz")).json()
                if health["ok"]:
                    break
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Container did not become healthy")
        wait = time.perf_counter() - ready

        async def request():
            r = await client.post(url + "/v1/eot?sr=16000", content=body)
            r.raise_for_status()
            return r.json()

        first = await request()
        for _ in range(4):
            await asyncio.gather(*[request() for _ in range(8)])
    record = {
        "container": container,
        "container_inspect": json.loads(await asyncio.to_thread(subprocess.check_output, ["docker", "inspect", container]))[0],
        "payload_sha256": digest,
        "payload_bytes": len(body),
        "health": health,
        "ready_wait_seconds": wait,
        "first_response": first,
        "levels": [],
        "timing": "Warm full HTTP; same frozen training PCM16 payload; no inference cache",
    }
    record["gpu_before"] = (
        await asyncio.to_thread(
            subprocess.check_output,
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        )
    ).strip()
    for concurrency in levels:
        row = await run_level(url, body, concurrency, requests)
        record["levels"].append(row)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    async with httpx.AsyncClient(timeout=15) as client:
        record["health_after"] = (await client.get(url + "/healthz")).json()
    record["gpu_after"] = (
        await asyncio.to_thread(
            subprocess.check_output,
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        )
    ).strip()
    record["accepted"] = all(r["errors"] == 0 for r in record["levels"])
    out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--requests", type=int, default=400)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    if a.out.exists():
        p.error("Output exists; preserve prior measurements")
    if a.requests < 1 or any(level < 1 for level in a.levels):
        p.error("Requests and concurrency must be positive")
    asyncio.run(measure(a.url, a.out, a.container, a.levels, a.requests, a.root))
