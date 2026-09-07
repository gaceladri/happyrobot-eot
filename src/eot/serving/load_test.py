"""Stress test for the inference service: closed-loop concurrency sweep with latency percentiles.

    uv run eot-loadtest --url http://localhost:8000 --concurrency 1 4 8 --requests 200 [--wav clip.wav]

Sends PCM16 bodies (8 s of a real clip if ``--wav`` is given, otherwise synthetic speech-like
noise) and reports, per concurrency level: throughput (req/s), client p50/p95/p99 total latency,
and the server-reported inference/feature timings so network/queue overhead is visible as the
difference. The first measured request is already warmed because the server warms the model at
startup; measure container startup-to-ready separately if cold-start latency is required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter

import httpx
import numpy as np

from eot.audio import SAMPLE_RATE, load_wav, to_pcm16_bytes


def _payload(wav: str | None, seconds: float) -> bytes:
    if wav:
        x = load_wav(wav)
    else:
        rng = np.random.default_rng(0)
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        x = (0.2 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.normal(size=len(t))).astype(
            np.float32
        )
    return to_pcm16_bytes(x[-int(seconds * SAMPLE_RATE) :])


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


async def run_level(url: str, body: bytes, concurrency: int, n_requests: int) -> dict:
    lat, inf, feat, errors = [], [], [], 0
    statuses = Counter()
    all_latencies = []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=30.0) as client:

        async def one():
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"{url}/v1/eot", params={"sr": SAMPLE_RATE}, content=body)
                    dt = (time.perf_counter() - t0) * 1000
                    statuses[str(r.status_code)] += 1
                    all_latencies.append(dt)
                    if r.status_code != 200:
                        errors += 1
                        return
                    j = r.json()
                    lat.append(dt)
                    inf.append(j["timings_ms"]["inference"])
                    feat.append(j["timings_ms"]["features"])
                except Exception:  # noqa: BLE001
                    errors += 1
                    statuses["transport_error"] += 1
                    all_latencies.append((time.perf_counter() - t0) * 1000)

        t_all = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n_requests)))
        wall = time.perf_counter() - t_all
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "errors": errors,
        "throughput_rps": len(lat) / wall if wall else 0,
        "status_counts": dict(statuses),
        "all_attempts_ms": {"p95": pct(all_latencies, 95), "p99": pct(all_latencies, 99)},
        "meets_100ms_p95_no_errors": errors == 0 and bool(lat) and pct(lat, 95) < 100,
        "client_total_ms": {
            "p50": pct(lat, 50),
            "p95": pct(lat, 95),
            "p99": pct(lat, 99),
            "mean": statistics.fmean(lat) if lat else float("nan"),
        },
        "server_inference_ms": {"p50": pct(inf, 50), "p95": pct(inf, 95), "p99": pct(inf, 99)},
        "server_features_ms": {"p50": pct(feat, 50), "p95": pct(feat, 95)},
    }


async def _main_async(args) -> dict:
    body = _payload(args.wav, args.seconds)
    async with httpx.AsyncClient(timeout=30.0) as client:
        health = (await client.get(f"{args.url}/healthz")).json()
        t0 = time.perf_counter()
        first = await client.post(f"{args.url}/v1/eot", params={"sr": SAMPLE_RATE}, content=body)
        warmed_first_ms = (time.perf_counter() - t0) * 1000
        first.raise_for_status()
    levels = [await run_level(args.url, body, c, args.requests) for c in args.concurrency]
    return {
        "health": health,
        "payload_bytes": len(body),
        "audio_seconds": args.seconds,
        "warmed_first_request_ms": warmed_first_ms,
        "levels": levels,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report = asyncio.run(_main_async(args))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
