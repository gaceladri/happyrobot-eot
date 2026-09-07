"""Compare serialized HTTP outputs before/after a service release on frozen probes.

This tests a source/runtime release, not endpointing quality or throughput.
The reference and candidate must serve the exact same model artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np

from eot.audio import SAMPLE_RATE, to_pcm16_bytes


def capture(inputs: Path, urls: dict[str, str]) -> dict:
    with inputs.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with np.load(inputs, allow_pickle=False) as panel:
        waves = panel["waves"]
    result = {"panel_sha256": digest, "examples": len(waves), "models": {}}
    with httpx.Client(timeout=30) as client:
        for name, url in urls.items():
            response = client.get(url + "/healthz")
            response.raise_for_status()
            health = response.json()
            probabilities, payloads = [], []
            for wave in waves:
                body = to_pcm16_bytes(wave)
                response = client.post(url + "/v1/eot", params={"sr": SAMPLE_RATE}, content=body)
                response.raise_for_status()
                value = response.json()
                if value["model_sha"] != health["model_sha"]:
                    raise ValueError("Model changed during verification")
                probabilities.append(value["p_eot"])
                payloads.append(hashlib.sha256(body).hexdigest())
            result["models"][name] = {
                "health": health,
                "probabilities": probabilities,
                "payload_sha256": payloads,
            }
    return result


def compare(reference: dict, candidate: dict, tolerance: float) -> dict:
    if reference["panel_sha256"] != candidate["panel_sha256"]:
        raise ValueError("Probe panels differ")
    if reference["models"].keys() != candidate["models"].keys():
        raise ValueError("Model sets differ")
    differences = {}
    for name, model in candidate["models"].items():
        original = reference["models"][name]
        if original["health"]["model_sha"] != model["health"]["model_sha"]:
            raise ValueError(f"Model artifact changed: {name}")
        if original["payload_sha256"] != model["payload_sha256"]:
            raise ValueError(f"HTTP payloads differ: {name}")
        actual = np.asarray(model["probabilities"])
        previous = np.asarray(original["probabilities"])
        if not np.isfinite(actual).all() or not np.isfinite(previous).all():
            raise ValueError("Nonfinite HTTP probabilities")
        differences[name] = float(np.max(np.abs(actual - previous)))
    return {
        "accepted": max(differences.values()) <= tolerance,
        "tolerance": tolerance,
        "max_probability_difference": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cpu-url", default="http://127.0.0.1:8891")
    parser.add_argument("--gpu-url", default="http://127.0.0.1:8892")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; preserve previous verification")
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("Tolerance must be finite and nonnegative")
    result = capture(args.inputs, {"whisper-cpu": args.cpu_url, "cohere-gpu": args.gpu_url})
    if args.reference:
        result["comparison"] = compare(json.loads(args.reference.read_text()), result, args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("comparison", {"captured": result["examples"]})))
    if args.reference and not result["comparison"]["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
