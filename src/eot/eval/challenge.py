"""Structured-data challenge set for the freight-brokerage phone domain.

HappyRobot's calls are carrier sales / dispatch: MC numbers, load IDs, rates, phone numbers,
ZIPs, dates. Dictated sequences are exactly where pause-based endpointing fails ("my MC is
three four seven ... [0.7 s] ... nine one two"). This module generates scripted user turns with
explicit pause markers and the expected HOLD/EOT decision at each pause, paired with the agent
question that precedes them (so the context branch can be evaluated too).

    uv run eot-challenge --n 200 --out challenge/ [--tts kokoro]

Output ``challenge/script.jsonl`` rows: {id, agent_text, segments: [text, pause_s], label=1,
category}. Each internal pause is a HOLD decision; the end is EOT. If a TTS backend is available
(``--tts kokoro`` requires the ``kokoro`` package + espeak-ng) the segments are synthesised with
real silences and written as wavs + ``clips.jsonl`` compatible with ``eot-mine``. Any TTS voice
used here must be excluded from training; this set is for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from eot.audio import SAMPLE_RATE, resample
from eot.io import atomic_write_jsonl, write_wav

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
CITIES = ["Dallas", "Atlanta", "Chicago", "Memphis", "Laredo", "Phoenix", "Columbus", "Kansas City", "Newark", "Savannah"]
FILLERS = ["uh", "um", "let me see", "hold on", "one second", "hmm"]


def _digits(n: int, rng: random.Random) -> list[str]:
    return [DIGITS[rng.randrange(10)] for _ in range(n)]


def _chunk(words: list[str], rng: random.Random) -> list[tuple[str, float]]:
    """Split a digit string into dictation groups separated by realistic pauses."""
    out, i = [], 0
    while i < len(words):
        k = rng.choice([2, 3, 3, 4])
        grp = " ".join(words[i : i + k])
        i += k
        pause = rng.choice([0.3, 0.5, 0.7, 0.9, 1.2]) if i < len(words) else 0.0
        out.append((grp, pause))
    return out


def make_turn(rng: random.Random, idx: int) -> dict:
    cat = rng.choice(["mc_number", "load_id", "rate", "phone", "location", "availability", "short_answer", "enumeration"])
    if cat == "mc_number":
        agent = rng.choice(["What's your MC number?", "Can I get your MC or DOT number?"])
        segs = [(rng.choice(["yeah it's", "MC number is", "sure,"]), rng.choice([0.2, 0.4]))] + _chunk(_digits(6, rng), rng)
    elif cat == "load_id":
        agent = "Which load are you calling about?"
        segs = [("the load number is", 0.3)] + _chunk(_digits(rng.choice([5, 7, 8]), rng), rng)
    elif cat == "rate":
        agent = rng.choice(["What rate are you looking for?", "What can you do on this load?"])
        amount = rng.choice(["fourteen fifty", "two thousand", "eighteen hundred", "twenty two hundred", "twelve seventy five"])
        segs = [(rng.choice(FILLERS), rng.choice([0.5, 0.8, 1.0])), ("I could do", rng.choice([0.3, 0.6])), (f"{amount} all in", 0.0)]
    elif cat == "phone":
        agent = "What's a good callback number?"
        d = _digits(10, rng)
        segs = [(" ".join(d[:3]), 0.5), (" ".join(d[3:6]), 0.5), (" ".join(d[6:]), 0.0)]
    elif cat == "location":
        agent = "Where is the truck right now?"
        segs = [
            ("we're in", 0.3),
            (rng.choice(CITIES), rng.choice([0.4, 0.7])),
            (rng.choice(["heading east", "empty since this morning", "about two hours out"]), 0.0),
        ]
    elif cat == "availability":
        agent = "When can you pick up?"
        segs = [
            (rng.choice(["tomorrow", "Friday", "tonight"]), rng.choice([0.3, 0.6])),
            (rng.choice(["around", "probably"]), 0.4),
            (rng.choice(["eight a m", "noon", "six in the evening"]), 0.0),
        ]
    elif cat == "short_answer":
        agent = rng.choice(
            ["Is the truck a fifty three foot dry van?", "Can you do it for fifteen hundred?", "Anything else I can help with?"]
        )
        segs = [(rng.choice(["yes", "yeah that works", "no that's it", "correct", "nope"]), 0.0)]
    else:
        agent = "What equipment do you run?"
        items = rng.sample(["dry van", "reefer", "flatbed", "step deck", "power only"], 3)
        segs = [(items[0], 0.5), (str(items[1]), 0.6), (f"and {items[2]}", 0.0)]
    return {
        "id": f"chal_{idx:04d}",
        "category": cat,
        "agent_text": agent,
        "segments": [{"text": t, "pause_s": p} for t, p in segs],
        "label": 1,
        "n_hold_decisions": sum(1 for _, p in segs if p > 0),
    }


def generate(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [make_turn(rng, i) for i in range(n)]


def synthesize(rows: list[dict], out_dir: Path, tts: str) -> Path:
    """Render segments with a TTS backend and stitch with real silences. Currently: kokoro."""
    if tts != "kokoro":
        raise ValueError(f"unknown tts backend {tts}")
    from kokoro import KPipeline  # type: ignore

    pipe = KPipeline(lang_code="a")
    voices = ["af_heart", "am_michael", "bf_emma", "am_adam"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "clips.jsonl"
    clips = []
    for i, r in enumerate(rows):
        parts = []
        for seg in r["segments"]:
            audio = np.concatenate([np.asarray(a) for _, _, a in pipe(seg["text"], voice=voices[i % len(voices)])])
            parts.append(resample(audio.astype(np.float32), 24_000))
            parts.append(np.zeros(int(seg["pause_s"] * SAMPLE_RATE), np.float32))
        parts.append(np.zeros(int(0.3 * SAMPLE_RATE), np.float32))
        p = out_dir / f"{r['id']}.wav"
        write_wav(p, np.concatenate(parts), SAMPLE_RATE)
        clips.append(
            {
                "id": r["id"],
                "path": p.name,
                "label": 1,
                "source": f"challenge_{tts}",
                "agent_text": r["agent_text"],
                "category": r["category"],
            }
        )
    atomic_write_jsonl(manifest, clips)
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of scripted user turns")
    ap.add_argument("--seed", type=int, default=0, help="script generator seed")
    ap.add_argument("--out", type=Path, required=True, help="directory for script.jsonl (and audio/ with --tts)")
    ap.add_argument("--tts", default=None, help="optional TTS backend, e.g. kokoro")
    args = ap.parse_args(argv)
    rows = generate(args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.out / "script.jsonl", rows)
    summary = {
        "n": len(rows),
        "hold_decisions": sum(r["n_hold_decisions"] for r in rows),
        "categories": sorted({r["category"] for r in rows}),
    }
    if args.tts:
        summary["manifest"] = str(synthesize(rows, args.out / "audio", args.tts))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
