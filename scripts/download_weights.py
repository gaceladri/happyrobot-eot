"""Download the model bundle from its public Google Drive link, verify it, and install it under artifacts/deployment.

    uv run python scripts/download_weights.py                 # everything (5.3 GB: Whisper ONNX + Cohere TensorRT plan)
    uv run python scripts/download_weights.py --model whisper-cpu

The bundle is one uncompressed tar holding exactly the files enumerated in
``configs/inference/selected.json``. The tar is verified against a pinned SHA-256 before it is
opened, and every file inside is verified again by ``eot.serving.artifacts.install``. Downloads
resume if interrupted. No account, password or API key is needed: the file is shared read-only with
anyone who has the link, and the link is pinned here.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eot.serving.artifacts import bundle_files, install  # noqa: E402

# Google Drive file id of happyrobot-eot-weights-316125d.tar (shared: anyone with the link, read-only).
DRIVE_FILE_ID = "1Knd2B29FpMuOiUNmvyOV1Nf_uDj2JgI9"
BUNDLE_NAME = "happyrobot-eot-weights-316125d.tar"
BUNDLE_SHA256 = "8c6ca046bde26f9a3860c5fe7e6ec0100cf6c6a70166beda95090814f1eb47b5"
BUNDLE_BYTES = 5_328_250_880
BUNDLE_PREFIX = "model-bundle"  # top-level directory inside the tar
PROGRESS_STEP = 256 << 20


def drive_url(file_id: str) -> str:
    # The usercontent host serves large files directly; confirm=t skips the virus-scan interstitial.
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_bytes: int) -> None:
    """Stream ``url`` to ``destination``, resuming a partial file with an HTTP Range request."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if expected_bytes and have >= expected_bytes:
        partial.replace(destination)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "happyrobot-eot/1.0"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:  # server ignored the range: start over
            have = 0
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                "Google Drive returned a web page instead of the file; the share link may have changed or the quota is exhausted"
            )
        total = have + int(response.headers.get("Content-Length") or 0)
        with partial.open("ab" if have else "wb") as out:
            done, reported = have, have
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if done - reported >= PROGRESS_STEP or done == total:  # one line per 256 MiB, log-friendly
                    print(f"  {done / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
                    reported = done
    partial.replace(destination)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "artifacts/deployment", help="install destination (the Compose default)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/inference/selected.json", help="selected-artifact manifest")
    ap.add_argument("--model", default=None, help="install one model only (whisper-cpu or cohere-gpu); default: both")
    ap.add_argument(
        "--cache", type=Path, default=ROOT / "artifacts/downloads", help="where the tar is kept (delete after install with --clean)"
    )
    ap.add_argument("--url", default=None, help="override the download URL (a mirror of the same tar)")
    ap.add_argument("--clean", action="store_true", help="delete the downloaded tar after a successful install")
    args = ap.parse_args(argv)
    if "REPLACE_WITH" in DRIVE_FILE_ID + BUNDLE_SHA256:
        sys.exit("this script has no download link pinned yet; ask the author for the bundle")

    tar_path = args.cache / BUNDLE_NAME
    if tar_path.exists() and sha256_of(tar_path) == BUNDLE_SHA256:
        print(f"bundle already downloaded and verified: {tar_path}")
    else:
        print(f"downloading {BUNDLE_NAME} ({BUNDLE_BYTES / 2**30:.2f} GiB) -> {tar_path}")
        download(args.url or drive_url(DRIVE_FILE_ID), tar_path, BUNDLE_BYTES)
        actual = sha256_of(tar_path)
        if actual != BUNDLE_SHA256:
            tar_path.unlink()
            sys.exit(f"downloaded bundle hash {actual} does not match the pinned {BUNDLE_SHA256}; deleted it, retry")
        print("bundle SHA-256 verified")

    files = bundle_files(args.manifest, args.model)
    with tempfile.TemporaryDirectory(dir=args.cache, prefix="extract-") as tmp:
        wanted = {f"{BUNDLE_PREFIX}/{entry['path']}" for entry in files}
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.name in wanted]
            missing = wanted - {m.name for m in members}
            if missing:
                sys.exit(f"bundle is missing {sorted(missing)}")
            tar.extractall(tmp, members=members, filter="data")
        installed = install(Path(tmp) / BUNDLE_PREFIX, args.root, files)
    for entry in installed:
        print(f"  installed {entry['path']}  {entry['sha256'][:12]}  {entry['bytes']:,} bytes")
    if args.clean:
        shutil.rmtree(args.cache, ignore_errors=True)
    print(f"done: {len(installed)} files under {args.root}; next: docker compose -f compose.inference.yaml up -d --build whisper-cpu")


if __name__ == "__main__":
    main()
