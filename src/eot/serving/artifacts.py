"""Verify or install an explicitly enumerated inference bundle without loading models.

    uv run eot-artifacts verify [--manifest configs/inference/selected.json] [--root artifacts/deployment]
    uv run eot-artifacts install --source /path/to/bundle [--model whisper-cpu]

The manifest is versioned; model files stay outside Git. Installation validates all source files
first and refuses to replace a different local artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from eot.io import sha256_file

MANIFEST_FORMAT = "eot-inference-bundle-v1"


def bundle_files(manifest: Path, model: str | None = None) -> list[dict]:
    """Validated file entries (``path``, ``sha256``) of every model in the manifest, or of ``model`` only."""
    payload = json.loads(manifest.read_text())
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("Unsupported inference manifest format")
    models = payload["models"]
    if model is not None and model not in models:
        raise ValueError(f"Unknown model: {model}; manifest lists {sorted(models)}")
    files: list[dict] = []
    seen: set[str] = set()
    for name, metadata in models.items():
        if model is not None and name != model:
            continue
        for entry in metadata["files"]:
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError(f"Unsafe artifact path: {relative}")
            if str(relative) in seen or not relative.parts:
                raise ValueError(f"Duplicate or empty artifact path: {relative}")
            if len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"]):
                raise ValueError(f"Invalid SHA256 for {relative}")
            seen.add(str(relative))
            files.append(entry)
    if not files:
        raise ValueError("Empty artifact bundle")
    return files


def contained(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes bundle directory: {relative}")
    return path


def verify(root: Path, files: list[dict]) -> list[dict]:
    results = []
    for entry in files:
        path = contained(root, entry["path"])
        if not path.is_file():
            raise ValueError(f"Missing artifact: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        results.append({"path": entry["path"], "sha256": actual, "bytes": path.stat().st_size})
    return results


def install(source: Path, destination: Path, files: list[dict]) -> list[dict]:
    verify(source, files)
    # Check every destination before copying any file. Existing correct files are reusable.
    for entry in files:
        target = contained(destination, entry["path"])
        if target.exists() and (not target.is_file() or sha256_file(target) != entry["sha256"]):
            raise ValueError(f"Refusing to replace a different artifact: {target}")
    for entry in files:
        target = contained(destination, entry["path"])
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                with contained(source, entry["path"]).open("rb") as stream:
                    shutil.copyfileobj(stream, output)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(temporary) != entry["sha256"]:
                raise ValueError(f"Source changed during copy: {entry['path']}")
            temporary.chmod(0o644)
            # Atomic publication without replacing files created by another process.
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return verify(destination, files)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("verify", "install"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/inference/selected.json"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/deployment"))
    parser.add_argument("--source", type=Path, help="Existing bundle root, for install")
    parser.add_argument("--model", help="One model name from the manifest (default: every model)")
    args = parser.parse_args(argv)
    if args.action == "install" and args.source is None:
        parser.error("install requires --source")
    try:
        files = bundle_files(args.manifest, args.model)
        result = install(args.source, args.root, files) if args.action == "install" else verify(args.root, files)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Artifact verification failed: {exc}\n")
    print(json.dumps({"status": "VERIFIED", "root": str(args.root), "files": result}, indent=2))


if __name__ == "__main__":
    main()
