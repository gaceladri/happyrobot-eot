"""Bundle integrity and non-destructive installation contracts."""

import hashlib
import json

import pytest

from eot.serving.artifacts import bundle_files, install, verify


def entries(*names):
    return [{"path": name, "sha256": hashlib.sha256(b"model").hexdigest()} for name in names]


def test_install_is_repeatable_and_verifies_bytes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "cpu").mkdir(parents=True)
    (source / "cpu/model").write_bytes(b"model")
    files = entries("cpu/model")
    assert install(source, target, files) == verify(source, files)
    assert install(source, target, files) == verify(target, files)
    (target / "cpu/model").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify(target, files)


def test_install_checks_whole_bundle_before_copying(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in ("first", "second"):
        (source / name).write_bytes(b"model")
    (target / "second").write_bytes(b"keep existing work")
    with pytest.raises(ValueError, match="Refusing to replace"):
        install(source, target, entries("first", "second"))
    assert not (target / "first").exists()
    assert (target / "second").read_bytes() == b"keep existing work"


@pytest.mark.parametrize("name", ["../outside", "/absolute", "cpu/../../outside", "cpu\\outside"])
def test_manifest_rejects_path_escape(tmp_path, name):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format": "eot-inference-bundle-v1",
                "models": {"cpu": {"files": entries(name)}},
            }
        )
    )
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        bundle_files(path)


def test_symlink_cannot_escape_bundle(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"model")
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "model").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes bundle"):
        verify(root, entries("model"))


def test_model_selection_is_validated_against_the_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format": "eot-inference-bundle-v1", "models": {"cpu": {"files": entries("cpu/model")}}}))
    assert bundle_files(path, "cpu") == entries("cpu/model")
    with pytest.raises(ValueError, match="Unknown model: gpu"):
        bundle_files(path, "gpu")
