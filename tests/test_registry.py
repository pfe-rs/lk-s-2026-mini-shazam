import tempfile
from pathlib import Path

from pipeline.registry import ModelRegistry


def test_register_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "models" / "manifest.json"
        reg = ModelRegistry(str(manifest))
        model_path = Path(tmp) / "model.onnx"
        model_path.write_bytes(b"dummy model data")
        reg.register("v1", str(model_path))
        assert reg.get("v1") == str(model_path)


def test_get_unknown_raises():
    reg = ModelRegistry()
    try:
        reg.get("nonexistent")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_list_models():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        reg = ModelRegistry(str(manifest))
        model_path = Path(tmp) / "model.onnx"
        model_path.write_bytes(b"dummy")
        reg.register("v1", str(model_path))
        reg.register("v2", str(model_path))
        assert set(reg.list_models()) == {"v1", "v2"}


def test_verify_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        reg = ModelRegistry(str(manifest))
        model_path = Path(tmp) / "model.onnx"
        model_path.write_bytes(b"hello world")
        reg.register("v1", str(model_path))
        assert reg.verify_checksum("v1") is True
        model_path.write_bytes(b"tampered")
        assert reg.verify_checksum("v1") is False


def test_load_persisted_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        model_path = Path(tmp) / "model.onnx"
        model_path.write_bytes(b"persist test")
        reg1 = ModelRegistry(str(manifest))
        reg1.register("v1", str(model_path))
        reg2 = ModelRegistry(str(manifest))
        assert reg2.get("v1") == str(model_path)
