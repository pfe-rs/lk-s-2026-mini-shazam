# TODO: NE pamti koji model je generisao embeddinge u bazi.
# Upit baze sa drugim modelom ce NEVIDLJIVO vratiti pogresne rezultate!!!

import hashlib
import json
from pathlib import Path

class ModelRegistry:
    def __init__(self, manifest_path: str = "models/manifest.json"):
        self.manifest_path = Path(manifest_path)
        self._manifest: dict[str, dict] = {}
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())

    def register(self, model_id: str, path: str, checksum: str = "") -> None:
        entry = {"path": path, "checksum": checksum or self._compute_checksum(path)}
        self._manifest[model_id] = entry
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def get(self, model_id: str) -> str:
        entry = self._manifest.get(model_id)
        if entry is None:
            raise KeyError(f"model not registered: {model_id}")
        return entry["path"]

    def verify_checksum(self, model_id: str) -> bool:
        entry = self._manifest.get(model_id)
        if entry is None:
            raise KeyError(f"model not registered: {model_id}")
        return entry["checksum"] == self._compute_checksum(entry["path"])

    def list_models(self) -> list[str]:
        return list(self._manifest.keys())

    @staticmethod
    def _compute_checksum(path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return hashlib.sha256(p.read_bytes()).hexdigest()
