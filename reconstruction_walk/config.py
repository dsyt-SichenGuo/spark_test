"""Scene descriptions are separate from locomotion and rendering implementations."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "reconstructions/office-hallway-matcha/configs/new_ratio1.json"


@dataclass
class SceneConfig:
    path: Path
    data: dict

    @classmethod
    def read(cls, path=DEFAULT_CONFIG):
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported scene config version")
        result = cls(path, data)
        for key in ("gaussian", "mesh", "cameras"):
            asset = result.asset(data[key]["path"] if isinstance(data[key], dict) else data[key])
            if not asset.is_file():
                raise FileNotFoundError(asset)
        if not math.isfinite(data.get("camera_height_m", 0)) or data.get("camera_height_m", 0) <= 0:
            raise ValueError("camera_height_m must be positive")
        if data.get("meters_per_unit") is not None and (
                not math.isfinite(data["meters_per_unit"]) or data["meters_per_unit"] <= 0):
            raise ValueError("meters_per_unit must be positive")
        for name in ("collision_voxel_m", "footprint_radius_m", "spawn_clearance_m"):
            if name in data and (not math.isfinite(data[name]) or data[name] <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if data.get("max_collision_faces", 800000) <= 0:
            raise ValueError("max_collision_faces must be positive")
        return result

    def asset(self, value):
        path = Path(value).expanduser()
        return (self.path.parent / path).resolve() if not path.is_absolute() else path.resolve()

    @property
    def cache(self):
        return self.asset(self.data["cache"])

    def asset_signature(self):
        paths = [self.asset(self.data[k]["path"] if isinstance(self.data[k], dict) else self.data[k])
                 for k in ("gaussian", "mesh", "cameras")]
        return {"config": self.data, "files": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths]}
