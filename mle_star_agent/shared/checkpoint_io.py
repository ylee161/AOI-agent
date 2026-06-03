import json
import os
from pathlib import Path


class _CheckpointEncoder(json.JSONEncoder):
    def default(self, obj):
        # Safely convert numpy scalars so they round-trip as the right type.
        # Import lazily so the module works without numpy installed.
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use with_name so the .tmp sibling always has the full original name as a
    # prefix — avoids the with_suffix(".tmp") pitfall that strips ".json".
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, cls=_CheckpointEncoder), encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Checkpoint at {path} is unreadable or corrupt: {e}") from e


def checkpoint_exists(path: Path) -> bool:
    return path.is_file()
