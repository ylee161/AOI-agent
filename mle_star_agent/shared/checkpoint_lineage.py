import hashlib
import json
from pathlib import Path
from typing import Any

from mle_star_agent import config


def stable_json_sha256(value: Any) -> str:
    def _default(obj):
        if isinstance(obj, (set, frozenset)):
            try:
                return sorted(obj)  # sets have non-deterministic str() across processes
            except TypeError:
                return sorted(str(x) for x in obj)  # mixed types: coerce to str
        return str(obj)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_split_sha256() -> str | None:
    return file_sha256(config.CKPT_DATA_SPLIT)


def acceptance_config_sha256() -> str:
    return stable_json_sha256({
        "accuracy_relaxed_min": config.ACCURACY_RELAXED_MIN,
        "ng_recall_relaxed_min": config.NG_RECALL_RELAXED_MIN,
        "miss_rate_relaxed_max": config.MISS_RATE_RELAXED_MAX,
        "overkill_relaxed_max": config.OVERKILL_RELAXED_MAX,
        "accuracy_final_min": config.ACCURACY_FINAL_MIN,
        "ng_recall_final_min": config.NG_RECALL_FINAL_MIN,
        "miss_rate_final_max": config.MISS_RATE_FINAL_MAX,
        "overkill_final_max": config.OVERKILL_FINAL_MAX,
        "threshold_min": config.THRESHOLD_MIN,
        "threshold_max": config.THRESHOLD_MAX,
        "threshold_step": config.THRESHOLD_STEP,
    })


def script_lineage(script: str, script_key: str = "script_sha256") -> dict:
    return {
        script_key: text_sha256(script),
        "data_split_sha256": data_split_sha256(),
        "acceptance_config_sha256": acceptance_config_sha256(),
    }


def lineage_matches(saved: dict | None, current: dict, keys: list[str] | None = None) -> bool:
    if not isinstance(saved, dict):
        return False
    keys_to_check = keys or sorted(current.keys())
    return all(saved.get(k) == current.get(k) for k in keys_to_check)
