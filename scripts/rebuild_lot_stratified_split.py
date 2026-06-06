#!/usr/bin/env python3
"""Rebuild checkpoints/data_split_grouped.json with lot+label stratification."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_PATH = Path("checkpoints/data_split_grouped.json")
SEED = 20260606
RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


def _bucket_counts(n):
    raw = {name: n * ratio for name, ratio in RATIOS.items()}
    counts = {name: int(raw[name]) for name in RATIOS}
    remaining = n - sum(counts.values())
    order = sorted(RATIOS, key=lambda name: (raw[name] - counts[name], RATIOS[name]), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    if n >= 3:
        for name in RATIOS:
            if counts[name] == 0:
                donor = max(RATIOS, key=lambda candidate: counts[candidate])
                counts[donor] -= 1
                counts[name] += 1
    return counts


def main():
    with SPLIT_PATH.open() as f:
        original = json.load(f)

    samples = []
    for split_name in ("train", "val", "test"):
        samples.extend(original[split_name])

    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate sample_id values found before rebuilding split")

    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["lot"], sample["label"])].append(sample)

    rng = random.Random(SEED)
    rebuilt = {
        "metadata": dict(original.get("metadata", {})),
        "train": [],
        "val": [],
        "test": [],
    }
    rebuilt["metadata"]["split_strategy"] = "lot_label_stratified_60_20_20"
    rebuilt["metadata"]["split_seed"] = SEED

    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        counts = _bucket_counts(len(bucket))
        cursor = 0
        for split_name in ("train", "val", "test"):
            take = counts[split_name]
            rebuilt[split_name].extend(bucket[cursor:cursor + take])
            cursor += take

    for split_name in ("train", "val", "test"):
        rebuilt[split_name].sort(key=lambda sample: sample["sample_id"])

    all_lots = {sample["lot"] for sample in samples}
    for split_name in ("train", "val", "test"):
        lots = {sample["lot"] for sample in rebuilt[split_name]}
        if lots != all_lots:
            raise RuntimeError(f"{split_name} lot coverage mismatch: {sorted(lots)}")

    rebuilt_ids = [sample["sample_id"] for name in ("train", "val", "test") for sample in rebuilt[name]]
    if Counter(rebuilt_ids) != Counter(sample["sample_id"] for sample in samples):
        raise RuntimeError("Rebuilt split does not preserve the original sample set exactly")

    with SPLIT_PATH.open("w") as f:
        json.dump(rebuilt, f, indent=2)
        f.write("\n")

    for split_name in ("train", "val", "test"):
        lot_counts = Counter(sample["lot"] for sample in rebuilt[split_name])
        label_counts = Counter(sample["label"] for sample in rebuilt[split_name])
        print(
            f"{split_name}: n={len(rebuilt[split_name])} "
            f"lots={dict(sorted(lot_counts.items()))} labels={dict(sorted(label_counts.items()))}"
        )


if __name__ == "__main__":
    main()
