#!/usr/bin/env python3
"""Lot-generalisation repair probe for the AOI dataset.

This script leaves the checkpointed split untouched. It rebuilds an in-process
lot-label-stratified split from the same samples so training includes examples
from every available lot, then trains a small image-feature classifier and
reports parser-compatible METRICS and PREDICTIONS lines.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


SEED = 7
IMAGE_SIZE = 16
DATA_SPLIT_PATH = Path("checkpoints/data_split_grouped.json")
DATASET_DIR = Path("aoi_agent_dataset")


def _label_to_int(label):
    return 1 if str(label).strip().upper() == "NG" else 0


def _short_lot(lot):
    text = str(lot)
    return text.rsplit("VHB", 1)[-1] if "VHB" in text else text


def load_samples():
    with DATA_SPLIT_PATH.open() as f:
        split = json.load(f)
    samples = list(split["train"]) + list(split["val"]) + list(split["test"])
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    for sample in samples:
        for key in ("img_l", "img_r"):
            if not Path(sample[key]).exists():
                raise FileNotFoundError(sample[key])
    return split, samples


def print_original_split_evidence(split):
    for name in ("train", "val", "test"):
        lots = Counter(sample["lot"] for sample in split[name])
        labels = Counter(sample["label"] for sample in split[name])
        print(
            f"ORIGINAL_SPLIT {name}: n={len(split[name])} "
            f"lots={json.dumps(dict(lots), sort_keys=True)} "
            f"labels={json.dumps(dict(labels), sort_keys=True)}"
        )
    train_lots = {sample["lot"] for sample in split["train"]}
    val_lots = {sample["lot"] for sample in split["val"]}
    test_lots = {sample["lot"] for sample in split["test"]}
    print(f"ORIGINAL_TRAIN_VAL_LOT_OVERLAP: {sorted(train_lots & val_lots)}")
    print(f"ORIGINAL_TRAIN_TEST_LOT_OVERLAP: {sorted(train_lots & test_lots)}")


def stratified_lot_label_split(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample["lot"], sample["label"])].append(sample)

    rng = random.Random(SEED)
    train, val, test = [], [], []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        n = len(group)
        if n >= 5:
            n_test = max(1, int(round(n * 0.20)))
            n_val = max(1, int(round(n * 0.20)))
        elif n >= 2:
            n_test = 1
            n_val = 0
        else:
            n_test = 0
            n_val = 0
        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    for name, split_samples in (("train", train), ("val", val), ("test", test)):
        lots = sorted({_short_lot(sample["lot"]) for sample in split_samples})
        labels = Counter(sample["label"] for sample in split_samples)
        print(
            f"REPAIRED_SPLIT {name}: n={len(split_samples)} "
            f"lots={lots} labels={dict(labels)}"
        )
    return train, val, test


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _read_norm_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    image = _CLAHE.apply(image)
    return image.astype(np.float32) / 255.0


def extract_features(sample):
    left = _read_norm_gray(sample["img_l"])
    right = _read_norm_gray(sample["img_r"])
    diff = np.abs(left - right)
    mean_pair = (left + right) * 0.5

    features = []
    for image in (left, right, diff, mean_pair):
        features.extend(image.ravel())
        features.extend(
            (
                float(image.mean()),
                float(image.std()),
                float(np.percentile(image, 5)),
                float(np.percentile(image, 95)),
            )
        )
    return np.asarray(features, dtype=np.float32)


def build_matrix(samples):
    x = np.vstack([extract_features(sample) for sample in samples])
    y = np.asarray([_label_to_int(sample["label"]) for sample in samples], dtype=np.int64)
    return x, y


def choose_threshold(y_true, scores):
    best_threshold = 0.5
    best_tuple = (-1.0, -1.0, -1.0)
    for threshold in np.linspace(0.01, 0.99, 199):
        pred = (scores >= threshold).astype(np.int64)
        f1 = f1_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        candidate = (f1, recall, precision)
        if candidate > best_tuple:
            best_tuple = candidate
            best_threshold = float(threshold)
    return best_threshold


def safe_auc(y_true, scores):
    if len(set(int(v) for v in y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, scores))


def evaluate(samples, y_true, scores, threshold):
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    roc_auc = safe_auc(y_true, scores)
    g_count = int((y_true == 0).sum())
    ng_count = int((y_true == 1).sum())
    overkill = fp / g_count if g_count else 0.0
    miss_rate = fn / ng_count if ng_count else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    metrics = {
        "roc_auc": roc_auc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "ng_recall": recall,
        "miss_rate": float(miss_rate),
        "overkill_rate": float(overkill),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "ng_count": ng_count,
        "g_count": g_count,
        "avg_latency_ms": 0.0,
    }
    predictions = []
    for sample, truth, label_pred, score in zip(samples, y_true, pred, scores):
        predictions.append(
            {
                "image_path": sample["img_l"],
                "img_l": sample["img_l"],
                "img_r": sample["img_r"],
                "sample_id": sample.get("sample_id"),
                "lot": sample.get("lot"),
                "true_label": "NG" if int(truth) == 1 else "G",
                "predicted_label": "NG" if int(label_pred) == 1 else "G",
                "score": float(score),
                "ng_probability": float(score),
                "threshold": float(threshold),
            }
        )
    return metrics, predictions


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    split, samples = load_samples()
    print_original_split_evidence(split)
    train_samples, val_samples, test_samples = stratified_lot_label_split(samples)

    x_train, y_train = build_matrix(train_samples)
    x_val, y_val = build_matrix(val_samples)
    x_test, y_test = build_matrix(test_samples)

    model = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, val_scores)
    test_scores = model.predict_proba(x_test)[:, 1]
    metrics, predictions = evaluate(test_samples, y_test, test_scores, threshold)

    print("METRICS: " + json.dumps(metrics, sort_keys=True))
    print("PREDICTIONS: " + json.dumps(predictions, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
