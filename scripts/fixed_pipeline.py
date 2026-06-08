#!/usr/bin/env python3
"""Candidate 1: EfficientNet-B0 — 9-channel pixel-diff stereo, full fine-tune (~5.3M)."""
import os, json, random, time, math, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from skimage.exposure import match_histograms
from PIL import Image

# ── Dry-run support ──
DRY_RUN = os.getenv("DRY_RUN") == "1"
DRY_RUN_EPOCHS = int(os.getenv("DRY_RUN_EPOCHS", "1"))
DRY_RUN_SAMPLES = int(os.getenv("DRY_RUN_SAMPLES", "10"))

# ── Reproducibility (from env) ──
_seed = int(os.environ.get("AOI_RANDOM_SEED", os.environ.get("SEED", "42")))
random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)

# ── Acceptance criteria ──
MISS_BUDGET     = 0.03
OVERKILL_BUDGET = 0.08
FP_MAX          = 2
BASE_LR         = 1e-3
epochs          = DRY_RUN_EPOCHS if DRY_RUN else 20
PATIENCE        = 3
BATCH_SIZE      = 8
IMAGE_SIZE      = 224
PROBE_EPOCHS    = min(DRY_RUN_EPOCHS, 5) if DRY_RUN else 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, Dry-run: {DRY_RUN}, Epochs: {epochs}")

# ── Data split ──
DATA_SPLIT_PATH = "/Users/yishinn/Downloads/Aoi agent 2/checkpoints/data_split_grouped.json"
with open(DATA_SPLIT_PATH) as f:
    data_split = json.load(f)

train_samples = data_split["train"]
val_samples   = data_split["val"]
test_samples  = data_split["test"]

reference_sample = next((s for s in train_samples if s["label"] == "G"), train_samples[0])
REFERENCE_IMAGE = np.array(
    TF.resize(Image.open(reference_sample["img_l"]).convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE))
)
MATCHED_IMAGE_CACHE = {}


def match_to_reference(image):
    matched = match_histograms(np.array(image), REFERENCE_IMAGE, channel_axis=-1)
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))

if DRY_RUN:
    train_samples = train_samples[:DRY_RUN_SAMPLES]
    val_samples   = val_samples[:DRY_RUN_SAMPLES]
    test_samples  = test_samples[:DRY_RUN_SAMPLES]

print(f"Samples — train:{len(train_samples)} val:{len(val_samples)} test:{len(test_samples)}")

# ── Stereo dataset (9-channel: L, R, abs(L-R)) ──
class StereoDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        label = 0 if s["label"] == "G" else 1

        img_l = self._load_normalized(s["img_l"])
        img_r = self._load_normalized(s["img_r"])

        if self.augment:
            # Identical geometric transforms for L & R
            if random.random() < 0.5:
                img_l = TF.hflip(img_l)
                img_r = TF.hflip(img_r)
            angle = random.uniform(-5, 5)
            img_l = TF.rotate(img_l, angle, fill=0)
            img_r = TF.rotate(img_r, angle, fill=0)
            # Mild colour jitter — identical params
            bf = random.uniform(0.9, 1.1)
            cf = random.uniform(0.9, 1.1)
            img_l = TF.adjust_brightness(img_l, bf)
            img_l = TF.adjust_contrast(img_l, cf)
            img_r = TF.adjust_brightness(img_r, bf)
            img_r = TF.adjust_contrast(img_r, cf)

        # To tensor + ImageNet normalisation
        img_l = TF.to_tensor(img_l)
        img_r = TF.to_tensor(img_r)
        img_l = TF.normalize(img_l, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        img_r = TF.normalize(img_r, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        diff = torch.abs(img_l - img_r)
        stacked = torch.cat([img_l, img_r, diff], dim=0)  # 9 channels
        return stacked, label, s["sample_id"]

    def _load_normalized(self, path):
        if path not in MATCHED_IMAGE_CACHE:
            image = Image.open(path).convert("RGB")
            image = TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            MATCHED_IMAGE_CACHE[path] = match_to_reference(image)
        return MATCHED_IMAGE_CACHE[path].copy()

# ── DataLoaders ──
train_ds = StereoDataset(train_samples, augment=True)
val_ds   = StereoDataset(val_samples,   augment=False)
test_ds  = StereoDataset(test_samples,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

# ── Model: EfficientNet-B0 adapted for 9-channel input ──
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace first conv: 3 → 9 channels, repeat pretrained weights / 3
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(9, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         dilation=old_conv.dilation,
                         groups=old_conv.groups,
                         padding_mode=old_conv.padding_mode,
                         bias=old_conv.bias is not None)
    with torch.no_grad():
        new_conv.weight.data = old_conv.weight.data.repeat(1, 3, 1, 1) / 3.0
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
    model.features[0][0] = new_conv

    # Binary classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model, new_conv

model, new_conv = build_model()
model = model.to(device)

# ── Layer-wise LR: stem 10× backbone ──
stem_params     = list(new_conv.parameters())
backbone_params = [p for p in model.parameters() if not any(p is sp for sp in stem_params)]
optimizer = torch.optim.AdamW([
    {"params": stem_params,     "lr": BASE_LR},
    {"params": backbone_params, "lr": BASE_LR / 10},
], weight_decay=1e-3)

# ── Scheduler: SGDR ──
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# ── Loss ──
pos_weight = torch.tensor([OVERKILL_BUDGET / MISS_BUDGET]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ═══════════════════════════════════════════════
#  PROBE (≥5 epochs for 9-channel warm-up)
# ═══════════════════════════════════════════════
def run_probe():
    probe_model = model  # use same model
    probe_opt   = torch.optim.AdamW([
        {"params": stem_params, "lr": BASE_LR},
        {"params": backbone_params, "lr": BASE_LR / 10},
    ], weight_decay=1e-3)
    probe_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        probe_opt, T_0=5, T_mult=2, eta_min=1e-6
    )

    ng_recalls = []
    overkills  = []
    g_probs    = []
    ng_probs   = []

    for ep in range(1, PROBE_EPOCHS + 1):
        probe_model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.float().to(device)
            probe_opt.zero_grad()
            loss = criterion(probe_model(x).squeeze(1), y)
            loss.backward()
            probe_opt.step()
            total_loss += loss.item()
        probe_sched.step()

        # Eval at threshold 0.5
        probe_model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = probe_model(x.to(device)).squeeze(1)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(y.tolist())
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        preds = (all_probs >= 0.5).astype(int)
        tp = int(((preds == 1) & (all_labels == 1)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        tn = int(((preds == 0) & (all_labels == 0)).sum())

        recall   = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        overkill = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        g_mean   = all_probs[all_labels == 0].mean() if (all_labels == 0).any() else 0.0
        ng_mean  = all_probs[all_labels == 1].mean() if (all_labels == 1).any() else 0.0

        ng_recalls.append(recall)
        overkills.append(overkill)
        g_probs.append(g_mean)
        ng_probs.append(ng_mean)

    # Abort check: sustained across ALL probe epochs
    all_bad_overkill = all(o > 0.9 for o in overkills)
    all_bad_recall   = all(r < 0.2 for r in ng_recalls)
    all_flat         = all(abs(ng_probs[i] - g_probs[i]) < 0.01 for i in range(len(g_probs)))
    should_continue = not (all_bad_overkill or all_bad_recall or all_flat)

    reason_parts = []
    if all_bad_overkill:
        reason_parts.append(f"overkill>{0.9} all epochs")
    if all_bad_recall:
        reason_parts.append(f"recall<{0.2} all epochs")
    if all_flat:
        reason_parts.append("prob_gap<0.01 all epochs")
    reason = "; ".join(reason_parts) if reason_parts else "model shows learning signal"

    print("PROBE_METRICS: " + json.dumps({
        "ng_recall": ng_recalls[-1], "overkill_rate": overkills[-1],
        "G_prob_mean": g_probs[-1], "NG_prob_mean": ng_probs[-1],
        "should_continue": should_continue, "reason": reason
    }))
    return should_continue

should_cont = run_probe()
if not should_cont:
    print("PROBE WARNING: continuing to full training despite weak early probe signal")

# ═══════════════════════════════════════════════
#  FULL TRAINING
# ═══════════════════════════════════════════════
best_val_loss = float("inf")
patience_counter = 0
best_state = None

epoch_logs = []

try:
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_probs, all_val_labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = model(x.to(device)).squeeze(1)
                y_dev = y.float().to(device)
                val_loss += criterion(logits, y_dev).item()
                all_val_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_val_labels.extend(y.tolist())
        avg_val_loss = val_loss / len(val_loader)
        all_val_probs = np.array(all_val_probs)
        all_val_labels = np.array(all_val_labels)

        preds = (all_val_probs >= 0.5).astype(int)
        tp = int(((preds == 1) & (all_val_labels == 1)).sum())
        fn = int(((preds == 0) & (all_val_labels == 1)).sum())
        fp = int(((preds == 1) & (all_val_labels == 0)).sum())
        tn = int(((preds == 0) & (all_val_labels == 0)).sum())
        recall   = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        overkill = fp / (tn + fp) if (tn + fp) > 0 else 0.0

        log_entry = {"epoch": epoch, "train_loss": round(avg_train_loss, 6),
                     "val_loss": round(avg_val_loss, 6),
                     "val_ng_recall": round(recall, 6),
                     "val_overkill": round(overkill, 6)}
        epoch_logs.append(log_entry)
        print(f"EPOCH_LOG: {json.dumps(log_entry)}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

except Exception as e:
    print(f"Training error: {e}")
    sys.exit(1)

if best_state is not None:
    model.load_state_dict(best_state)

# ═══════════════════════════════════════════════
#  ISOTONIC CALIBRATION (fit on VAL)
# ═══════════════════════════════════════════════
model.eval()
raw_val_scores, val_labels_list = [], []
with torch.no_grad():
    for x, y, _ in val_loader:
        logits = model(x.to(device)).squeeze(1)
        raw_val_scores.extend(torch.sigmoid(logits).cpu().tolist())
        val_labels_list.extend(y.tolist())

raw_val_scores = np.array(raw_val_scores)
val_labels_arr = np.array(val_labels_list)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_val_scores, val_labels_arr)
cal_val_probs = iso.transform(raw_val_scores)

# Calibration stats
ng_mask_val = val_labels_arr == 1
g_mask_val  = val_labels_arr == 0
ng_prob_mean = cal_val_probs[ng_mask_val].mean() if ng_mask_val.any() else 0.0
ng_prob_std  = cal_val_probs[ng_mask_val].std()  if ng_mask_val.any() else 0.0
g_prob_mean  = cal_val_probs[g_mask_val].mean()  if g_mask_val.any() else 0.0
g_prob_std   = cal_val_probs[g_mask_val].std()   if g_mask_val.any() else 0.0

print(json.dumps({"CALIBRATION_STATS:": {
    "G_prob_mean": round(float(g_prob_mean), 6),
    "G_prob_std": round(float(g_prob_std), 6),
    "NG_prob_mean": round(float(ng_prob_mean), 6),
    "NG_prob_std": round(float(ng_prob_std), 6)
}}))

# ═══════════════════════════════════════════════
#  THRESHOLD SWEEP (Stage 0 → 1 → 2 on VAL)
# ═══════════════════════════════════════════════
all_candidates = []
for threshold in [round(i / 100.0, 2) for i in range(1, 100)]:
    preds = (cal_val_probs >= threshold).astype(int)
    tp = int(((preds == 1) & (val_labels_arr == 1)).sum())
    fn = int(((preds == 0) & (val_labels_arr == 1)).sum())
    fp = int(((preds == 1) & (val_labels_arr == 0)).sum())
    tn = int(((preds == 0) & (val_labels_arr == 0)).sum())
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    overkill_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    all_candidates.append((threshold, miss_rate, overkill_rate, fp, recall, accuracy))

# Stage 0: FP <= FP_MAX
survivors = [c for c in all_candidates if c[3] <= FP_MAX]

if survivors:
    # Stage 1: minimise miss_rate
    min_miss = min(c[1] for c in survivors)
    stage1 = [c for c in survivors if c[1] == min_miss]
    # Stage 2: lowest overkill_rate
    best = min(stage1, key=lambda c: c[2])
else:
    best = min(all_candidates, key=lambda c: (c[3], c[1]))

best_threshold, best_miss, best_overkill, best_fp, best_recall, best_acc = best

# ── Degenerate threshold guard ──
if best_threshold <= 0.15 and best_overkill > 0.50:
    print(f"DEGENERATE_THRESHOLD_WARNING: threshold={best_threshold:.4f} overkill={best_overkill:.4f} — near-all-NG operating point")

# Print threshold curve
curve_entries = []
for t, miss, ovk, fp, rec, acc in all_candidates:
    curve_entries.append({"t": t, "recall": round(rec, 6), "overkill": round(ovk, 6),
                          "miss_rate": round(miss, 6), "accuracy": round(acc, 6)})
print(f"THRESHOLD_CURVE: {json.dumps(curve_entries)}")

# ═══════════════════════════════════════════════
#  TEST EVALUATION (calibrated, best threshold)
# ═══════════════════════════════════════════════
raw_test_scores, test_labels, test_ids = [], [], []
inference_start = time.time()
with torch.no_grad():
    for x, y, sid in test_loader:
        logits = model(x.to(device)).squeeze(1)
        raw_test_scores.extend(torch.sigmoid(logits).cpu().tolist())
        test_labels.extend(y.tolist())
        test_ids.extend(sid)
inference_time = time.time() - inference_start

raw_test_scores = np.array(raw_test_scores)
test_labels_arr = np.array(test_labels)
cal_test_probs = iso.transform(raw_test_scores)

test_preds = (cal_test_probs >= best_threshold).astype(int)
tp = int(((test_preds == 1) & (test_labels_arr == 1)).sum())
fn = int(((test_preds == 0) & (test_labels_arr == 1)).sum())
fp = int(((test_preds == 1) & (test_labels_arr == 0)).sum())
tn = int(((test_preds == 0) & (test_labels_arr == 0)).sum())

ng_count = int((test_labels_arr == 1).sum())
g_count  = int((test_labels_arr == 0).sum())

miss_rate     = fn / (tp + fn) if (tp + fn) > 0 else 0.0
overkill_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
accuracy      = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
ng_recall     = tp / (tp + fn) if (tp + fn) > 0 else 1.0
precision     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1            = 2 * precision * ng_recall / (precision + ng_recall) if (precision + ng_recall) > 0 else 0.0
avg_latency_ms = (inference_time / len(test_samples)) * 1000 if len(test_samples) > 0 else 0.0

# roc_auc
if len(np.unique(test_labels_arr)) > 1:
    roc_auc = roc_auc_score(test_labels_arr, cal_test_probs)
else:
    roc_auc = 0.0

# prob_gap
ng_prob_test_mean = cal_test_probs[test_labels_arr == 1].mean() if (test_labels_arr == 1).any() else 0.0
g_prob_test_mean  = cal_test_probs[test_labels_arr == 0].mean() if (test_labels_arr == 0).any() else 0.0
prob_gap = ng_prob_test_mean - g_prob_test_mean

metrics = {
    "accuracy": round(accuracy, 6),
    "ng_recall": round(ng_recall, 6),
    "miss_rate": round(miss_rate, 6),
    "overkill_rate": round(overkill_rate, 6),
    "f1": round(f1, 6),
    "avg_latency_ms": round(avg_latency_ms, 4),
    "threshold": round(float(best_threshold), 6),
    "ng_count": ng_count,
    "g_count": g_count,
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    "roc_auc": round(roc_auc, 6),
    "prob_gap": round(float(prob_gap), 6)
}
print(f"METRICS: {json.dumps(metrics)}")

# Per-sample predictions (guarded for dry-run)
if not DRY_RUN:
    predictions = []
    for i in range(len(test_samples)):
        predictions.append({
            "sample_id": test_ids[i],
            "true_label": "NG" if test_labels[i] == 1 else "G",
            "predicted_label": "NG" if int(test_preds[i]) == 1 else "G",
            "ng_probability": round(float(cal_test_probs[i]), 6),
            "threshold": round(float(best_threshold), 6)
        })
    print(f"PREDICTIONS: {json.dumps(predictions)}")
