"""Mandatory AOI training script template for generated candidate models."""


def get_script_template(data_split_path: str, input_modality: str = 'stereo') -> str:
    """Return a complete candidate training script with only architecture left blank."""
    template = '''#!/usr/bin/env python3
"""AOI candidate script — architecture: __ARCHITECTURE_NAME__"""
import os, json, random, time, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from PIL import Image

DRY_RUN         = os.getenv('DRY_RUN') == '1'
DRY_RUN_EPOCHS  = int(os.getenv('DRY_RUN_EPOCHS', '1'))
DRY_RUN_SAMPLES = int(os.getenv('DRY_RUN_SAMPLES', '10'))
_seed = int(os.environ.get('AOI_RANDOM_SEED', os.environ.get('SEED', '42')))
random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)
MISS_BUDGET     = 0.03
OVERKILL_BUDGET = 0.08
FP_MAX          = 2
epochs          = DRY_RUN_EPOCHS if DRY_RUN else 20
PATIENCE        = 3
BATCH_SIZE      = 8
IMAGE_SIZE      = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_SPLIT_PATH = __DATA_SPLIT_PATH__
with open(DATA_SPLIT_PATH) as f:
    data_split = json.load(f)
train_samples = data_split['train']
val_samples   = data_split['val']
test_samples  = data_split['test']
if DRY_RUN:
    train_samples = train_samples[:DRY_RUN_SAMPLES]
    val_samples   = val_samples[:DRY_RUN_SAMPLES]
    test_samples  = test_samples[:DRY_RUN_SAMPLES]

n_ng = sum(1 for s in train_samples if s['label'] == 'NG')
n_g  = sum(1 for s in train_samples if s['label'] == 'G')
pos_weight = torch.tensor([n_g / n_ng])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FEATURE_DIFF_CANDIDATE = False


def _normalize_group(tensor):
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class StereoDataset(Dataset):
    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        return Image.open(path).convert('RGB')

    def _transform_pair(self, img_l, img_r):
        img_l = TF.resize(img_l, [IMAGE_SIZE, IMAGE_SIZE])
        img_r = TF.resize(img_r, [IMAGE_SIZE, IMAGE_SIZE])
        if self.augment:
            if random.random() < 0.5:
                img_l = TF.hflip(img_l)
                img_r = TF.hflip(img_r)
            angle = random.uniform(-5.0, 5.0)
            img_l = TF.rotate(img_l, angle)
            img_r = TF.rotate(img_r, angle)
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            img_l = TF.adjust_brightness(img_l, brightness)
            img_r = TF.adjust_brightness(img_r, brightness)
            img_l = TF.adjust_contrast(img_l, contrast)
            img_r = TF.adjust_contrast(img_r, contrast)
        img_l = TF.to_tensor(img_l)
        img_r = TF.to_tensor(img_r)
        return img_l, img_r

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_l, img_r = self._transform_pair(
            self._load_rgb(sample['img_l']),
            self._load_rgb(sample['img_r']),
        )
        label = torch.tensor(1.0 if sample['label'] == 'NG' else 0.0)
        sample_id = sample.get('sample_id', str(idx))
        img_l = _normalize_group(img_l)
        img_r = _normalize_group(img_r)
        if FEATURE_DIFF_CANDIDATE:
            return img_l, img_r, label, sample_id
        diff = _normalize_group(torch.abs(img_l * IMAGENET_STD + IMAGENET_MEAN - (img_r * IMAGENET_STD + IMAGENET_MEAN)))
        image = torch.cat([img_l, img_r, diff], dim=0)
        return image, label, sample_id


train_loader = DataLoader(StereoDataset(train_samples, augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(StereoDataset(val_samples, augment=False), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(StereoDataset(test_samples, augment=False), batch_size=BATCH_SIZE, shuffle=False)

# <<< ARCHITECTURE BLOCK START >>>
# The LLM fills in: model class definition, build_model() function,
# optimizer, scheduler, and PROBE_EPOCHS constant.
# Requirements:
#   - build_model() returns the model on `device`
#   - Use /3 repeat trick for 9-channel CNN: new_conv.weight.data = old_conv.weight.data.repeat(1,3,1,1)/3.0
#   - For ViT backbones: set FEATURE_DIFF_CANDIDATE=True and use Siamese feature-diff (no patch embedding modification)
#   - optimizer and scheduler must be defined
#   - PROBE_EPOCHS = min(DRY_RUN_EPOCHS, 5) if DRY_RUN else 5  (9-channel CNN)
#   - PROBE_EPOCHS = min(DRY_RUN_EPOCHS, 3) if DRY_RUN else 3  (3-channel / FEATURE_DIFF)
# <<< ARCHITECTURE BLOCK END >>>
model = build_model()


def _reset_optimizer_scheduler():
    global optimizer, scheduler
    if 'build_optimizer' in globals():
        optimizer = build_optimizer(model)
    elif 'optimizer' not in globals() or optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    if 'build_scheduler' in globals():
        scheduler = build_scheduler(optimizer)
    elif 'scheduler' not in globals() or scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)


_reset_optimizer_scheduler()


def _unpack_batch(batch):
    if FEATURE_DIFF_CANDIDATE:
        img_l, img_r, labels, sample_ids = batch
        return (img_l.to(device), img_r.to(device)), labels.float().to(device), sample_ids
    images, labels, sample_ids = batch
    return images.to(device), labels.float().to(device), sample_ids


def _forward(inputs):
    if FEATURE_DIFF_CANDIDATE:
        img_l, img_r = inputs
        logits = model(img_l, img_r)
    else:
        logits = model(inputs)
    return logits.view(-1)


def _binary_counts(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return tp, tn, fp, fn


def _metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    ng_recall = tp / (tp + fn) if (tp + fn) else 1.0
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    overkill_rate = fp / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * ng_recall / (precision + ng_recall) if (precision + ng_recall) else 0.0
    return accuracy, ng_recall, miss_rate, overkill_rate, f1


def _evaluate_loss(loader, criterion):
    model.eval()
    total_loss = 0.0
    labels_all, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, _ = _unpack_batch(batch)
            logits = _forward(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.numel()
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    denom = max(1, len(labels_all))
    return total_loss / denom, np.array(labels_all), np.array(probs_all)


def _collect_logits(loader):
    model.eval()
    logits_all, labels_all, sample_ids_all = [], [], []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs, labels, sample_ids = _unpack_batch(batch)
            logits = _forward(inputs)
            logits_all.extend(logits.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            sample_ids_all.extend(list(sample_ids))
    elapsed = time.time() - start
    return np.array(logits_all), np.array(labels_all), sample_ids_all, elapsed


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
probe_epoch_metrics = []
for probe_epoch in range(PROBE_EPOCHS):
    model.train()
    for batch in train_loader:
        inputs, labels, _ = _unpack_batch(batch)
        optimizer.zero_grad()
        logits = _forward(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    _, probe_labels, probe_probs = _evaluate_loss(val_loader, criterion)
    tp, tn, fp, fn = _binary_counts(probe_labels, probe_probs, 0.5)
    _, ng_recall, _, overkill_rate, _ = _metrics_from_counts(tp, tn, fp, fn)
    g_probs = probe_probs[probe_labels == 0]
    ng_probs = probe_probs[probe_labels == 1]
    g_mean = float(g_probs.mean()) if len(g_probs) else 0.0
    ng_mean = float(ng_probs.mean()) if len(ng_probs) else 0.0
    probe_epoch_metrics.append({
        'ng_recall': float(ng_recall),
        'overkill_rate': float(overkill_rate),
        'G_prob_mean': g_mean,
        'NG_prob_mean': ng_mean,
        'probability_gap': float(ng_mean - g_mean),
    })

if probe_epoch_metrics:
    ng_recall = float(np.mean([m['ng_recall'] for m in probe_epoch_metrics]))
    overkill_rate = float(np.mean([m['overkill_rate'] for m in probe_epoch_metrics]))
    g_mean = float(np.mean([m['G_prob_mean'] for m in probe_epoch_metrics]))
    ng_mean = float(np.mean([m['NG_prob_mean'] for m in probe_epoch_metrics]))
else:
    ng_recall, overkill_rate, g_mean, ng_mean = 0.0, 0.0, 0.0, 0.0

should_continue = True
reason = 'OK'
if probe_epoch_metrics and all(m['overkill_rate'] > 0.90 for m in probe_epoch_metrics):
    should_continue = False
    reason = 'Catastrophic overkill'
elif probe_epoch_metrics and all(m['ng_recall'] < 0.05 for m in probe_epoch_metrics):
    should_continue = False
    reason = 'Recall collapse'
elif probe_epoch_metrics and all(abs(m['NG_prob_mean'] - m['G_prob_mean']) < 0.01 for m in probe_epoch_metrics):
    should_continue = False
    reason = 'No G/NG separation'

print('PROBE_METRICS:', json.dumps({
    'ng_recall': ng_recall,
    'overkill_rate': overkill_rate,
    'G_prob_mean': g_mean,
    'NG_prob_mean': ng_mean,
    'should_continue': should_continue,
    'reason': reason,
}))

if not should_continue:
    dummy_metrics = {
        'accuracy': 0.0, 'ng_recall': 0.0, 'miss_rate': 0.0, 'overkill_rate': 0.0,
        'f1': 0.0, 'avg_latency_ms': 0.0, 'threshold': 0.5, 'ng_count': 0,
        'g_count': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'roc_auc': 0.0, 'prob_gap': 0.0,
    }
    print('METRICS:', json.dumps(dummy_metrics))
    sys.exit(0)

model = build_model()
_reset_optimizer_scheduler()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
best_val_loss = math.inf
best_state = None
patience_left = PATIENCE

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    train_count = 0
    for batch in train_loader:
        inputs, labels, _ = _unpack_batch(batch)
        optimizer.zero_grad()
        logits = _forward(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.numel()
        train_count += labels.numel()
    val_loss, val_labels, val_probs = _evaluate_loss(val_loader, criterion)
    tp, tn, fp, fn = _binary_counts(val_labels, val_probs, 0.5)
    _, val_ng_recall, _, val_overkill, _ = _metrics_from_counts(tp, tn, fp, fn)
    print('EPOCH_LOG:', json.dumps({
        'epoch': epoch,
        'train_loss': train_loss / max(1, train_count),
        'val_loss': val_loss,
        'val_ng_recall': val_ng_recall,
        'val_overkill': val_overkill,
    }))
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            break

if best_state is not None:
    model.load_state_dict(best_state)

val_logits, val_labels, val_sample_ids, _ = _collect_logits(val_loader)
test_logits, test_labels, test_sample_ids, test_elapsed = _collect_logits(test_loader)
val_raw_probs = 1.0 / (1.0 + np.exp(-val_logits))
test_raw_probs = 1.0 / (1.0 + np.exp(-test_logits))
iso = IsotonicRegression(out_of_bounds='clip')
if len(np.unique(val_labels)) >= 2:
    iso.fit(val_raw_probs, val_labels)
    val_probs = iso.transform(val_raw_probs)
    test_probs = iso.transform(test_raw_probs)
else:
    val_probs = val_raw_probs
    test_probs = test_raw_probs

val_g_probs = val_probs[val_labels == 0]
val_ng_probs = val_probs[val_labels == 1]
print('CALIBRATION_STATS:', json.dumps({
    'G_prob_mean': float(val_g_probs.mean()) if len(val_g_probs) else 0.0,
    'G_prob_std': float(val_g_probs.std()) if len(val_g_probs) else 0.0,
    'NG_prob_mean': float(val_ng_probs.mean()) if len(val_ng_probs) else 0.0,
    'NG_prob_std': float(val_ng_probs.std()) if len(val_ng_probs) else 0.0,
}))

all_candidates = []
threshold_curve = []
for i in range(1, 100):
    threshold = round(i / 100.0, 2)
    tp, tn, fp, fn = _binary_counts(val_labels, val_probs, threshold)
    accuracy, recall, miss_rate, overkill, _ = _metrics_from_counts(tp, tn, fp, fn)
    all_candidates.append((threshold, miss_rate, overkill, fp))
    threshold_curve.append({
        't': threshold,
        'recall': recall,
        'overkill': overkill,
        'miss_rate': miss_rate,
        'accuracy': accuracy,
    })

survivors = [c for c in all_candidates if c[3] <= FP_MAX]
if survivors:
    min_miss = min(c[1] for c in survivors)
    stage1 = [c for c in survivors if c[1] == min_miss]
    best_threshold = min(stage1, key=lambda c: c[2])[0]
else:
    best_threshold = min(all_candidates, key=lambda c: (c[3], c[1]))[0] if all_candidates else 0.5
print('THRESHOLD_CURVE:', json.dumps(threshold_curve))

tp, tn, fp, fn = _binary_counts(test_labels, test_probs, best_threshold)
accuracy, ng_recall, miss_rate, overkill_rate, f1 = _metrics_from_counts(tp, tn, fp, fn)
try:
    roc_auc = float(roc_auc_score(test_labels, test_probs)) if len(np.unique(test_labels)) >= 2 else 0.0
except ValueError:
    roc_auc = 0.0
test_g_probs = test_probs[test_labels == 0]
test_ng_probs = test_probs[test_labels == 1]
prob_gap = (float(test_ng_probs.mean()) if len(test_ng_probs) else 0.0) - (float(test_g_probs.mean()) if len(test_g_probs) else 0.0)
pred_labels = np.where(test_probs >= best_threshold, 'NG', 'G')
true_labels = np.where(test_labels == 1, 'NG', 'G')
avg_latency_ms = (test_elapsed / max(1, len(test_samples))) * 1000.0

metrics = {
    'accuracy': accuracy,
    'ng_recall': ng_recall,
    'miss_rate': miss_rate,
    'overkill_rate': overkill_rate,
    'f1': f1,
    'avg_latency_ms': avg_latency_ms,
    'threshold': best_threshold,
    'ng_count': int((test_labels == 1).sum()),
    'g_count': int((test_labels == 0).sum()),
    'tp': tp,
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'roc_auc': roc_auc,
    'prob_gap': prob_gap,
}
print('METRICS:', json.dumps(metrics))

predictions = []
fp_samples, fn_samples = [], []
for sample_id, true_label, predicted_label, prob in zip(test_sample_ids, true_labels, pred_labels, test_probs):
    row = {
        'sample_id': sample_id,
        'true_label': str(true_label),
        'predicted_label': str(predicted_label),
        'ng_probability': float(prob),
        'threshold': best_threshold,
    }
    predictions.append(row)
    if true_label == 'G' and predicted_label == 'NG':
        fp_samples.append({k: row[k] for k in ('sample_id', 'true_label', 'predicted_label', 'ng_probability')})
    if true_label == 'NG' and predicted_label == 'G':
        fn_samples.append({k: row[k] for k in ('sample_id', 'true_label', 'predicted_label', 'ng_probability')})
print('PREDICTIONS:', json.dumps(predictions))
print('ERROR_ANALYSIS:', json.dumps({'fp_samples': fp_samples, 'fn_samples': fn_samples}))
'''
    return template.replace('__DATA_SPLIT_PATH__', repr(data_split_path))
