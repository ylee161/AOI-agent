"""Mandatory AOI training script template for generated candidate models.

Typed hook system
-----------------
Generated candidate scripts no longer rely on the LLM free-writing the
"architecture block" (the historical #1 source of broken scripts). Instead the
block is composed from *typed switches* that each resolve to a pre-written,
known-good code path:

  * LOSS_MODE         — bce | focal | logit_adjust      (universal)
  * OPTIMIZER_MODE    — adamw | sgd | adam              (universal)
  * VIEW_FUSION_MODE  — concat_diff | siamese_feature_diff (AOI / stereo only)
  * GROUP_ROBUST_MODE — group_dro | group_balanced | off  (group-aware only)

Two of those switches are *capability-gated* so the repo stays a clean,
dataset-agnostic pipeline anyone can clone:

  * VIEW_FUSION_MODE is only offered when ``DatasetCapabilities.HAS_STEREO_VIEWS``
    is True; otherwise the block falls back to a single-image (3-channel) model
    and every stereo option stays dormant.
  * GROUP_ROBUST_MODE is only offered when ``DatasetCapabilities.GROUP_COLUMN``
    is set; otherwise group-robust selection disables and model selection falls
    back to plain validation.

The switch *mechanism* is fully general; only the AOI-specific menu entries are
gated. ``build_architecture_block(cfg, caps)`` returns the resolved block, and
``get_script_template(..., hook_config=, capabilities=)`` renders a full script
with that block already inserted — no placeholder, no free-writing.
"""

from dataclasses import dataclass
from typing import Optional


# ── Dataset capability descriptor ────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetCapabilities:
    """What the *dataset* can support. Drives capability gating of AOI options.

    HAS_STEREO_VIEWS — True when each sample carries a stereo image pair
        (``img_l`` + ``img_r``). When False, view-fusion options are not offered
        and the template falls back to a single-image model.
    GROUP_COLUMN — name of the per-sample group key (e.g. ``"board"``) used for
        group-robust training/selection, or None to disable group robustness.
    """

    HAS_STEREO_VIEWS: bool = True
    GROUP_COLUMN: Optional[str] = None


# ── Typed hook configuration ─────────────────────────────────────────────────
@dataclass(frozen=True)
class HookConfig:
    """The chosen switch positions. Universal switches always apply; AOI switches
    are honoured only when their capability flag (above) is on."""

    LOSS_MODE: str = "bce"
    OPTIMIZER_MODE: str = "adamw"
    VIEW_FUSION_MODE: str = "concat_diff"
    GROUP_ROBUST_MODE: str = "off"


# ── Resolved view-fusion descriptor ──────────────────────────────────────────
@dataclass(frozen=True)
class ResolvedView:
    mode: str
    in_channels: int
    feature_diff: bool
    model_code: str


# ── Universal menus ──────────────────────────────────────────────────────────
# Each entry is a self-contained code snippet inserted verbatim into the
# architecture block. They reference only names the template already defines
# (``device``, ``pos_weight``, ``nn``, ``torch``, ``optim``, ``os``).

LOSS_MENU = {
    "bce": '''
def build_criterion():
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
''',
    "focal": '''
class _BinaryFocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017). pos_weight keeps the BCE class
    re-weighting; gamma down-weights easy examples."""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight.to(logits.device))
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


def build_criterion():
    return _BinaryFocalLoss(pos_weight=pos_weight.to(device)).to(device)
''',
    "logit_adjust": '''
class _LogitAdjustedBCE(nn.Module):
    """Logit-adjusted BCE (Menon et al. 2021): shift logits by tau*log-prior so
    the boundary accounts for class imbalance. pos_weight = n_g/n_ng encodes the
    prior odds, so the NG-logit shift is -tau*log(pos_weight)."""
    def __init__(self, pos_weight=None, tau=1.0):
        super().__init__()
        self.tau = tau
        pw = pos_weight if pos_weight is not None else torch.tensor(1.0)
        self.register_buffer('logit_shift', -tau * torch.log(pw.clamp_min(1e-6)))

    def forward(self, logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(
            logits + self.logit_shift.to(logits.device), targets)


def build_criterion():
    return _LogitAdjustedBCE(pos_weight=pos_weight.to(device)).to(device)
''',
}

OPTIMIZER_MENU = {
    "adamw": '''
def build_optimizer(model):
    return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)


def build_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
''',
    "sgd": '''
def build_optimizer(model):
    return optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)


def build_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
''',
    "adam": '''
def build_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)


def build_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
''',
}

# ── Group-robust menu (gated by GROUP_COLUMN) ────────────────────────────────
# These are deliberately clean STUBS: the wired hook + capability flag, with the
# full worst-group optimisation math left for a follow-up. When inactive they
# fall back to the global objective so selection always works.
_GROUP_ROBUST_OFF = '''
GROUP_ROBUST_ACTIVE = False
GROUP_COLUMN = None


def group_robust_objective(val_loss, val_labels=None, val_probs=None, val_groups=None):
    # GROUP_COLUMN is unset -> plain validation selection.
    return val_loss
'''

_GROUP_ROBUST_TEMPLATE = '''
GROUP_ROBUST_ACTIVE = True
GROUP_COLUMN = {group_column!r}
GROUP_ROBUST_MODE = {mode!r}


def group_robust_objective(val_loss, val_labels=None, val_probs=None, val_groups=None):
    """STUB ({mode}): worst-group model-selection objective. The per-group
    optimisation math is intentionally not implemented here yet — this is the
    wired hook the selection loop calls when GROUP_COLUMN is set. Until filled
    in it falls back to the global validation loss so selection still works."""
    return val_loss
'''

GROUP_ROBUST_MODES = ("group_dro", "group_balanced")

# ── View-fusion menu (gated by HAS_STEREO_VIEWS) ─────────────────────────────
STEREO_VIEW_FUSION_MODES = ("concat_diff", "siamese_feature_diff")

# Default fallback backbone = EfficientNet-B1. This is the backbone validated on
# SUP046 (board-stereo Arm A and per-side Arm B both used B1). It is still a
# small, fast, pretrained CNN
# suited to the tiny dataset, but stronger than ResNet18 — so the deterministic
# typed-hook path clears the in-distribution board baseline on its own, without
# needing the LLM to author an EfficientNet block. The stem-conv (features[0][0])
# and classifier (classifier[1]) accessors below are EfficientNet's equivalents
# of ResNet's conv1 / fc.
_PRETRAINED_SETUP = '''
import torchvision
_PRETRAINED = os.getenv('AOI_PRETRAINED', '1') == '1'
try:
    _BACKBONE_WEIGHTS = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1 if _PRETRAINED else None
except Exception:
    _BACKBONE_WEIGHTS = None
'''

_CONCAT_MODEL_CODE = '''
def build_model():
    base = torchvision.models.efficientnet_b1(weights=_BACKBONE_WEIGHTS)
    if IN_CHANNELS != 3:
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(IN_CHANNELS, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                             padding=old_conv.padding, bias=old_conv.bias is not None)
        if _BACKBONE_WEIGHTS is not None and IN_CHANNELS % 3 == 0:
            # /N repeat trick: tile the pretrained 3-ch filters across the N views.
            with torch.no_grad():
                reps = IN_CHANNELS // 3
                new_conv.weight.copy_(old_conv.weight.repeat(1, reps, 1, 1) / reps)
        base.features[0][0] = new_conv
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, 1)
    return base.to(device)
'''

_SIAMESE_MODEL_CODE = '''
class SiameseFeatureDiff(nn.Module):
    """Weight-shared EfficientNet-B1 over each view; classifies the
    |feature_l - feature_r| difference. Pairs with FEATURE_DIFF_CANDIDATE=True
    (forward takes img_l, img_r)."""
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.efficientnet_b1(weights=_BACKBONE_WEIGHTS)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, img_l, img_r):
        f_l = self.backbone(img_l)
        f_r = self.backbone(img_r)
        return self.head(torch.abs(f_l - f_r))


def build_model():
    return SiameseFeatureDiff().to(device)
'''

# Resolved view descriptors. ``single`` is the capability-off fallback.
_VIEW_FUSION_RESOLVED = {
    "concat_diff": ResolvedView("concat_diff", 9, False, _CONCAT_MODEL_CODE),
    "siamese_feature_diff": ResolvedView("siamese_feature_diff", 3, True, _SIAMESE_MODEL_CODE),
    "single": ResolvedView("single", 3, False, _CONCAT_MODEL_CODE),
}


# ── Offered-menu helpers (capability gating) ─────────────────────────────────
def offered_loss_modes() -> list:
    return list(LOSS_MENU)


def offered_optimizer_modes() -> list:
    return list(OPTIMIZER_MENU)


def offered_view_fusion_modes(caps: DatasetCapabilities) -> list:
    """Stereo options are offered only when the dataset has stereo views."""
    return list(STEREO_VIEW_FUSION_MODES) if caps.HAS_STEREO_VIEWS else ["single"]


def offered_group_robust_modes(caps: DatasetCapabilities) -> list:
    """Group-robust modes are offered only when a group column exists."""
    return list(GROUP_ROBUST_MODES) if caps.GROUP_COLUMN is not None else ["off"]


# ── Resolvers ────────────────────────────────────────────────────────────────
def resolve_loss(cfg: HookConfig) -> str:
    """LOSS_MODE is universal; unknown modes fall back to bce."""
    return LOSS_MENU.get(cfg.LOSS_MODE, LOSS_MENU["bce"])


def resolve_optimizer(cfg: HookConfig) -> str:
    """OPTIMIZER_MODE is universal; unknown modes fall back to adamw."""
    return OPTIMIZER_MENU.get(cfg.OPTIMIZER_MODE, OPTIMIZER_MENU["adamw"])


def resolve_view_fusion(cfg: HookConfig, caps: DatasetCapabilities) -> ResolvedView:
    """Capability-gated. When the dataset has no stereo views the requested mode
    is ignored and we fall back to a single-image model (AOI options dormant)."""
    if not caps.HAS_STEREO_VIEWS:
        return _VIEW_FUSION_RESOLVED["single"]
    mode = cfg.VIEW_FUSION_MODE if cfg.VIEW_FUSION_MODE in STEREO_VIEW_FUSION_MODES else "concat_diff"
    return _VIEW_FUSION_RESOLVED[mode]


def resolve_group_robust(cfg: HookConfig, caps: DatasetCapabilities) -> str:
    """Capability-gated. With no GROUP_COLUMN, group robustness disables and
    selection falls back to plain validation."""
    if caps.GROUP_COLUMN is None:
        return _GROUP_ROBUST_OFF
    mode = cfg.GROUP_ROBUST_MODE if cfg.GROUP_ROBUST_MODE in GROUP_ROBUST_MODES else "group_dro"
    return _GROUP_ROBUST_TEMPLATE.format(group_column=caps.GROUP_COLUMN, mode=mode)


def build_architecture_block(cfg: HookConfig, caps: DatasetCapabilities) -> str:
    """Compose the full architecture block from the typed switches — the
    pre-written replacement for the historical LLM free-write block. Returns a
    module-level code string defining IN_CHANNELS, FEATURE_DIFF_CANDIDATE,
    build_model, build_optimizer/build_scheduler, build_criterion, the
    group-robust hook, and PROBE_EPOCHS."""
    view = resolve_view_fusion(cfg, caps)
    probe = 8 if view.feature_diff else 5
    group_mode = cfg.GROUP_ROBUST_MODE if caps.GROUP_COLUMN is not None else "off"
    header = (
        f"# <<< TYPED HOOK BLOCK (auto-generated from typed switches — no free-writing) >>>\n"
        f"# LOSS_MODE={cfg.LOSS_MODE!r}  OPTIMIZER_MODE={cfg.OPTIMIZER_MODE!r}  "
        f"VIEW_FUSION_MODE={view.mode!r}  GROUP_ROBUST_MODE={group_mode!r}"
    )
    parts = [
        header,
        _PRETRAINED_SETUP.strip("\n"),
        f"IN_CHANNELS = {view.in_channels}",
        f"FEATURE_DIFF_CANDIDATE = {view.feature_diff}",
        view.model_code.strip("\n"),
        resolve_optimizer(cfg).strip("\n"),
        resolve_loss(cfg).strip("\n"),
        resolve_group_robust(cfg, caps).strip("\n"),
        f"PROBE_EPOCHS = min(DRY_RUN_EPOCHS, {probe}) if DRY_RUN else {probe}",
        "# <<< END TYPED HOOK BLOCK >>>",
    ]
    return "\n\n\n".join(parts)


# The original free-write block, used when no hook_config is supplied (kept for
# backward compatibility / human-authored candidates).
_FREEWRITE_ARCHITECTURE_BLOCK = '''# <<< ARCHITECTURE BLOCK START >>>
# The LLM fills in: model class definition, build_model() function,
# optimizer, scheduler, and PROBE_EPOCHS constant.
# Requirements:
#   - build_model() returns the model on `device`
#   - Use /3 repeat trick for 9-channel CNN: new_conv.weight.data = old_conv.weight.data.repeat(1,3,1,1)/3.0
#   - For ViT backbones: set FEATURE_DIFF_CANDIDATE=True and use Siamese feature-diff (no patch embedding modification)
#   - optimizer and scheduler must be defined
#   - PROBE_EPOCHS = min(DRY_RUN_EPOCHS, 5) if DRY_RUN else 5  (9-channel CNN)
#   - PROBE_EPOCHS = min(DRY_RUN_EPOCHS, 8) if DRY_RUN else 8  (FEATURE_DIFF / ViT / DINOv2 — frozen backbone needs more warm-up)
#   - For DINOv2/ViT: MUST partially unfreeze (e.g. unfreeze last 6 transformer blocks) from epoch 0 — a fully frozen backbone + random head will never develop prob_gap in any number of probe epochs
# <<< ARCHITECTURE BLOCK END >>>'''


def get_script_template(
    data_split_path: str,
    input_modality: str = 'stereo',
    hook_config: Optional[HookConfig] = None,
    capabilities: Optional[DatasetCapabilities] = None,
    architecture_block: Optional[str] = None,
    label_granularity: str = 'board',
) -> str:
    """Return a complete candidate training script.

    When ``hook_config`` is None the architecture block is left as the historical
    free-write placeholder (LLM fills it). When ``hook_config`` is supplied the
    block is composed from the typed switches via ``build_architecture_block`` —
    a complete, pre-written model + optimizer + loss + group hook with no
    free-writing. ``architecture_block`` (highest precedence) inserts the given
    block text verbatim — used by the baseline coder's render-from-block path,
    where the LLM authors ONLY the block and the rest of the script is this
    canonical template byte-for-byte. ``capabilities`` defaults to the AOI stereo
    descriptor; pass an explicit one to gate AOI options. Single-image datasets
    (``HAS_STEREO_VIEWS=False``) are not rendered by this stereo template — use
    ``build_architecture_block`` directly against a single-image template.

    ``label_granularity`` selects the data/eval path baked into the rendered
    script (the runtime path is also re-derived from the split metadata, so the
    two always agree). "per_side" renders a single-image (3-channel) model — each
    side is its own mono sample and the two sides are pooled back to a board score
    at eval. The body branches at runtime on ``data_split['metadata']
    ['label_granularity']``; here we only need to ensure the architecture block is
    3-channel when per_side, which we do by gating the view-fusion menu to the
    single-image fallback regardless of the requested stereo fusion mode.
    """
    per_side = label_granularity == 'per_side'
    if architecture_block is not None:
        pass  # caller-supplied block wins; resolved below
    elif hook_config is not None:
        caps = capabilities if capabilities is not None else DatasetCapabilities()
        if per_side:
            # per_side feeds single 3-channel mono images, so the model must be
            # single-image. Reuse the existing capability gate (HAS_STEREO_VIEWS
            # off -> "single" 3-ch view) rather than special-casing the block.
            caps = DatasetCapabilities(HAS_STEREO_VIEWS=False, GROUP_COLUMN=caps.GROUP_COLUMN)
        elif not caps.HAS_STEREO_VIEWS:
            raise ValueError(
                "get_script_template renders the stereo AOI template, which requires "
                "HAS_STEREO_VIEWS=True. For single-image datasets, compose a script "
                "with build_architecture_block(cfg, caps) against a single-image template."
            )
        architecture_block = build_architecture_block(hook_config, caps)
    else:
        architecture_block = _FREEWRITE_ARCHITECTURE_BLOCK
    template = '''#!/usr/bin/env python3
"""AOI candidate script — architecture: __ARCHITECTURE_NAME__"""
import os, json, random, time, math, sys, hashlib
from pathlib import Path
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
VAL_NG_RECALL_TARGET = float(os.getenv('VAL_NG_RECALL_TARGET', '0.90'))
epochs          = DRY_RUN_EPOCHS if DRY_RUN else 20
PATIENCE        = 3
BATCH_SIZE      = 8
# Override with AOI_IMAGE_SIZE so refinement variants can sweep input resolution
# without rewriting the dataset code (the on-disk cache below is keyed by size).
IMAGE_SIZE      = int(os.getenv('AOI_IMAGE_SIZE', '224'))


def _select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


device = _select_device()
print(f'Using device: {device}')
NUM_WORKERS = 2 if device.type == 'cuda' else 0

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

# Label granularity (board vs per_side). per_side = each side is its own 3-channel
# mono sample; the two sides are pooled back to a board score (max) at eval so the
# headline metric stays the board-level G/NG AUC. board = the original 9-channel
# stereo path. HONEST NOTE: per_side attacks the under-learning/clean-side-dilution
# axis only — it does NOT move the cross-lot generalisation wall.
LABEL_GRANULARITY = data_split.get('metadata', {}).get('label_granularity', 'board')
PER_SIDE = LABEL_GRANULARITY == 'per_side'
_BOARD_ID_OF = {}
_BOARD_LABEL_OF = {}
if PER_SIDE:
    for _s in train_samples + val_samples + test_samples:
        _BOARD_ID_OF[_s['sample_id']] = _s['board_id']
        _BOARD_LABEL_OF[_s['board_id']] = 1 if _s['board_label'] == 'NG' else 0

n_ng = sum(1 for s in train_samples if s['label'] == 'NG')
n_g  = sum(1 for s in train_samples if s['label'] == 'G')
pos_weight = torch.tensor([n_g / n_ng])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FEATURE_DIFF_CANDIDATE = False


def _normalize_group(tensor):
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


# ── Cached image loading ─────────────────────────────────────────────────────
# The raw dataset is 4K PNGs (~14 MB each); decoding + resizing them on every
# epoch dominates wall-clock training time. Resize once per (image, IMAGE_SIZE)
# and store the small uint8 array on disk; subsequent epochs/seeds/scripts all
# reuse the cache. Writes are atomic (tmp + os.replace) so parallel runs are safe.
IMG_CACHE_DIR = Path(DATA_SPLIT_PATH).parent / 'img_cache' / str(IMAGE_SIZE)
IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_rgb_cached(path):
    key = hashlib.md5(path.encode()).hexdigest()
    cache_file = IMG_CACHE_DIR / (key + '.npy')
    if cache_file.is_file():
        try:
            return Image.fromarray(np.load(cache_file))
        except Exception:
            pass  # corrupt cache entry — rebuild it below
    image = Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    tmp_file = cache_file.with_name(cache_file.name + f'.{os.getpid()}.tmp.npy')
    try:
        np.save(tmp_file, arr)
        os.replace(tmp_file, cache_file)
    except OSError:
        pass  # cache write failure is non-fatal
    return image


class StereoDataset(Dataset):
    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        return _load_rgb_cached(path)

    def _transform_pair(self, img_l, img_r):
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


class PerSideDataset(Dataset):
    """per_side mode: one 3-channel mono image per side, labelled by its OWN
    defect count. Returns (image, label, sample_id) — same signature as the
    non-feature-diff StereoDataset path, so _unpack_batch/_forward are unchanged.
    Only hflip augmentation (the lone label-preserving geometric op for a single
    AOI view; matches the validated A/B recipe)."""
    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = _load_rgb_cached(sample['img'])
        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
        image = _normalize_group(TF.to_tensor(img))
        label = torch.tensor(1.0 if sample['label'] == 'NG' else 0.0)
        sample_id = sample.get('sample_id', str(idx))
        return image, label, sample_id


_DatasetClass = PerSideDataset if PER_SIDE else StereoDataset
train_loader = DataLoader(_DatasetClass(train_samples, augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(_DatasetClass(val_samples, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(_DatasetClass(test_samples, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

__ARCHITECTURE_BLOCK__
model = build_model()

if os.getenv('AOI_PREFETCH_ONLY') == '1':
    # Build-only pass: the harness runs this once with a generous timeout so any
    # pretrained-weight download lands in the local cache BEFORE the time-capped
    # smoke run. Must exit before any training begins.
    print('PREFETCH_OK')
    sys.exit(0)


def _resolve_criterion():
    # Typed hook block defines build_criterion(); the free-write block may omit
    # it, in which case we fall back to the historical pos_weighted BCE.
    if 'build_criterion' in globals():
        return build_criterion()
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


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
    labels_all, probs_all, sample_ids_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, sample_ids = _unpack_batch(batch)
            logits = _forward(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.numel()
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            sample_ids_all.extend(list(sample_ids))
    denom = max(1, len(labels_all))
    return total_loss / denom, np.array(labels_all), np.array(probs_all), sample_ids_all


def _board_pooled_auc(probs, labels, sample_ids):
    """per_side eval: pool a board's two side scores to one board score (max) and
    score against the board G/NG label. Returns the board-level AUC (0.0 when a
    single class is present). board = labels are ignored except as a fallback."""
    from collections import defaultdict
    by_board = defaultdict(list)
    for prob, sid in zip(probs, sample_ids):
        by_board[_BOARD_ID_OF.get(sid, sid)].append(float(prob))
    board_ids = sorted(by_board)
    board_scores = np.array([max(by_board[b]) for b in board_ids])
    board_labels = np.array([_BOARD_LABEL_OF.get(b, 0) for b in board_ids])
    if len(np.unique(board_labels)) < 2:
        return 0.0, len(board_ids), int(board_labels.sum())
    try:
        return float(roc_auc_score(board_labels, board_scores)), len(board_ids), int(board_labels.sum())
    except ValueError:
        return 0.0, len(board_ids), int(board_labels.sum())


def _pool_for_metrics(probs, labels, sample_ids):
    if not PER_SIDE:
        return labels.astype(int), probs
    from collections import defaultdict
    by_board = defaultdict(list)
    for prob, sid in zip(probs, sample_ids):
        by_board[_BOARD_ID_OF.get(sid, sid)].append(float(prob))
    board_ids = sorted(by_board)
    board_scores = np.array([max(by_board[b]) for b in board_ids])
    board_labels = np.array([_BOARD_LABEL_OF.get(b, 0) for b in board_ids])
    return board_labels.astype(int), board_scores


def _require_non_empty_eval(split_name, labels, sample_ids):
    labels = np.asarray(labels).astype(int)
    sample_count = len(sample_ids) if sample_ids is not None else len(labels)
    ng_count = int((labels == 1).sum())
    g_count = int((labels == 0).sum())
    if sample_count == 0 or len(labels) == 0 or ng_count == 0 or g_count == 0:
        print('EVAL_ABORT:', json.dumps({
            'reason': 'empty_or_single_class_eval_slice',
            'split': split_name,
            'sample_count': int(sample_count),
            'ng_count': ng_count,
            'g_count': g_count,
        }))
        sys.exit(2)


def _safe_auc(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.0


def _selection_auc(probs, labels, sample_ids):
    if PER_SIDE:
        return _board_pooled_auc(probs, labels, sample_ids)[0]
    return _safe_auc(labels, probs)


def _threshold_metrics(labels, probs, threshold):
    tp, tn, fp, fn = _binary_counts(labels, probs, threshold)
    accuracy, ng_recall, miss_rate, overkill_rate, f1 = _metrics_from_counts(tp, tn, fp, fn)
    return {
        'accuracy': accuracy,
        'ng_recall': ng_recall,
        'miss_rate': miss_rate,
        'overkill_rate': overkill_rate,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


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


criterion = _resolve_criterion()
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
    _, probe_labels, probe_probs, _ = _evaluate_loss(val_loader, criterion)
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
if DRY_RUN:
    # Micro-runs (validator dry runs + debug smoke) take only a handful of
    # gradient steps on ~16 samples — the probe cannot move off its loss-weight
    # bias yet, so these gates would abort nearly every candidate on noise.
    # They only carry evidence on the full run; report the probe, never abort.
    reason = 'OK (DRY_RUN: probe gates report-only)'
elif probe_epoch_metrics and all(m['overkill_rate'] > 0.90 for m in probe_epoch_metrics):
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

# Probe passed — keep training the SAME model/optimizer: the probe epochs are
# the first PROBE_EPOCHS epochs of the run. (Rebuilding from scratch here used
# to throw away 5-8 completed epochs per script.) Dry runs keep the full epoch
# budget so the debug smoke run still emits enough EPOCH_LOG points for the
# curve-abort fit.
remaining_epochs = epochs if DRY_RUN else max(1, epochs - PROBE_EPOCHS)
best_val_loss = math.inf
best_val_auc = -math.inf
best_state = None
patience_left = PATIENCE


def _is_better_checkpoint(val_auc_for_selection, val_loss):
    return (
        val_auc_for_selection > best_val_auc
        or (val_auc_for_selection == best_val_auc and val_loss < best_val_loss)
    )

for epoch in range(1, remaining_epochs + 1):
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
    val_loss, val_labels, val_probs, val_sample_ids = _evaluate_loss(val_loader, criterion)
    val_auc_for_selection = _selection_auc(val_probs, val_labels, val_sample_ids)
    tp, tn, fp, fn = _binary_counts(val_labels, val_probs, 0.5)
    _, val_ng_recall, _, val_overkill, _ = _metrics_from_counts(tp, tn, fp, fn)
    val_board_pooled_auc = val_auc_for_selection if PER_SIDE else None
    print('EPOCH_LOG:', json.dumps({
        'epoch': epoch,
        'train_loss': train_loss / max(1, train_count),
        'val_loss': val_loss,
        'val_auc_for_selection': val_auc_for_selection,
        'val_board_pooled_auc': val_board_pooled_auc,
        'val_ng_recall': val_ng_recall,
        'val_overkill': val_overkill,
    }))
    scheduler.step()
    if _is_better_checkpoint(val_auc_for_selection, val_loss):
        best_val_loss = val_loss
        best_val_auc = val_auc_for_selection
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
val_metric_labels, val_metric_probs = _pool_for_metrics(val_probs, val_labels, val_sample_ids)
test_metric_labels, test_metric_probs = _pool_for_metrics(test_probs, test_labels, test_sample_ids)
_require_non_empty_eval('validation', val_metric_labels, val_sample_ids)
_require_non_empty_eval('test', test_metric_labels, test_sample_ids)
for i in range(10, 91):
    threshold = round(i / 100.0, 2)
    val_threshold_metrics = _threshold_metrics(val_metric_labels, val_metric_probs, threshold)
    recall = val_threshold_metrics['ng_recall']
    overkill = val_threshold_metrics['overkill_rate']
    miss_rate = val_threshold_metrics['miss_rate']
    accuracy = val_threshold_metrics['accuracy']
    all_candidates.append({
        'threshold': threshold,
        'recall': recall,
        'miss_rate': miss_rate,
        'overkill': overkill,
        'fp': val_threshold_metrics['fp'],
    })
    threshold_curve.append({
        't': threshold,
        'recall': recall,
        'overkill': overkill,
        'miss_rate': miss_rate,
        'accuracy': accuracy,
    })

recall_candidates = [c for c in all_candidates if c['recall'] >= VAL_NG_RECALL_TARGET]
if recall_candidates:
    best_threshold = min(recall_candidates, key=lambda c: c['threshold'])['threshold']
else:
    best_threshold = min(all_candidates, key=lambda c: (-c['recall'], c['threshold']))['threshold'] if all_candidates else 0.5
val_threshold_metrics = _threshold_metrics(val_metric_labels, val_metric_probs, best_threshold)
print('THRESHOLD_CURVE:', json.dumps(threshold_curve))
if best_threshold <= 0.15 and val_threshold_metrics['overkill_rate'] > 0.50:
    print('THRESHOLD_WARNING:', json.dumps({
        'reason': 'Low threshold selected to satisfy validation NG-recall target; overkill is expected to rise.',
        'threshold': best_threshold,
        'val_overkill_rate': val_threshold_metrics['overkill_rate'],
    }))

test_threshold_metrics = _threshold_metrics(test_metric_labels, test_metric_probs, best_threshold)
accuracy = test_threshold_metrics['accuracy']
ng_recall = test_threshold_metrics['ng_recall']
miss_rate = test_threshold_metrics['miss_rate']
overkill_rate = test_threshold_metrics['overkill_rate']
f1 = test_threshold_metrics['f1']
tp = test_threshold_metrics['tp']
tn = test_threshold_metrics['tn']
fp = test_threshold_metrics['fp']
fn = test_threshold_metrics['fn']
roc_auc = _safe_auc(test_metric_labels, test_metric_probs)
test_g_probs = test_metric_probs[test_metric_labels == 0]
test_ng_probs = test_metric_probs[test_metric_labels == 1]
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
    'threshold_selection_target': VAL_NG_RECALL_TARGET,
    'threshold_selection_target_met': val_threshold_metrics['ng_recall'] >= VAL_NG_RECALL_TARGET,
    'val_ng_recall_at_threshold': val_threshold_metrics['ng_recall'],
    'val_overkill_rate_at_threshold': val_threshold_metrics['overkill_rate'],
    'ng_count': int((test_metric_labels == 1).sum()),
    'g_count': int((test_metric_labels == 0).sum()),
    'tp': tp,
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'roc_auc': roc_auc,
    'prob_gap': prob_gap,
}
if PER_SIDE:
    # Pool the two sides back to a board score (max) and report the board-level
    # AUC on BOTH val and test. The headline ``roc_auc`` becomes the board-pooled
    # test AUC so it is apples-to-apples with board mode and with downstream gates;
    # the native per-image AUC is preserved under ``per_image_roc_auc``.
    board_val_auc, n_val_boards, _ = _board_pooled_auc(val_probs, val_labels, val_sample_ids)
    board_test_auc, n_test_boards, _ = _board_pooled_auc(test_probs, test_labels, test_sample_ids)
    metrics['label_granularity'] = 'per_side'
    metrics['per_image_roc_auc'] = roc_auc
    metrics['board_pooled_val_roc_auc'] = board_val_auc
    metrics['board_pooled_test_roc_auc'] = board_test_auc
    metrics['n_val_boards'] = n_val_boards
    metrics['n_test_boards'] = n_test_boards
    metrics['roc_auc'] = board_test_auc
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
    return (
        template
        .replace('__ARCHITECTURE_BLOCK__', architecture_block)
        .replace('__DATA_SPLIT_PATH__', repr(data_split_path))
    )
