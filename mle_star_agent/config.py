import glob
import os
from pathlib import Path
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# LiteLLM's deepseek/ provider reads DEEPSEEK_API_KEY directly — no OPENAI_* vars needed
if not os.environ.get("DEEPSEEK_API_KEY"):
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env before running.")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ─── Dataset location ────────────────────────────────────────────────────────
# NEW USERS: change this ONE line to point at your data. It is a glob (relative
# to PROJECT_ROOT, or an absolute path) that must match your lot folders — one
# folder per lot, each containing the lot's PNG images and a single .xlsx label
# sheet. The default matches the bundled SUP046 lots, whose folders look like
# "[SUP046]2026040301.0002_VHB48301B0701". The "[[]" escapes the literal "[".
DATASET_GLOB = "aoi_agent_dataset/[[]SUP046]*"

# Resolved list of lot folders (sorted for determinism). Leave this as-is.
DATASET_FOLDERS = sorted(glob.glob(str(PROJECT_ROOT / DATASET_GLOB)))

# ─── Label convention ────────────────────────────────────────────────────────
# Your .xlsx result column may use any wording. List every raw value that means
# "defect / reject" in FAIL_LABELS and every value that means "good / accept" in
# PASS_LABELS. Matching is case-insensitive. Internally these are mapped to the
# canonical NG (fail) and G (pass) codes the pipeline uses everywhere.
# The defaults below cover G/NG, PASS/FAIL, OK, 0/1, etc. — extend as needed.
# A label value that appears in neither set causes a loud error at data load
# (it is never silently dropped), so add new conventions here.
FAIL_LABELS = {"fail", "ng", "1", "defect", "defective", "true", "positive"}
PASS_LABELS = {"pass", "ok", "g", "good", "0", "false", "negative"}

# ─── Board grouping ──────────────────────────────────────────────────────────
# Samples from the same physical board are kept in the same train/val/test
# partition to prevent leakage. BOARD_CODE_PATTERN is a regex matched against
# each lot-folder name; the matched substring is the board's group key. The
# default extracts the "VHB…" code from SUP046 lot names. Set to a pattern that
# captures your board identifier, or to r".+" to treat each lot as its own board.
BOARD_CODE_PATTERN = r"VHB[A-Z0-9]+"
# Trailing digits stripped off the matched code so different lot-sequence runs of
# the same board share one group (SUP046: "VHB48301B0701" -> "VHB48301B07").
# Set to 0 to disable stripping.
BOARD_CODE_STRIP_SUFFIX_DIGITS = 2

# Models
# num_retries is forwarded to litellm.acompletion (ADK's LiteLlm passes through
# **kwargs), which retries transient 429/503/timeout errors with exponential
# backoff at the SDK layer. This is the layer that can actually re-issue the
# call — an ADK on_model_error_callback cannot (see shared/callbacks.py).
MODEL = LiteLlm(
    model="deepseek/deepseek-v4-flash",
    thinking={"type": "disabled"},
    num_retries=3,
)       # fast — evaluators, gates, bookkeeping
MODEL_PRO = LiteLlm(
    model="deepseek/deepseek-v4-pro",
    thinking={"type": "enabled", "budget_tokens": 16000},
    reasoning_effort="max",
    output_config={"effort": "max"},
    num_retries=3,
)  # strong — coders, diagnosis, selection

# Loop caps
OUTER_LOOP_MAX = 10
INNER_LOOP_MAX = 10
ENSEMBLE_LOOP_MAX = 10
NO_IMPROVE_MAX = 2              # patience once final acceptance is met
NO_IMPROVE_MAX_CONSTRAINED = 5  # patience once relaxed (but not final) acceptance is met
INNER_STAGNATION_MAX_UNCONSTRAINED = 5  # restart outer diagnosis when below relaxed and stuck
HIGH_OVERKILL_STAGNATION_THRESHOLD = 0.50   # overkill above this is considered catastrophic
INNER_STAGNATION_MAX_HIGH_OVERKILL = 3      # max inner attempts before forcing new diagnosis when overkill is catastrophic
TARGET_COMPONENT_ROTATION_LOCK_FAILED_ATTEMPTS = 6
SUBMISSION_RETRY_MAX = 2  # max Phase 2→3→4 retry cycles if submission fails relaxed minimum

# Training — enforced minimums so generated scripts cannot under-train
MIN_EPOCHS = 20          # full-run epoch floor; generated scripts must use this or higher
EARLY_STOPPING_PATIENCE = 3  # epochs without val-loss improvement before stopping
MULTISEED_CONFIRMATION_SEEDS = (42, 101, 202)  # includes the discovery seed so confirmed avg is consistent
MULTISEED_CONFIRMATION_OVERKILL_MAX = 0.12  # only pay 3x compute for plausible near-target candidates
REFINEMENT_POPULATION_MAX = 5  # keep a small Pareto-style archive for non-linear refinement
ERROR_ANALYSIS_SAMPLE_CAP = 20  # max FP/FN samples surfaced into an LLM prompt (full set stays on disk)
VALIDATION_CACHE_MAX = 50       # cap on the in-state {sha256: status} validation cache
TRIED_APPROACHES_RECENT_K = 8   # full recent tried_approaches entries shown in the planner prompt (rest aggregated)
PROBE_OVERKILL_REJECT_MAX = 0.50  # reject cheap probes that already false-reject most G samples
PROBE_NG_RECALL_REJECT_MIN = 0.80  # reject probes with severe NG recall collapse
PROBE_PROBABILITY_GAP_MIN = 0.05  # require NG mean probability to exceed G mean by this much

# KompeteAI-style predictive early-abort on the debug smoke-run (max_epochs=1, 5% data).
# These gates run on the METRICS the *micro-run* already prints, so an obviously-bad
# refinement variant is pruned BEFORE paying for the 45-60 min full training run.
# They are deliberately MUCH looser than the probe-gate / acceptance thresholds above
# because a 1-epoch / 5%-data run is NOISY: we only abort on EGREGIOUS failure and let
# every borderline / near-target candidate fall through to the full run. When in doubt,
# run the full job (absence of parseable METRICS on 5% data is NOT evidence of a bad script).
DEBUG_PREDICT_OVERKILL_MAX = 0.60   # abort only if the micro-run already false-rejects most G
DEBUG_PREDICT_NG_RECALL_MIN = 0.50  # abort only on severe NG-recall collapse

# Execution
TIMEOUT_SECONDS = 7200  # 2 hours — ResNet18 15-epoch CPU training takes ~45-60 min per script
DEBUG_CHECK_TIMEOUT_SECONDS = 120  # cap for debug_mode smoke runs (max_epochs=1, 5% data)
DEBUGGER_RETRY_CAP = 3

# Token budget
TOKEN_BUDGET = 10_000_000
LITE_ABLATION_MODE = True
TOKEN_LITE_THRESHOLD = 7_000_000  # Switch to flash model once tokens exceed this (must be < TOKEN_BUDGET)
# P0: Miss Rate  P1: NG Recall  P2: Overkill Rate  P4: Accuracy
OVERKILL_RELAXED_MAX  = 0.08   # §9.1 relaxed minimum
OVERKILL_FINAL_MAX    = 0.05   # §9.2 final target
NG_RECALL_RELAXED_MIN = 0.97   # §9.1 relaxed minimum
NG_RECALL_FINAL_MIN   = 1.00   # §9.2 final target
MISS_RATE_RELAXED_MAX = 0.03   # §9.1 relaxed minimum (P0 — highest priority)
MISS_RATE_FINAL_MAX   = 0.00   # §9.2 final target
ACCURACY_RELAXED_MIN  = 0.92   # §9.1 relaxed minimum
ACCURACY_FINAL_MIN    = 0.97   # §9.2 final target

# Threshold sweep
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.05

# Rate limiting — sleep between sequential LLM-heavy steps (free-tier RPM guard)
RATE_LIMIT_DELAY_SECONDS = 3

# Checkpoint file names
DATA_SPLIT_VERSION = "data_split_grouped_v1"
CKPT_DATA_SPLIT_LEGACY = CHECKPOINT_DIR / "data_split.json"
CKPT_DATA_SPLIT = CHECKPOINT_DIR / "data_split_grouped.json"
CKPT_CANDIDATE_SCRIPTS = CHECKPOINT_DIR / "candidate_scripts.json"
CKPT_CANDIDATE_SCORES = CHECKPOINT_DIR / "candidate_scores.json"
CKPT_L0 = CHECKPOINT_DIR / "L0.json"
CKPT_PHASE2_INIT = CHECKPOINT_DIR / "phase2_init.json"
CKPT_BEST_PIPELINE = CHECKPOINT_DIR / "best_pipeline.json"
CKPT_ENSEMBLE = CHECKPOINT_DIR / "ensemble.json"
CKPT_SUBMISSION = CHECKPOINT_DIR / "submission.json"
CKPT_SUBMISSION_ATTEMPTS = CHECKPOINT_DIR / "submission_attempts.json"
CKPT_TRIED_APPROACHES = CHECKPOINT_DIR / "tried_approaches.json"
CKPT_TRIED_ENSEMBLE_APPROACHES = CHECKPOINT_DIR / "tried_ensemble_approaches.json"
CKPT_PERSISTENT_KB = CHECKPOINT_DIR / "persistent_aoi_kb.json"


def ckpt_ablation(n: int) -> Path:
    return CHECKPOINT_DIR / f"ablation_{n}.json"


def ckpt_ablation_variant(n: int, variant_index: int) -> Path:
    return CHECKPOINT_DIR / f"ablation_{n}_variant_{variant_index}.json"


def ckpt_diagnosis(n: int) -> Path:
    return CHECKPOINT_DIR / f"diagnosis_{n}.json"


def ckpt_refinement(n: int, m: int) -> Path:
    return CHECKPOINT_DIR / f"refinement_{n}_{m}.json"


def ckpt_error_analysis(n: int, m: int) -> Path:
    return CHECKPOINT_DIR / f"error_analysis_{n}_{m}.json"


def ckpt_error_analysis_report(n: int, m: int) -> Path:
    return CHECKPOINT_DIR / f"error_analysis_report_{n}_{m}.json"


def ckpt_ensemble_attempt(n: int) -> Path:
    return CHECKPOINT_DIR / f"ensemble_{n}.json"
