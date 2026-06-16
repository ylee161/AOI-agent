import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from mle_star_agent.shared.labels import normalize_label

logger = logging.getLogger(__name__)


def _find_xlsx(lot_folder: str) -> Path:
    matches = list(Path(lot_folder).glob("*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"No xlsx found in {lot_folder}")
    return matches[0]


def _find_col(df: "pd.DataFrame", candidates: list) -> str:
    """Return the first column name (case-insensitive) that matches any candidate."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def _load_label_map(xlsx_path: Path) -> dict:
    df = pd.read_excel(xlsx_path, sheet_name=0)
    row_col = _find_col(df, ["Row", "row", "ROW", "y"])
    col_col = _find_col(df, ["Column", "column", "Col", "col", "COL", "x"])
    result_col = _find_col(df, ["TestResult", "testresult", "Result", "result", "label", "Label"])
    df = df.dropna(subset=[row_col, col_col, result_col])
    label_map = {}
    for _, r in df.iterrows():
        key = (int(r[row_col]), int(r[col_col]))
        result = str(r[result_col]).strip()
        # Map the dataset's raw result value to the canonical G/NG code using the
        # PASS_LABELS / FAIL_LABELS configured in config.py. strict=True so an
        # unrecognised value fails loudly here rather than being silently dropped.
        label = normalize_label(result, strict=True, context=f"(row={key[0]}, col={key[1]}) of {xlsx_path}")
        label_map[key] = label
    return label_map


# Per-side defect-count columns. per_side labelling needs the number of defects
# on each side of the board so a side can be labelled NG iff its OWN count > 0.
# Listed case-insensitively; a dataset without these columns falls back to board
# labelling (see dataset_supports_per_side), keeping the pipeline reusable.
_NG_SUM_L_CANDIDATES = ["NG_SUM_L", "ng_sum_l", "NGSUM_L", "NG_L", "ngsum_l"]
_NG_SUM_R_CANDIDATES = ["NG_SUM_R", "ng_sum_r", "NGSUM_R", "NG_R", "ngsum_r"]


def _find_col_optional(df: "pd.DataFrame", candidates: list):
    """Like _find_col but returns None instead of raising when nothing matches."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def dataset_supports_per_side(dataset_folders: List[str]) -> bool:
    """True iff every lot's label sheet carries per-side NG-count columns."""
    for lot_folder in dataset_folders:
        try:
            df = pd.read_excel(_find_xlsx(lot_folder), sheet_name=0)
        except Exception:
            return False
        if (_find_col_optional(df, _NG_SUM_L_CANDIDATES) is None
                or _find_col_optional(df, _NG_SUM_R_CANDIDATES) is None):
            return False
    return True


def _load_side_label_map(xlsx_path: Path) -> dict:
    """Return {(row, col): {"L": "NG"/"G", "R": "NG"/"G", "board": "NG"/"G"}}.

    Side label = NG iff that side's own defect count (NG_SUM_L / NG_SUM_R) > 0 —
    this is what removes the clean-side dilution. The board label reuses the
    canonical TestResult mapping so board-pooled metrics stay comparable to
    "board" mode. Callers gate on dataset_supports_per_side() first.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)
    row_col = _find_col(df, ["Row", "row", "ROW", "y"])
    col_col = _find_col(df, ["Column", "column", "Col", "col", "COL", "x"])
    result_col = _find_col(df, ["TestResult", "testresult", "Result", "result", "label", "Label"])
    ng_l_col = _find_col(df, _NG_SUM_L_CANDIDATES)
    ng_r_col = _find_col(df, _NG_SUM_R_CANDIDATES)
    df = df.dropna(subset=[row_col, col_col, result_col, ng_l_col, ng_r_col])
    out = {}
    for _, r in df.iterrows():
        key = (int(r[row_col]), int(r[col_col]))
        board = normalize_label(
            str(r[result_col]).strip(), strict=True,
            context=f"(row={key[0]}, col={key[1]}) of {xlsx_path}",
        )
        out[key] = {
            "L": "NG" if float(r[ng_l_col]) > 0 else "G",
            "R": "NG" if float(r[ng_r_col]) > 0 else "G",
            "board": board,
        }
    return out


def _build_pairs(lot_folder: str) -> dict:
    """Return {pair_key: {"img_l": path, "img_r": path}} for one lot folder."""
    pairs: dict = {}
    for png in Path(lot_folder).glob("*.png"):
        stem = png.stem
        if "_L_" in stem:
            side = "img_l"
        elif "_R_" in stem:
            side = "img_r"
        else:
            continue  # skip non-stereo PNGs
        pair_key = re.sub(r"_[LR]_", "_", stem)
        if pair_key not in pairs:
            pairs[pair_key] = {}
        pairs[pair_key][side] = str(png)
    return pairs


def _row_col_from_key(pair_key: str):
    """Parse row and col from pair_key format '{row}-{col}_{rest}'."""
    row_col = pair_key.split("_")[0]
    row, col = row_col.split("-")
    return int(row), int(col)


def _board_code_config() -> tuple:
    """Return (pattern, strip_digits) from config, with SUP046-compatible
    defaults so this module stays usable standalone (e.g. in tests) even when
    config is not importable.
    """
    pattern, strip_digits = r"VHB[A-Z0-9]+", 2
    try:
        from mle_star_agent import config

        pattern = getattr(config, "BOARD_CODE_PATTERN", pattern)
        strip_digits = getattr(config, "BOARD_CODE_STRIP_SUFFIX_DIGITS", strip_digits)
    except Exception:
        pass
    return pattern, strip_digits


def _extract_board_code(lot_name: str) -> str:
    """Extract the board group code from a lot-folder name using the configured
    BOARD_CODE_PATTERN, stripping any configured trailing lot-sequence digits.

    Default (SUP046): '[SUP046]2026040301.0002_VHB48301B0701' -> 'VHB48301B07'.
    Returns the full lot_name as fallback when the pattern does not match.
    """
    pattern, strip_digits = _board_code_config()
    m = re.search(pattern, lot_name)
    if not m:
        return lot_name
    code = m.group(0)
    # Strip the trailing lot-sequence digits so different lots of the same
    # physical board share one group key.
    if strip_digits > 0 and len(code) > strip_digits and code[-strip_digits:].isdigit():
        return code[:-strip_digits]
    return code


def _board_grouped_split(
    samples: list,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Assign whole board groups to train / val / test so that no board appears in
    both training and held-out sets.

    Strategy:
      - Group samples by board_code.
      - Sort groups smallest-first; fill val+test from the smallest groups until
        the holdout target (val_frac + test_frac) is reached.
      - Remaining groups go entirely to train.
      - The holdout pool is then stratified-split 50/50 into val and test.
      - Falls back to sample-level stratified split when fewer than 3 distinct
        board groups exist (pure group split cannot fill all three partitions).

    Returns (train_ids, val_ids, test_ids) as lists of sample_id strings.
    """
    board_to_samples: dict[str, list] = defaultdict(list)
    for s in samples:
        board_to_samples[s["board_code"]].append(s)

    n_boards = len(board_to_samples)

    if n_boards < 3:
        logger.warning(
            "Only %d distinct board group(s) found — falling back to "
            "stratified sample-level split to keep all three partitions non-empty.",
            n_boards,
        )
        ids = [s["sample_id"] for s in samples]
        labels = [s["label"] for s in samples]
        train_ids, temp_ids, _, temp_labels = train_test_split(
            ids, labels, test_size=val_frac + test_frac, stratify=labels,
            random_state=random_state,
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.50, stratify=temp_labels,
            random_state=random_state,
        )
        return train_ids, val_ids, test_ids

    total = len(samples)
    holdout_target = total * (val_frac + test_frac)

    # Smallest groups first so we consume the minimum number of boards for holdout
    sorted_boards = sorted(board_to_samples.keys(), key=lambda b: len(board_to_samples[b]))

    holdout_boards: list[str] = []
    train_boards: list[str] = []
    holdout_count = 0

    for board in sorted_boards:
        if holdout_count < holdout_target:
            holdout_boards.append(board)
            holdout_count += len(board_to_samples[board])
        else:
            train_boards.append(board)

    # Ensure at least one board in train
    if not train_boards:
        train_boards.append(holdout_boards.pop())

    train_ids = [s["sample_id"] for b in train_boards for s in board_to_samples[b]]

    holdout_samples = [s for b in holdout_boards for s in board_to_samples[b]]
    holdout_ids = [s["sample_id"] for s in holdout_samples]
    holdout_labels = [s["label"] for s in holdout_samples]

    if len(holdout_ids) >= 2:
        try:
            val_ids, test_ids = train_test_split(
                holdout_ids, test_size=0.50, stratify=holdout_labels,
                random_state=random_state,
            )
        except ValueError:
            # Can't stratify (e.g. only one class in holdout) — split without stratify
            val_ids, test_ids = train_test_split(
                holdout_ids, test_size=0.50, random_state=random_state,
            )
    else:
        val_ids = holdout_ids
        test_ids = []

    logger.info(
        "Board-grouped split: %d train boards (%d samples), "
        "%d holdout boards (%d samples → %d val / %d test). "
        "Train boards: %s  Holdout boards: %s",
        len(train_boards), len(train_ids),
        len(holdout_boards), len(holdout_ids), len(val_ids), len(test_ids),
        train_boards, holdout_boards,
    )
    return train_ids, val_ids, test_ids


def board_grouped_kfold(samples: list, k: int = 3, random_state: int = 42) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Return board-grouped train/val folds from sample dictionaries.

    Each board group appears in validation exactly once and never appears in
    both train and validation for the same fold. Board keys reuse the same
    configured extraction path as the fixed grouped split when a sample does not
    already carry ``board_code``.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if not samples:
        raise ValueError("samples must not be empty")

    normalized_samples = []
    for sample in samples:
        item = dict(sample)
        if not item.get("board_code"):
            lot_name = item.get("lot") or str(item.get("sample_id", "")).split("/", 1)[0]
            item["board_code"] = _extract_board_code(str(lot_name))
        normalized_samples.append(item)

    board_to_samples: dict[str, list] = defaultdict(list)
    for sample in normalized_samples:
        board_to_samples[sample["board_code"]].append(sample)

    if len(board_to_samples) < k:
        raise ValueError(
            f"Need at least {k} board groups for board-grouped CV; "
            f"found {len(board_to_samples)}."
        )

    boards = sorted(
        board_to_samples,
        key=lambda board: (len(board_to_samples[board]), board),
        reverse=True,
    )
    fold_boards = [[] for _ in range(k)]
    fold_sizes = [0 for _ in range(k)]
    for board in boards:
        fold_index = min(range(k), key=lambda i: (fold_sizes[i], i))
        fold_boards[fold_index].append(board)
        fold_sizes[fold_index] += len(board_to_samples[board])

    folds = []
    for val_boards in fold_boards:
        val_board_set = set(val_boards)
        train_rows = [
            sample
            for sample in normalized_samples
            if sample["board_code"] not in val_board_set
        ]
        val_rows = [
            sample
            for board in val_boards
            for sample in board_to_samples[board]
        ]
        folds.append((pd.DataFrame(train_rows), pd.DataFrame(val_rows)))

    return folds


def stratified_kfold(samples: list, k: int = 3, random_state: int = 42) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Return label-stratified train/val folds for the MIXED (in-distribution) regime.

    Mirrors ``board_grouped_kfold``'s return shape (list of (train_df, val_df))
    but imposes NO lot/board grouping — every fold's val set is a stratified
    random slice, so lots are mixed across train and val exactly as in the mixed
    split itself. Use this (not the grouped k-fold) to confirm a candidate that
    was trained and selected under ``SPLIT_STRATEGY="mixed"``; using the grouped
    k-fold there would measure a different (cross-lot) regime and can wrongly
    reject an in-distribution candidate.
    """
    from sklearn.model_selection import StratifiedKFold

    if k < 2:
        raise ValueError("k must be at least 2")
    if not samples:
        raise ValueError("samples must not be empty")

    rows = [dict(s) for s in samples]
    labels = [r.get("label") for r in rows]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(rows, labels):
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        folds.append((pd.DataFrame(train_rows), pd.DataFrame(val_rows)))
    return folds


def detect_input_modality(dataset_folders: List[str]) -> str:
    """Auto-detect whether the dataset is stereo (paired _L_/_R_ PNGs) or mono.

    Uses the same non-recursive root-level glob as ``_build_pairs`` and inspects
    ``Path(p).stem`` for the "_L_" / "_R_" stereo infix.

    Returns "stereo" or "mono". Raises ValueError on a mixed dataset or when no
    image files are found at all.
    """
    pngs = []
    for folder in dataset_folders:
        pngs.extend(Path(folder).glob("*.png"))

    # Drop board-level overview/map images (e.g. '..._Map.png'). They are not
    # per-cell samples — _build_pairs() already ignores them — and they carry
    # neither the _L_/_R_ stereo infix nor a row-col sample key, so counting
    # them would spuriously flag a stereo dataset as mixed-modality.
    pngs = [p for p in pngs if not Path(p).stem.lower().endswith("_map")]

    if not pngs:
        raise ValueError(
            f"No image files found in any dataset folder: {list(dataset_folders)}"
        )

    has_L = any("_L_" in Path(p).stem for p in pngs)
    has_R = any("_R_" in Path(p).stem for p in pngs)
    has_mono = any(
        "_L_" not in Path(p).stem and "_R_" not in Path(p).stem for p in pngs
    )

    if has_L and has_R and not has_mono:
        return "stereo"
    if has_mono and not has_L and not has_R:
        return "mono"
    raise ValueError(
        f"Mixed modality detected in {list(dataset_folders)}: "
        f"has_L={has_L}, has_R={has_R}, has_mono={has_mono}. "
        f"Expected either all stereo (paired _L_/_R_) or all mono PNGs."
    )


def build_data_split(dataset_folders: List[str], label_granularity: str = "board", split_strategy: str = "grouped") -> dict:
    """
    Collect samples and labels from all lot folders, then produce a
    board-grouped 70/15/15 train/val/test split.

    Samples from the same VHB board group are kept in the same partition to
    prevent board-specific pattern leakage between train and held-out sets.

    ``label_granularity`` (default "board" so the function's standalone behaviour
    is unchanged; the live pipeline passes ``config.LABEL_GRANULARITY``):
      * "board"    — one stereo (or mono) sample per board, board G/NG label.
      * "per_side" — each side of a stereo pair becomes its own 3-channel mono
        sample, labelled NG iff its OWN defect count > 0. Each sample also carries
        ``board_id`` and ``board_label`` so training can pool the two sides back
        to a board score (max) at eval and preserve the board-level metric.
        Falls back to "board" (with a warning) when the dataset is not stereo or
        lacks per-side defect columns, keeping the pipeline reusable.

    Returns a dict suitable for JSON serialisation and writing to state.
    """
    modality = detect_input_modality(dataset_folders)

    per_side = label_granularity == "per_side"
    if per_side and modality != "stereo":
        logger.warning(
            "per_side labelling requires stereo pairs but dataset is %s — "
            "falling back to board labelling.", modality,
        )
        per_side = False
    if per_side and not dataset_supports_per_side(dataset_folders):
        logger.warning(
            "per_side labelling requested but per-side defect columns "
            "(NG_SUM_L / NG_SUM_R) are missing — falling back to board labelling.",
        )
        per_side = False
    used_granularity = "per_side" if per_side else "board"

    samples = []

    for lot_folder in dataset_folders:
        xlsx_path = _find_xlsx(lot_folder)
        lot_name = str(Path(lot_folder).name)
        board_code = _extract_board_code(lot_name)

        if per_side:
            # Each side becomes its own mono sample, labelled by its own defects.
            # NOTE (honest framing): this attacks the under-learning / clean-side
            # dilution axis (it ~doubles samples); it does NOT add lots, so it
            # cannot move the cross-lot domain-shift wall.
            side_map = _load_side_label_map(xlsx_path)
            for pair_key, sides in _build_pairs(lot_folder).items():
                if "img_l" not in sides or "img_r" not in sides:
                    logger.warning("Incomplete stereo pair skipped: %s", pair_key)
                    continue
                try:
                    row, col = _row_col_from_key(pair_key)
                except (ValueError, IndexError):
                    logger.warning("Cannot parse row/col from pair key: %s", pair_key)
                    continue
                sm = side_map.get((row, col))
                if sm is None:
                    logger.warning("No label for (%d, %d) in %s — skipping", row, col, lot_folder)
                    continue
                board_id = f"{lot_name}/{pair_key}"
                for side, img_key in (("L", "img_l"), ("R", "img_r")):
                    samples.append({
                        "sample_id": f"{board_id}::{side}",
                        "board_id": board_id,
                        "lot": lot_name,
                        "board_code": board_code,
                        "side": side,
                        "img": sides[img_key],
                        "label": sm[side],
                        "board_label": sm["board"],
                    })
            continue

        label_map = _load_label_map(xlsx_path)

        if modality == "mono":
            for png in Path(lot_folder).glob("*.png"):
                key = png.stem
                if key.lower().endswith("_map"):
                    continue  # board overview image, not a per-cell sample
                try:
                    row, col = _row_col_from_key(key)
                except (ValueError, IndexError):
                    logger.warning("Cannot parse row/col from mono key: %s", key)
                    continue

                label = label_map.get((row, col))
                if label is None:
                    logger.warning("No label for (%d, %d) in %s — skipping", row, col, lot_folder)
                    continue

                samples.append({
                    "sample_id": f"{lot_name}/{key}",
                    "lot": lot_name,
                    "board_code": board_code,
                    "img": str(png),
                    "label": label,
                })
            continue

        pairs = _build_pairs(lot_folder)
        for pair_key, sides in pairs.items():
            if "img_l" not in sides or "img_r" not in sides:
                logger.warning("Incomplete stereo pair skipped: %s", pair_key)
                continue

            try:
                row, col = _row_col_from_key(pair_key)
            except (ValueError, IndexError):
                logger.warning("Cannot parse row/col from pair key: %s", pair_key)
                continue

            label = label_map.get((row, col))
            if label is None:
                logger.warning("No label for (%d, %d) in %s — skipping", row, col, lot_folder)
                continue

            samples.append({
                "sample_id": f"{lot_name}/{pair_key}",
                "lot": lot_name,
                "board_code": board_code,
                "img_l": sides["img_l"],
                "img_r": sides["img_r"],
                "label": label,
            })

    if not samples:
        raise ValueError(f"No valid {modality} samples found across all lot folders.")

    labels = [s["label"] for s in samples]
    ng_count = labels.count("NG")
    g_count = labels.count("G")
    logger.info("Total samples: %d  (NG=%d, G=%d)", len(samples), ng_count, g_count)

    board_codes = sorted({s["board_code"] for s in samples})
    logger.info("Board groups found (%d): %s", len(board_codes), board_codes)

    if split_strategy == "mixed":
        # Simple stratified random split — all lots appear in every partition.
        # This is the in-distribution ("MIXED") regime where the 0.65–0.71
        # results were produced. No lot-grouping constraint is applied.
        if per_side:
            # For per_side, split at board level first to keep both sides together.
            board_reps, seen = [], set()
            for s in samples:
                if s["board_id"] in seen:
                    continue
                seen.add(s["board_id"])
                board_reps.append({"sample_id": s["board_id"], "board_code": s["board_code"], "label": s["board_label"]})
            rep_ids = [r["sample_id"] for r in board_reps]
            rep_labels = [r["label"] for r in board_reps]
            tr_b, temp_b, _, temp_l = train_test_split(rep_ids, rep_labels, test_size=0.30, stratify=rep_labels, random_state=42)
            val_b, test_b = train_test_split(temp_b, test_size=0.50, stratify=temp_l, random_state=42)
            partition = {bid: p for p, ids in (("train", tr_b), ("val", val_b), ("test", test_b)) for bid in ids}
            buckets: dict = {"train": [], "val": [], "test": []}
            for s in samples:
                dest = partition.get(s["board_id"])
                if dest is not None:
                    buckets[dest].append(s)
            train_ids = [s["sample_id"] for s in buckets["train"]]
            val_ids = [s["sample_id"] for s in buckets["val"]]
            test_ids = [s["sample_id"] for s in buckets["test"]]
        else:
            ids = [s["sample_id"] for s in samples]
            lbls = [s["label"] for s in samples]
            train_ids, temp_ids, _, temp_lbls = train_test_split(ids, lbls, test_size=0.30, stratify=lbls, random_state=42)
            val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, stratify=temp_lbls, random_state=42)
        logger.info("Mixed split: train=%d val=%d test=%d (all lots in every partition)", len(train_ids), len(val_ids), len(test_ids))
    elif per_side:
        # Grouped + per_side: split at the BOARD level so both sides of a board
        # always land in the same partition, then expand to side samples.
        board_reps, seen = [], set()
        for s in samples:
            if s["board_id"] in seen:
                continue
            seen.add(s["board_id"])
            board_reps.append({
                "sample_id": s["board_id"],
                "board_code": s["board_code"],
                "label": s["board_label"],
            })
        tr_b, val_b, test_b = _board_grouped_split(board_reps)
        partition = {}
        for name, ids in (("train", tr_b), ("val", val_b), ("test", test_b)):
            for bid in ids:
                partition[bid] = name
        gbuckets: dict = {"train": [], "val": [], "test": []}
        for s in samples:
            dest = partition.get(s["board_id"])
            if dest is not None:
                gbuckets[dest].append(s)
        train_ids = [s["sample_id"] for s in gbuckets["train"]]
        val_ids = [s["sample_id"] for s in gbuckets["val"]]
        test_ids = [s["sample_id"] for s in gbuckets["test"]]
    else:
        train_ids, val_ids, test_ids = _board_grouped_split(samples)

    sample_map = {s["sample_id"]: s for s in samples}

    def subset(id_list):
        return [sample_map[i] for i in id_list]

    # per_side emits single 3-channel images, so downstream modality-aware logic
    # (augmentation safety, stereo-fusion gates) must see "mono".
    recorded_modality = "mono" if per_side else modality
    split = {
        "metadata": {
            "input_modality": recorded_modality,
            "label_granularity": used_granularity,
            "split_strategy": split_strategy,
        },
        "train": subset(train_ids),
        "val": subset(val_ids),
        "test": subset(test_ids),
        "stats": {
            "total": len(samples),
            "ng_count": ng_count,
            "g_count": g_count,
            "train_size": len(train_ids),
            "val_size": len(val_ids),
            "test_size": len(test_ids),
            "board_groups": board_codes,
        },
    }

    logger.info(
        "Split: train=%d val=%d test=%d",
        len(train_ids), len(val_ids), len(test_ids),
    )
    return split
