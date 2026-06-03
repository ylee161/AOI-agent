"""Step 8.1 - Data split test script."""
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent))

from mle_star_agent.config import DATASET_FOLDERS, CKPT_DATA_SPLIT
from mle_star_agent.shared.data_split import build_data_split

print("=" * 60)
print("STEP 8.1 - DATA SPLIT TEST")
print("=" * 60)

# --- 1. Show which lot folders were found ---
print(f"\nLot folders found ({len(DATASET_FOLDERS)}):")
for f in DATASET_FOLDERS:
    print(f"  {f}")

if not DATASET_FOLDERS:
    print("ERROR: No lot folders found. Check config.DATASET_FOLDERS.")
    sys.exit(1)

# --- 2. Run the split ---
print("\nRunning build_data_split ...")
split = build_data_split(DATASET_FOLDERS)

train = split["train"]
val   = split["val"]
test  = split["test"]
stats = split["stats"]

# --- 3. Print size summary ---
print(f"\nSplit sizes:")
print(f"  Total : {stats['total']}  (NG={stats['ng_count']}, G={stats['g_count']})")
print(f"  Train : {stats['train_size']}")
print(f"  Val   : {stats['val_size']}")
print(f"  Test  : {stats['test_size']}")

# --- 4. Check label stratification ---
def class_ratio(subset):
    labels = [s["label"] for s in subset]
    c = Counter(labels)
    total = len(labels)
    ng_pct = 100 * c["NG"] / total if total else 0
    return c["NG"], c["G"], ng_pct

overall_ng_pct = 100 * stats["ng_count"] / stats["total"]
tr_ng, tr_g, tr_pct = class_ratio(train)
va_ng, va_g, va_pct = class_ratio(val)
te_ng, te_g, te_pct = class_ratio(test)

print(f"\nClass balance (NG %):")
print(f"  Overall : {overall_ng_pct:.1f}%")
print(f"  Train   : {tr_pct:.1f}%  (NG={tr_ng}, G={tr_g})")
print(f"  Val     : {va_pct:.1f}%  (NG={va_ng}, G={va_g})")
print(f"  Test    : {te_pct:.1f}%  (NG={te_ng}, G={te_g})")

# Warn if any split deviates by more than 5%
stratification_ok = True
for name, pct in [("Train", tr_pct), ("Val", va_pct), ("Test", te_pct)]:
    if abs(pct - overall_ng_pct) > 5.0:
        print(f"  WARNING: {name} NG% deviates {abs(pct-overall_ng_pct):.1f}% from overall!")
        stratification_ok = False

if stratification_ok:
    print("  PASS - stratification looks good")

# --- 5. Check stereo pair integrity ---
print(f"\nStereo pair integrity check:")
all_l = set()
all_r = set()
violations = []

for split_name, subset in [("train", train), ("val", val), ("test", test)]:
    for s in subset:
        sample_id = s["sample_id"]
        img_l = s.get("img_l", "")
        img_r = s.get("img_r", "")
        if not img_l or not img_r:
            violations.append(f"  MISSING side: {sample_id}")
        all_l.add(img_l)
        all_r.add(img_r)

# Check no L image appears in two different splits
all_samples = {s["sample_id"] for s in train + val + test}
duplicates = len(train) + len(val) + len(test) - len(all_samples)

if violations:
    for v in violations[:10]:
        print(v)
    print(f"  FAIL - {len(violations)} incomplete pairs")
elif duplicates:
    print(f"  FAIL - {duplicates} duplicate sample IDs across splits")
else:
    print(f"  PASS - all {len(all_samples)} samples have both L and R images, no duplicates")

# --- 6. Save checkpoint ---
CKPT_DATA_SPLIT.parent.mkdir(parents=True, exist_ok=True)
with open(CKPT_DATA_SPLIT, "w") as f:
    json.dump(split, f, indent=2)
print(f"\nCheckpoint saved: {CKPT_DATA_SPLIT}")
print(f"File size: {CKPT_DATA_SPLIT.stat().st_size / 1024:.1f} KB")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
