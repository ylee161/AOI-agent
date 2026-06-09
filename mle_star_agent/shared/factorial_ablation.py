"""Factorialized ablation grid for Phase 2.

Instead of removing one pipeline component at a time, the factorial design
toggles every combination of the core components on/off. With K=4 factors this
is a 2^4 = 16-cell grid; the all-on cell is the baseline (the current best
pipeline), so it is skipped, leaving 15 ablation variants.

The baseline is "all factors on". A variant's description therefore lists only
what was turned *off* relative to that baseline.
"""

import itertools

# Core pipeline components toggled by the factorial grid. Order is fixed: it
# determines variant names and (via sorted product) the stable variant ordering.
ABLATION_FACTORS = [
    {
        "name": "stereo_fusion",
        "on_description": (
            "Stereo fusion ON: load and combine both the left (_L) and right (_R) "
            "images."
        ),
        "off_description": (
            "Remove stereo fusion: use only the left (_L) image instead of combining "
            "L and R images. Drop all code that loads or merges the _R image."
        ),
    },
    {
        "name": "threshold_sweep",
        "on_description": (
            "Threshold sweep ON: select the classification threshold by sweeping the "
            "validation set."
        ),
        "off_description": (
            "Remove threshold sweep: use a fixed classification threshold of 0.5 "
            "instead of sweeping thresholds on the validation set."
        ),
    },
    {
        "name": "augmentation",
        "on_description": (
            "Augmentation ON: train with the data augmentation transforms."
        ),
        "off_description": (
            "Remove data augmentation: train without any augmentation transforms. "
            "Apply only basic normalisation / resize."
        ),
    },
]


def generate_factorial_variants(factors: list[dict]) -> list[dict]:
    """Build the 2^K factorial grid, minus the all-on baseline cell.

    Each factor is toggled on (1) or off (0). ``sorted(itertools.product(...))``
    yields a deterministic ordering of the 2^K combinations so variant indices
    stay stable across restarts. The all-on combination is the baseline (the
    current best pipeline) and is skipped, leaving 2^K - 1 variants.

    Returns a list of dicts with keys: ``name``, ``factors`` (factor name -> 0/1),
    ``description`` (only what changed from the all-on baseline), and
    ``is_factorial=True``.
    """
    factor_names = [f["name"] for f in factors]
    off_descriptions = {f["name"]: f["off_description"] for f in factors}

    variants: list[dict] = []
    for combo in sorted(itertools.product((0, 1), repeat=len(factors))):
        # Skip the all-on baseline (every factor enabled).
        if all(v == 1 for v in combo):
            continue

        factor_values = dict(zip(factor_names, combo))
        name = "_".join(
            f"{fname}={'on' if factor_values[fname] else 'off'}"
            for fname in factor_names
        )
        # Describe only the changes relative to the all-on baseline: list the
        # off_description of every factor that is turned off.
        changes = [
            off_descriptions[fname]
            for fname in factor_names
            if factor_values[fname] == 0
        ]
        variants.append({
            "name": name,
            "factors": factor_values,
            "description": " ".join(changes),
            "is_factorial": True,
        })

    return variants
