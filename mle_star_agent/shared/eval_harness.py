"""Small helpers for documenting evaluator slice consistency."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _count_predictions(predictions: Sequence[Mapping[str, Any]] | None) -> int | None:
    if predictions is None:
        return None
    return len(predictions)


def _metrics_counts(metrics: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(metrics, Mapping):
        return None, None
    ng = metrics.get("ng_count")
    g = metrics.get("g_count")
    return (int(ng) if ng is not None else None, int(g) if g is not None else None)


def _slice_text(split_name: str, sample_count: int | None, ng: int | None, g: int | None) -> str:
    count_text = "unknown samples" if sample_count is None else f"{sample_count} samples"
    ng_text = "?" if ng is None else str(ng)
    g_text = "?" if g is None else str(g)
    return f"{split_name} test: {count_text} (NG={ng_text}, G={g_text})"


def compare_eval_slices(
    *,
    baseline_metrics: Mapping[str, Any],
    baseline_predictions: Sequence[Mapping[str, Any]] | None,
    ablation_results: Sequence[Mapping[str, Any]],
    ablation_split_name: str,
) -> dict[str, Any]:
    """Compare the held-out slices used by a retrained baseline and ablations.

    The baseline side usually records its split name and per-class counts in a
    metrics JSON plus one prediction row per held-out sample. Ablation records
    store per-class counts in their METRICS payload; their lineage hash proves
    the split file but not the human-readable name, so the caller supplies it.
    """
    baseline_split = str(baseline_metrics.get("split") or "unknown")
    baseline_count = _count_predictions(baseline_predictions)
    baseline_ng, baseline_g = _metrics_counts(baseline_metrics)

    first_ablation_metrics = None
    first_ablation_lineage = None
    for result in ablation_results:
        metrics = result.get("metrics") if isinstance(result, Mapping) else None
        if isinstance(metrics, Mapping):
            first_ablation_metrics = metrics
            first_ablation_lineage = result.get("lineage")
            break

    ablation_ng, ablation_g = _metrics_counts(first_ablation_metrics)
    ablation_count = (
        ablation_ng + ablation_g
        if ablation_ng is not None and ablation_g is not None
        else None
    )

    same = (
        baseline_split == ablation_split_name
        and baseline_count == ablation_count
        and baseline_ng == ablation_ng
        and baseline_g == ablation_g
    )
    return {
        "verdict": "same" if same else "different",
        "baseline_slice": _slice_text(
            baseline_split, baseline_count, baseline_ng, baseline_g
        ),
        "ablation_slice": _slice_text(
            ablation_split_name, ablation_count, ablation_ng, ablation_g
        ),
        "ablation_lineage": first_ablation_lineage or {},
    }
