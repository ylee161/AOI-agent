"""TF-IDF semantic ranking for persistent-KB record retrieval.

Pure, dependency-light, and NEVER raises. Used by the refinement planner to rank
past AOI knowledge-base records by *meaning* (not just exact tag match) against
the current failure context. Whenever ranking is not meaningful — too few
records, empty corpus, no semantic signal, or scikit-learn unavailable — it
returns ``None`` so callers fall back to their existing exact/recency behavior
unchanged.

This module only READS records. The append-only KB file format is untouched.
scikit-learn (already a project dependency) is imported lazily so its absence
degrades gracefully instead of breaking import.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Bounded code_diff slice so a long diff cannot dominate the bag-of-words.
_DIFF_EXCERPT_CHARS = 400


def _record_text(record) -> str:
    """Flatten a KB record into a single bag-of-words string.

    Combines plan + tags (+ legacy label) + a bounded code_diff excerpt so legacy
    records lacking tags/code_diff still contribute via their plan text.
    Underscores are split to spaces so categorical tags like ``g_ng_overlap``
    tokenize into ``g``/``ng``/``overlap`` and can overlap with related-but-
    differently-worded failures. Never raises.
    """
    if not isinstance(record, dict):
        return ""
    parts: list[str] = []
    plan = record.get("plan")
    if plan:
        parts.append(str(plan))
    tags = record.get("tags")
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags if t)
    label = record.get("label")  # legacy records: "success"/"failure"
    if label:
        parts.append(str(label))
    diff = record.get("code_diff")
    if diff:
        parts.append(str(diff)[:_DIFF_EXCERPT_CHARS])
    return " ".join(parts).replace("_", " ")


def rank_records_by_similarity(records, query):
    """Rank ``records`` by TF-IDF cosine similarity to ``query`` (descending).

    Returns a list of integer indices into ``records`` ordered most- to
    least-similar (ties broken by original append order). Returns ``None``
    (signaling the caller to fall back) when ranking is not meaningful:
      - fewer than 2 records,
      - empty/blank query,
      - the record corpus is entirely empty text,
      - no record shares any term with the query (max similarity == 0),
      - scikit-learn is unavailable, or
      - anything at all goes wrong.
    Never raises.
    """
    try:
        if not query or not str(query).strip():
            return None
        record_list = list(records or [])
        if len(record_list) < 2:
            return None
        corpus = [_record_text(r) for r in record_list]
        if not any(text.strip() for text in corpus):
            return None

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            return None

        query_text = str(query).replace("_", " ")
        vectorizer = TfidfVectorizer()
        # Empty vocabulary (all single chars / stop words) raises ValueError,
        # caught below -> None.
        matrix = vectorizer.fit_transform(corpus + [query_text])
        sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()

        # No record shares any term with the query: semantic gives no signal,
        # so let the caller keep its recency fallback rather than reorder.
        if float(sims.max()) <= 0.0:
            return None

        # Descending similarity; ties keep original (append) order via index.
        return sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
    except Exception:
        logger.debug("kb_semantic ranking failed; falling back", exc_info=True)
        return None
