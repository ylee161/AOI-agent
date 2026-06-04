"""Tests for semantic (TF-IDF) KB retrieval in the refinement planner.

The persistent AOI KB used to be retrieved by EXACT tag match only, backfilled by
recency. That missed past runs whose failure was RELATED but worded differently.
``retrieve_kb_records`` is now HYBRID:
  - exact tag matches keep priority (most-recent first), then
  - remaining top-K slots are filled by TF-IDF cosine similarity to the
    failure_mode over the non-matching records, then
  - if semantic ranking is not meaningful (too few records, no shared terms, or
    scikit-learn unavailable) it falls back to the original recency backfill.

Coverage:
  (a) the pure helper ``rank_records_by_similarity`` returns None on the
      fall-back conditions and ranks by meaning otherwise (legacy records too);
  (b) a record tagged with a related-but-not-exact phrase is surfaced by
      semantic rank where the old recency backfill would have picked a more
      recent but unrelated record;
  (c) exact-tag matches still take priority over semantic fills;
  (d) empty / single-record KBs fall back cleanly;
  (e) when the semantic helper returns None, the original recency backfill is
      used unchanged.
"""

import importlib
import unittest
from unittest import mock

from mle_star_agent.shared import kb_semantic

planner_agent = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.planner_agent"
)


class HelperFallbackTests(unittest.TestCase):
    def test_empty_query_returns_none(self):
        records = [{"plan": "a overlap"}, {"plan": "b threshold"}]
        self.assertIsNone(kb_semantic.rank_records_by_similarity(records, ""))
        self.assertIsNone(kb_semantic.rank_records_by_similarity(records, "   "))

    def test_fewer_than_two_records_returns_none(self):
        self.assertIsNone(kb_semantic.rank_records_by_similarity([], "overlap"))
        self.assertIsNone(
            kb_semantic.rank_records_by_similarity([{"plan": "overlap"}], "overlap")
        )

    def test_no_shared_terms_returns_none(self):
        # Query terms appear in no record -> no signal -> caller keeps recency.
        records = [
            {"plan": "raise threshold", "tags": ["threshold_collapse"]},
            {"plan": "tune learning rate", "tags": ["lr_too_high"]},
        ]
        self.assertIsNone(
            kb_semantic.rank_records_by_similarity(records, "g_ng_overlap")
        )

    def test_sklearn_unavailable_returns_none(self):
        records = [{"plan": "a overlap"}, {"plan": "b threshold"}]
        # Force the lazy sklearn import to fail.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("simulated missing sklearn")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=fake_import):
            self.assertIsNone(
                kb_semantic.rank_records_by_similarity(records, "overlap")
            )

    def test_ranks_related_record_first_including_legacy(self):
        records = [
            # legacy record (no tags/code_diff) that is topically related
            {"plan": "reduce ng region overlap between gerber and ng"},
            {"plan": "raise decision threshold", "tags": ["threshold_collapse"]},
        ]
        order = kb_semantic.rank_records_by_similarity(records, "g_ng_overlap")
        self.assertIsNotNone(order)
        self.assertEqual(order[0], 0)  # the overlap record outranks threshold


class HybridRetrievalTests(unittest.TestCase):
    def test_semantic_surfaces_related_where_recency_would_miss(self):
        # Append order: related record FIRST, unrelated distractor MORE RECENT.
        records = [
            {
                "plan": "reduce ng region overlap",
                "tags": ["ng_region_overlap", "regressed"],
            },
            {
                "plan": "raise decision threshold",
                "tags": ["threshold_collapse", "regressed"],
            },
        ]

        # Old behavior (pure recency backfill) would pick the most-recent record,
        # i.e. the unrelated threshold one.
        self.assertEqual(records[-1]["plan"], "raise decision threshold")

        selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=1)
        self.assertEqual([r["plan"] for r in selected], ["reduce ng region overlap"])

    def test_exact_match_takes_priority_then_semantic_fills(self):
        records = [
            {"plan": "exact overlap fix", "tags": ["g_ng_overlap", "improved"]},
            {"plan": "reduce ng region overlap", "tags": ["ng_region_overlap"]},
            {"plan": "raise decision threshold", "tags": ["threshold_collapse"]},
        ]

        selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=2)
        plans = [r["plan"] for r in selected]

        # Exact tag match is included...
        self.assertIn("exact overlap fix", plans)
        # ...and the remaining slot goes to the semantically-related record, not
        # the more-recent-but-unrelated threshold one.
        self.assertIn("reduce ng region overlap", plans)
        self.assertNotIn("raise decision threshold", plans)

    def test_empty_kb_returns_empty(self):
        self.assertEqual(planner_agent.retrieve_kb_records([], "g_ng_overlap", k=6), [])

    def test_single_record_falls_back_cleanly(self):
        records = [{"plan": "lonely", "tags": ["threshold_collapse"]}]
        selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=6)
        self.assertEqual([r["plan"] for r in selected], ["lonely"])

    def test_recency_fallback_when_semantic_returns_none(self):
        # No shared terms -> helper returns None -> original recency backfill.
        records = [
            {"plan": "tune learning rate", "tags": ["lr_too_high"]},
            {"plan": "raise decision threshold", "tags": ["threshold_collapse"]},
        ]
        selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=1)
        # Recency backfill keeps the most-recent non-matching record.
        self.assertEqual([r["plan"] for r in selected], ["raise decision threshold"])

    def test_forced_none_helper_uses_recency_backfill(self):
        records = [
            {"plan": "reduce ng region overlap", "tags": ["ng_region_overlap"]},
            {"plan": "raise decision threshold", "tags": ["threshold_collapse"]},
        ]
        with mock.patch.object(
            planner_agent, "rank_records_by_similarity", return_value=None
        ):
            selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=1)
        # Pure recency: most-recent record, regardless of semantic relatedness.
        self.assertEqual([r["plan"] for r in selected], ["raise decision threshold"])


if __name__ == "__main__":
    unittest.main()
