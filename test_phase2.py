"""
test_phase2.py — Regression suite for the context-bloat fix (CONTEXT_FIX_PLAN.md).

Runs fast and offline: NO DeepSeek API calls and NO training. The end-to-end
bounded-growth proof uses a fake ADK model through the real Runner, so it exercises
the genuine ADK content-assembly path that include_contents="none" depends on.

What it guards against (per §8 of the plan):
  1. CONFIG       — every agent stays include_contents="none"; the flash downgrade
                    (lite_mode_callback) is never re-wired; reasoning agents keep
                    budget_stop_callback.
  2. LOAD TOOLS   — the load-on-demand tools return the script and do NOT leak large
                    artifacts (full population/candidate scripts).
  3. TIER 2       — rotation-lock counts over the FULL tried_approaches list (not the
                    truncated prompt view); validation cache caps; error-samples cap.
  4. TIER 0       — budget_stop_callback halts over budget / no-ops under it;
                    _reset_for_retry zeroes token_count.
  5. BOUNDED GROWTH — with include_contents="none", a looping agent's per-call input
                    size stays ~flat across iterations, while "default" grows. This is
                    the core regression test for the whole fix.

Run:  .venv/bin/python test_phase2.py
"""
import asyncio
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Tiny assertion harness (pytest is not installed in this venv)
# ---------------------------------------------------------------------------
_PASS, _FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  — {detail}"
    print(line)


class Ctx:
    """Mock ADK tool_context / callback_context: has .state and .actions.escalate."""
    def __init__(self, state=None):
        self.state = state if state is not None else {}
        self.actions = type("A", (), {"escalate": None})()


# ===========================================================================
# 1. CONFIG GUARDS
# ===========================================================================
def section_config():
    print("\n[1] CONFIG GUARDS")
    from google.adk.agents import LlmAgent
    from mle_star_agent.agent import root_agent

    seen = {}

    def walk(a):
        if isinstance(a, LlmAgent):
            seen[id(a)] = a
        for s in getattr(a, "sub_agents", []) or []:
            walk(s)
        for t in getattr(a, "tools", []) or []:
            inner = getattr(t, "agent", None)
            if inner is not None:
                walk(inner)

    walk(root_agent)
    agents = list(seen.values())

    bad = [a.name for a in agents if a.include_contents != "none"]
    check("all reachable agents include_contents='none'", not bad,
          f"{len(agents)} agents" if not bad else f"offenders: {bad}")

    def cbs(a):
        c = a.before_model_callback
        seq = c if isinstance(c, list) else [c] if c else []
        return [getattr(f, "__name__", "") for f in seq]

    lite = [a.name for a in agents if "lite_mode_callback" in cbs(a)]
    check("flash downgrade (lite_mode_callback) is NOT wired anywhere", not lite,
          "removed" if not lite else f"still wired: {lite}")

    # The 4 reasoning families that previously downgraded must now hard-stop instead.
    reasoning = [a for a in agents if a.name in (
        "refinement_coder_agent", "refinement_planner_agent", "error_analysis_agent",
    ) or a.name.startswith("ablation_step_")]
    missing = [a.name for a in reasoning if "budget_stop_callback" not in cbs(a)]
    check("reasoning agents wire budget_stop_callback", not missing,
          f"{len(reasoning)} reasoning agents" if not missing else f"missing: {missing}")

    # Load tools registered as the first tool on the agents that needed an input path.
    def first_tool(agent_name):
        ag = next(a for a in agents if a.name == agent_name)
        t = ag.tools[0]
        return getattr(getattr(t, "func", t), "__name__", str(t))

    check("refinement_coder first tool is load_refinement_context_fn",
          first_tool("refinement_coder_agent") == "load_refinement_context_fn")
    check("ensemble_coder first tool is load_ensemble_context_fn",
          first_tool("ensemble_coder_agent") == "load_ensemble_context_fn")
    check("evaluator first tool is load_current_script_fn",
          first_tool("evaluator_agent") == "load_current_script_fn")
    check("ensemble_evaluator first tool is load_ensemble_script_fn",
          first_tool("ensemble_evaluator_agent") == "load_ensemble_script_fn")


# ===========================================================================
# 2. LOAD TOOLS — return script, never leak large artifacts
# ===========================================================================
def section_load_tools():
    print("\n[2] LOAD TOOLS")
    from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import load_refinement_context_fn
    from mle_star_agent.phases.phase3_ensemble.ensemble_coder_agent import load_ensemble_context_fn
    from mle_star_agent.phases.phase2_refinement.evaluator_agent import load_current_script_fn
    from mle_star_agent.phases.phase3_ensemble.ensemble_evaluator_agent import load_ensemble_script_fn

    SENTINEL_SCRIPT = "import torch\n# THE-BEST-PIPELINE-SCRIPT\n"
    LEAK = "POPULATION-FULL-SCRIPT-SHOULD-NOT-LEAK"

    state = {
        "best_pipeline_script": SENTINEL_SCRIPT,
        "diagnosis_report": {"target_component": "stereo_fusion", "recommended_changes": "x"},
        "selected_refinement_strategy": "focal_loss: replace BCE",
        "best_overkill_rate": 0.15, "best_miss_rate": 0.02,
        "refinement_population": [
            {"script_sha256": "deadbeefcafe0001", "metrics": {"overkill_rate": 0.1},
             "archive_reason": "lower_overkill_candidate", "script": LEAK}
        ],
    }
    out = load_refinement_context_fn(Ctx(state))
    check("refinement bundle contains the best script", SENTINEL_SCRIPT.strip() in out)
    check("refinement bundle does NOT leak full population script", LEAK not in out)

    ens_state = {
        "best_pipeline_script": SENTINEL_SCRIPT,
        "current_best_score": 0.97, "ensemble_iteration": 0,
        "candidate_scores": [{"name": "resnet18", "metrics": {"ng_recall": 0.9}, "script": LEAK}],
    }
    out2 = load_ensemble_context_fn(Ctx(ens_state))
    check("ensemble bundle contains the best script", SENTINEL_SCRIPT.strip() in out2)
    check("ensemble bundle does NOT leak candidate script", LEAK not in out2)

    check("evaluator load returns current_script",
          load_current_script_fn(Ctx({"current_script": "X=1"})) == "X=1")
    check("evaluator load errors cleanly when missing",
          load_current_script_fn(Ctx({})).startswith("ERROR"))
    check("ensemble_evaluator load returns ensemble_script",
          load_ensemble_script_fn(Ctx({"ensemble_script": "Y=2"})) == "Y=2")

    # Disk fallback must not crash when nothing is available.
    import mle_star_agent.config as config
    saved = config.CKPT_BEST_PIPELINE
    try:
        config.CKPT_BEST_PIPELINE = Path(tempfile.gettempdir()) / "definitely_missing_best_pipeline.json"
        out3 = load_refinement_context_fn(Ctx({}))
        check("refinement load survives missing script/checkpoint", "BEST_PIPELINE_SCRIPT" in out3)
    finally:
        config.CKPT_BEST_PIPELINE = saved


# ===========================================================================
# 3. TIER 2 — summarization safety, caps
# ===========================================================================
def section_tier2():
    print("\n[3] TIER 2")
    import mle_star_agent.config as config
    from mle_star_agent.phases.phase2_refinement.planner_agent import (
        _tried_approaches_view, _failed_attempt_count_for_target,
    )
    from mle_star_agent.phases.phase2_refinement.error_analysis_agent import load_latest_error_analysis_fn
    from mle_star_agent.guards.code_validator_agent import store_validation_cache_fn

    # >K failed attempts on one target — the rotation-lock count must see them ALL,
    # even though the prompt view only shows the last K.
    N = config.TRIED_APPROACHES_RECENT_K + 4
    tried = [{
        "outer": 0, "inner": i, "target_component": "stereo_fusion",
        "changes_summary": f"change {i}",
        "strategy_fingerprint": {"target_component": "stereo_fusion", "mechanism_class": f"m{i}"},
        "result": {"improved": False}, "failure_reason": "recall_regression",
    } for i in range(N)]

    view = _tried_approaches_view(tried)
    agg = json.loads([l for l in view.split("\n") if l.startswith("TRIED_APPROACHES_AGGREGATE")][0].split(": ", 1)[1])
    check("aggregate counts ALL attempts", agg["total_attempts"] == N, f"{agg['total_attempts']} == {N}")
    cnt = _failed_attempt_count_for_target(tried, "stereo_fusion")
    check("rotation-lock counts over FULL list (not truncated to K)", cnt == N,
          f"count={cnt}, expected {N} (truncating to {config.TRIED_APPROACHES_RECENT_K} would undercount)")

    # Validation cache FIFO cap.
    st = {}
    ctx = Ctx(st)
    total = config.VALIDATION_CACHE_MAX + 25
    for i in range(total):
        store_validation_cache_fn(ctx, f"script_{i}", "VALIDATED")
    import hashlib
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    cache = st["_validation_cache"]
    check("validation cache capped", len(cache) == config.VALIDATION_CACHE_MAX, f"size={len(cache)}")
    check("validation cache evicts oldest, keeps newest",
          h("script_0") not in cache and h(f"script_{total-1}") in cache)

    # Error-sample cap in the load tool's prompt view.
    cap = config.ERROR_ANALYSIS_SAMPLE_CAP
    ev = {
        "available": True, "fp_count": cap + 15, "fn_count": 3,
        "fp_samples": [{"sample_id": f"s{i}", "ng_probability": 0.4} for i in range(cap + 15)],
        "fn_samples": [{"sample_id": "f1"}],
        "probability_summary": {"NG_prob_mean": 0.6}, "metrics_consistency": {"matches_metrics": True},
    }
    out = load_latest_error_analysis_fn(Ctx({"latest_error_analysis": ev, "outer_iteration": 0, "inner_iteration": 1}))
    body = json.loads(out.split("\n", 1)[1])
    check("error-analysis FP samples capped", len(body["fp_samples"]) == cap, f"{len(body['fp_samples'])} == {cap}")
    check("error-analysis flags truncation", body["fp_samples_truncated"] is True)
    check("error-analysis still surfaces probability_summary", body["probability_summary"]["NG_prob_mean"] == 0.6)


# ===========================================================================
# 4. TIER 0 — hard stop instead of downgrade; retry token reset
# ===========================================================================
def section_tier0():
    print("\n[4] TIER 0")
    import mle_star_agent.config as config
    from mle_star_agent.shared.callbacks import budget_stop_callback

    over = Ctx({"token_count": config.TOKEN_LITE_THRESHOLD + 1})
    budget_stop_callback(over, None)
    check("budget_stop escalates over threshold", over.actions.escalate is True)
    check("budget_stop sets stop_outer_loop (container-agnostic stop path)",
          over.state.get("stop_outer_loop") is True)

    under = Ctx({"token_count": config.TOKEN_LITE_THRESHOLD - 1})
    r = budget_stop_callback(under, None)
    check("budget_stop is a no-op under threshold",
          under.actions.escalate is None and r is None and under.state.get("stop_outer_loop") is None)

    # _reset_for_retry must zero token_count (in state AND the checkpoint).
    from mle_star_agent.phases import retry_loop_agent as rla
    saved = (config.CHECKPOINT_DIR, config.CKPT_BEST_PIPELINE, config.CKPT_CANDIDATE_SCORES)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config.CHECKPOINT_DIR = tmp
        config.CKPT_BEST_PIPELINE = tmp / "best_pipeline.json"
        config.CKPT_CANDIDATE_SCORES = tmp / "candidate_scores.json"
        try:
            state = {"token_count": 8_000_000, "best_pipeline_script": "x=1",
                     "current_best_score": 0.9, "refinement_population": []}
            rla._reset_for_retry(state, attempt=1)
            check("_reset_for_retry zeroes token_count in state", state["token_count"] == 0)
            written = json.loads(config.CKPT_BEST_PIPELINE.read_text())
            check("_reset_for_retry zeroes token_count in checkpoint", written["token_count"] == 0,
                  f"checkpoint token_count={written['token_count']}")
        finally:
            config.CHECKPOINT_DIR, config.CKPT_BEST_PIPELINE, config.CKPT_CANDIDATE_SCORES = saved


# ===========================================================================
# 5. BOUNDED GROWTH — the core regression test (real ADK Runner, fake model)
# ===========================================================================
def section_bounded_growth():
    print("\n[5] BOUNDED GROWTH (real ADK content-assembly, fake model)")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as gt

    BIG = "Z" * 6000  # simulate a large per-turn payload (e.g. a generated script)

    def make_recorder():
        sizes = []

        class FakeLlm(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                payload = json.dumps([c.model_dump() for c in (llm_request.contents or [])], default=str)
                sizes.append(len(payload))
                yield LlmResponse(content=gt.Content(role="model", parts=[gt.Part(text="ok")]))

        return FakeLlm, sizes

    async def run_three_turns(include_contents):
        FakeLlm, sizes = make_recorder()
        agent = LlmAgent(
            name=f"probe_{include_contents}",
            model=FakeLlm(model="fake"),
            include_contents=include_contents,
            instruction="Reply with ok.",
        )
        ss = InMemorySessionService()
        await ss.create_session(app_name="t", user_id="u", session_id="s", state={})
        runner = Runner(agent=agent, app_name="t", session_service=ss)
        for _ in range(3):
            msg = gt.Content(role="user", parts=[gt.Part(text=BIG)])
            async for _ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
                pass
        return sizes

    none_sizes = asyncio.run(run_three_turns("none"))
    default_sizes = asyncio.run(run_three_turns("default"))

    print(f"      none    per-call input sizes: {none_sizes}")
    print(f"      default per-call input sizes: {default_sizes}")

    check("captured 3 calls for each mode", len(none_sizes) == 3 and len(default_sizes) == 3)

    none_growth = none_sizes[-1] - none_sizes[0]
    default_growth = default_sizes[-1] - default_sizes[0]

    check("default mode grows with history (sanity check on the probe)",
          default_sizes[2] > default_sizes[0] + 4000,
          f"grew by {default_growth} chars over 3 turns")
    check("none mode stays bounded across iterations",
          none_growth < 1500,
          f"none grew {none_growth} chars vs default {default_growth} chars")
    check("none growth is far smaller than default growth",
          none_growth * 3 < default_growth,
          f"{none_growth} (none) vs {default_growth} (default)")


# ===========================================================================
def main():
    print("=" * 64)
    print("PHASE 2 / CONTEXT-FIX REGRESSION SUITE  (offline, no API, no training)")
    print("=" * 64)
    section_config()
    section_load_tools()
    section_tier2()
    section_tier0()
    section_bounded_growth()

    print("\n" + "=" * 64)
    total = len(_PASS) + len(_FAIL)
    print(f"RESULT: {len(_PASS)}/{total} checks passed")
    if _FAIL:
        print("FAILED:")
        for n in _FAIL:
            print(f"  - {n}")
        print("=" * 64)
        return 1
    print("ALL CHECKS PASSED — context-fix invariants hold.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
