from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.phases.phase1_init.baseline_coder_agent import baseline_coder_agent
from mle_star_agent.phases.phase1_init.candidate_evaluator_agent import candidate_evaluator_agent
from mle_star_agent.phases.phase1_init.merger_agent import merger_agent
from mle_star_agent.phases.phase1_init.retriever_agent import retriever_agent
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint


def check_phase1_done_fn(tool_context) -> str:
    """Escalate immediately if Phase 1 outputs already exist on disk.

    Checks for candidate_scores.json and L0.json. If both are present, restores
    the key state variables that downstream phases expect and sets escalate so
    the rest of phase1_init is skipped entirely.
    """
    scores_done = checkpoint_exists(config.CKPT_CANDIDATE_SCORES)
    l0_done = checkpoint_exists(config.CKPT_L0)

    if scores_done and l0_done:
        # Restore state that merger_agent would normally write, so Phase 2 has
        # everything it needs without re-running Phase 1.
        l0 = load_checkpoint(config.CKPT_L0)
        tool_context.state["current_best_score"] = l0.get("L0_score", 0.0)
        tool_context.state["best_miss_rate"] = l0.get("L0_miss_rate", 1.0)
        tool_context.state["best_overkill_rate"] = l0.get("L0_overkill_rate", 1.0)
        tool_context.state["best_accuracy"] = l0.get("L0_accuracy", 0.0)
        tool_context.state["best_f1"] = l0.get("L0_f1", 0.0)
        tool_context.state["best_candidate_name"] = l0.get("best_candidate_name", "")
        tool_context.actions.escalate = True
        return (
            f"PHASE1_DONE: candidate_scores.json and L0.json already exist — "
            f"skipping Phase 1 entirely. Best candidate: {l0.get('best_candidate_name')} "
            f"(ng_recall={l0.get('L0_score')}, miss={l0.get('L0_miss_rate')}, "
            f"overkill={l0.get('L0_overkill_rate')})."
        )

    missing = []
    if not scores_done:
        missing.append("candidate_scores.json")
    if not l0_done:
        missing.append("L0.json")
    return f"PHASE1_NEEDED: missing {missing} — running Phase 1 normally."


_phase1_skip_gate = LlmAgent(
    name="phase1_skip_gate",
    model=config.MODEL,
    description="Skips Phase 1 entirely if candidate_scores.json and L0.json already exist.",
    instruction=(
        "Call `check_phase1_done_fn` immediately. "
        "If it returns PHASE1_DONE, report the loaded scores and stop — Phase 1 is skipped. "
        "If it returns PHASE1_NEEDED, report which files are missing and stop — Phase 1 will run."
    ),
    tools=[FunctionTool(func=check_phase1_done_fn)],
    include_contents="none",
)

phase1_init = SequentialAgent(
    name="phase1_init",
    description=(
        "Phase 1 Initialisation: skipped entirely on restart if outputs already exist. "
        "Otherwise retrieves candidate models via web search, splits data, generates and "
        "evaluates candidate training scripts, and selects L0 while initialising all Phase 2 "
        "loop state."
    ),
    sub_agents=[
        _phase1_skip_gate,
        retriever_agent,
        baseline_coder_agent,
        candidate_evaluator_agent,
        merger_agent,
    ],
)

__all__ = ["phase1_init"]
