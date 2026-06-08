from google.adk.agents import loop_agent

from mle_star_agent import config
from mle_star_agent.phases.phase2_refinement.ablation_agent import ablation_agent
from mle_star_agent.phases.phase2_refinement.diagnosis_agent import diagnosis_agent
from mle_star_agent.phases.phase2_refinement.error_analysis_agent import (
    error_analysis_agent,
    error_analysis_gate_agent,
)
from mle_star_agent.phases.phase2_refinement.evaluator_agent import evaluator_agent
from mle_star_agent.phases.phase2_refinement.planner_agent import (
    refinement_planner_agent,
    strategy_gate_agent,
)
from mle_star_agent.phases.phase2_refinement.reflexion_agent import reflexion_agent
from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import refinement_coder_agent

# Inner loop: error_analysis_gate → reflexion → planner → strategy_gate → refinement_coder → evaluator → error_analysis,
# up to INNER_LOOP_MAX iterations.
# evaluator_agent escalates (via FunctionTool) when the inner cap is reached
# or an early-stop condition fires; LoopAgent.max_iterations is a safety ceiling.
inner_loop_agent = loop_agent.LoopAgent(
    name="inner_loop_agent",
    description=(
        "Inner refinement loop: iteratively improves the pipeline script targeting "
        "the component identified by the diagnosis agent. Exits when inner_iteration "
        "reaches INNER_LOOP_MAX or an early-stop condition fires."
    ),
    max_iterations=config.INNER_LOOP_MAX,
    sub_agents=[
        error_analysis_gate_agent,
        reflexion_agent,
        refinement_planner_agent,
        strategy_gate_agent,
        refinement_coder_agent,
        evaluator_agent,
        error_analysis_agent,
    ],
)

# Outer loop: ablation → diagnosis → inner_loop, up to OUTER_LOOP_MAX iterations.
# ablation_agent's flag_checker escalates when stop_outer_loop is True;
# LoopAgent.max_iterations is a safety ceiling.
phase2_refinement = loop_agent.LoopAgent(
    name="phase2_refinement",
    description=(
        "Phase 2 Refinement: for each outer iteration, runs ablation to identify the "
        "weakest component, diagnoses it, then runs the inner refinement loop to improve "
        "it. Exits when outer_iteration reaches OUTER_LOOP_MAX, the token budget is "
        "exhausted, or an already-accepted pipeline reaches no-improvement patience."
    ),
    max_iterations=config.OUTER_LOOP_MAX,
    sub_agents=[ablation_agent, diagnosis_agent, inner_loop_agent],
)

__all__ = ["phase2_refinement", "inner_loop_agent"]
