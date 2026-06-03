from google.adk.agents import LoopAgent

from mle_star_agent import config
from mle_star_agent.phases.phase3_ensemble.ensemble_coder_agent import ensemble_coder_agent
from mle_star_agent.phases.phase3_ensemble.ensemble_evaluator_agent import ensemble_evaluator_agent

# Ensemble loop: ensemble_coder → ensemble_evaluator, up to ENSEMBLE_LOOP_MAX iterations.
# ensemble_evaluator_agent escalates (via FunctionTool) when the iteration cap is reached
# or no improvement is detected; LoopAgent.max_iterations is a safety ceiling.
phase3_ensemble = LoopAgent(
    name="phase3_ensemble",
    description=(
        "Phase 3 Ensemble: iteratively generates and evaluates ensemble scripts that combine "
        "the best Phase 2 pipeline with complementary approaches. Exits when "
        "ensemble_iteration reaches ENSEMBLE_LOOP_MAX or no improvement is detected."
    ),
    max_iterations=config.ENSEMBLE_LOOP_MAX,
    sub_agents=[ensemble_coder_agent, ensemble_evaluator_agent],
)

__all__ = ["phase3_ensemble"]
