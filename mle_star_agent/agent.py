from google.adk.agents import SequentialAgent

from mle_star_agent import config
from mle_star_agent.phases.phase1_init import phase1_init
from mle_star_agent.phases.phase2_refinement import phase2_refinement
from mle_star_agent.phases.phase3_ensemble import phase3_ensemble
from mle_star_agent.phases.phase4_submission import phase4_submission
from mle_star_agent.phases.retry_loop_agent import RetryLoopAgent

# RetryLoopAgent runs Phase 2 → Phase 3 → Phase 4 and retries the whole cycle
# if the submission fails the relaxed minimum acceptance criteria. It strips
# escalate signals from Phase 2 and Phase 3 so their internal loop exits do
# not prematurely kill the retry cycle (see retry_loop_agent.py for details).
refinement_submission_loop = RetryLoopAgent(
    name="refinement_submission_loop",
    description=(
        "Runs Phase 2 refinement, Phase 3 ensemble search, and Phase 4 submission "
        "in a retry loop. Exits when the submission passes the relaxed minimum "
        f"acceptance criteria or {config.SUBMISSION_RETRY_MAX} retry cycle(s) are exhausted."
    ),
    sub_agents=[phase2_refinement, phase3_ensemble, phase4_submission],
)

root_agent = SequentialAgent(
    name="mle_star_root",
    description=(
        "Root MLE-STAR AOI pipeline: runs Phase 1 initialisation once, then runs "
        "the refinement–ensemble–submission loop until the relaxed minimum is met "
        f"or {config.SUBMISSION_RETRY_MAX} retry cycle(s) are exhausted."
    ),
    sub_agents=[
        phase1_init,
        refinement_submission_loop,
    ],
)
