from google.adk.agents import SequentialAgent

from mle_star_agent.phases.phase4_submission.submission_agent import submission_agent

phase4_submission = SequentialAgent(
    name="phase4_submission",
    description=(
        "Phase 4 Submission: runs the final selected AOI script on the test split, "
        "writes final_metrics to state, and saves checkpoints/submission.json."
    ),
    sub_agents=[submission_agent],
)

__all__ = ["phase4_submission", "submission_agent"]
