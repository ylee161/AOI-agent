import logging
import warnings

# Suppress LiteLLM's startup warnings about optional AWS providers (botocore/sagemaker)
# not being installed — we're using DeepSeek, not Bedrock/SageMaker.
warnings.filterwarnings("ignore", message=".*could not pre-load bedrock.*")
warnings.filterwarnings("ignore", message=".*could not pre-load sagemaker.*")

# Suppress Google ADK experimental-feature warnings — these features are stable
# enough for production use in ADK 2.x despite the experimental label.
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*BaseCredentialService.*")
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*JSON_SCHEMA_FOR_FUNC_DECL.*")

# Reduce LiteLLM chattiness: suppress INFO-level "LiteLLM completion() model=..." lines
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

from mle_star_agent.agent import root_agent

__all__ = ["root_agent"]
