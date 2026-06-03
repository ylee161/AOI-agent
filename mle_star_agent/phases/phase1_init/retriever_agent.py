"""A_retriever — paper-faithful web-search model retrieval (MLE-STAR Section 3.1).

MLE-STAR replaces hardcoded/case-bank model choices with a search-as-a-tool step:
the agent issues M=4 web searches for the given task and retrieves, per result, a
``{model description, example code}`` pair (paper Eq. 1, retrieving four candidates).
Filtering is empirical — every retrieved candidate is later turned into a training
script and run; the val score is the filter. This module does NOT reason about
small-data fit (the existing ``small_data_strategy_validator`` is the static guard).

Design notes
------------
* The runtime model is DeepSeek (LiteLlm), which has no native browsing, and no
  search API key is assumed present. ``web_search`` is therefore *pluggable*:
  a real search API is used when its key is in the environment
  (``TAVILY_API_KEY`` or ``SERPER_API_KEY``); otherwise it falls back to the
  keyless DuckDuckGo HTML endpoint; on any failure it returns a clear
  "search unavailable" signal that tells the LLM to lean on knowledge of
  *current* (2024-2025) small-data backbones — never a legacy ResNet18-only set.
* Extraction of ``{model_name, description, example_code}`` is done by the LLM
  (an ``LlmAgent``), exactly as the paper's A_retriever does — a plain
  ``FunctionTool`` cannot synthesise example code from raw search snippets.
* ``format_candidate_block`` lives in ``baseline_coder_agent`` (where the
  consuming function also needs it); this module imports it so the two return
  strings stay identical. The dependency is one-directional
  (retriever -> baseline_coder) to avoid a circular import.
"""

import html
import json
import logging
import os
import re

import httpx
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.phases.phase1_init.baseline_coder_agent import (
    ensure_data_split_fn,
    format_candidate_block,
)
from mle_star_agent.shared.callbacks import count_tokens_callback, rate_limit_retry_callback
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT = 20.0
_MAX_RESULTS = 6
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Pluggable web search (real API -> keyless DuckDuckGo -> graceful signal)
# ---------------------------------------------------------------------------

def _search_tavily(query: str, api_key: str) -> list[dict]:
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": _MAX_RESULTS},
        timeout=_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "snippet": r.get("content", ""), "url": r.get("url", "")}
        for r in data.get("results", [])
    ]


def _search_serper(query: str, api_key: str) -> list[dict]:
    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": _MAX_RESULTS},
        timeout=_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in data.get("organic", [])[:_MAX_RESULTS]:
        out.append({"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")})
    return out


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _search_duckduckgo(query: str) -> list[dict]:
    """Keyless fallback against DuckDuckGo's HTML endpoint."""
    resp = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=_SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    text = resp.text
    results: list[dict] = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
        results.append({"title": _strip_tags(m.group(2)), "snippet": "", "url": m.group(1)})
        if len(results) >= _MAX_RESULTS:
            break
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', text, re.S)
    for i, s in enumerate(snippets[: len(results)]):
        results[i]["snippet"] = _strip_tags(s)
    return results


def web_search(query: str) -> str:
    """Run ONE web search and return the top results as readable text.

    Pass a single search query string. Returns a formatted list of
    ``[n] title / snippet / url`` lines for the LLM to read. Uses a real search
    API when ``TAVILY_API_KEY`` or ``SERPER_API_KEY`` is set, otherwise the
    keyless DuckDuckGo endpoint. On failure returns a ``SEARCH_UNAVAILABLE``
    marker — in that case rely on current (2024-2025) small-data backbones and
    do NOT default to a legacy ResNet18-only choice.
    """
    try:
        tavily = os.environ.get("TAVILY_API_KEY")
        serper = os.environ.get("SERPER_API_KEY")
        if tavily:
            results, source = _search_tavily(query, tavily), "tavily"
        elif serper:
            results, source = _search_serper(query, serper), "serper"
        else:
            results, source = _search_duckduckgo(query), "duckduckgo"
    except Exception as exc:  # network/parse failure — degrade gracefully
        logger.warning("web_search failed for %r: %s", query, exc)
        return (
            f"SEARCH_UNAVAILABLE for query {query!r} ({type(exc).__name__}). "
            "No live results returned. Use your knowledge of CURRENT (2024-2025) "
            "small-data image-classification backbones (e.g. EfficientNet, ConvNeXt, "
            "ViT/DeiT families) — do NOT default to a legacy ResNet18-only choice."
        )

    logger.info("web_search via %s for %r — %d result(s)", source, query, len(results))

    if not results:
        return (
            f"SEARCH_EMPTY for query {query!r}. No results returned. Use your knowledge "
            "of current small-data image-classification backbones; avoid a legacy "
            "ResNet18-only choice."
        )

    lines = [f"SEARCH RESULTS ({source}) for: {query}"]
    for i, r in enumerate(results, 1):
        snippet = r.get("snippet", "").strip()
        lines.append(f"[{i}] {r.get('title', '').strip()}")
        if snippet:
            lines.append(f"    {snippet}")
        lines.append(f"    {r.get('url', '').strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State / checkpoint helpers
# ---------------------------------------------------------------------------

def check_retriever_needed_fn(tool_context) -> str:
    """Skip retrieval if candidate scripts are already generated (resume support)."""
    if tool_context.state.get("retrieved_candidates"):
        return "ALREADY_RETRIEVED: state['retrieved_candidates'] is populated — skip searching."
    if checkpoint_exists(config.CKPT_CANDIDATE_SCRIPTS):
        data = load_checkpoint(config.CKPT_CANDIDATE_SCRIPTS)
        if len(data.get("scripts", [])) >= 3:
            return (
                "SCRIPTS_ALREADY_BUILT: candidate_scripts.json already has >=3 scripts — "
                "retrieval is unnecessary on this restart. Stop without searching."
            )
    return "RETRIEVAL_NEEDED: proceed with data-split check, then 4 web searches."


def store_retrieved_candidates(tool_context, candidates_json: str) -> str:
    """Persist the retrieved candidates and return the formatted candidate menu.

    Pass a JSON array of exactly 4 objects, each with keys ``model_name``,
    ``description`` (1-2 sentences on fit), and ``example_code`` (minimal PyTorch
    snippet that loads/instantiates the model). Writes
    ``state['retrieved_candidates']`` and returns the formatted menu.
    """
    try:
        candidates = json.loads(candidates_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return (
            f"ERROR: candidates_json is not valid JSON ({exc}). Pass a JSON list of 4 "
            "objects with keys model_name, description, example_code."
        )
    if not isinstance(candidates, list) or not candidates:
        return "ERROR: expected a non-empty JSON list of candidate objects."

    cleaned = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        name = str(c.get("model_name", "")).strip()
        if not name:
            continue
        cleaned.append(
            {
                "model_name": name,
                "description": str(c.get("description", "")).strip(),
                "example_code": str(c.get("example_code", "")).strip(),
            }
        )
    if not cleaned:
        return "ERROR: no candidate had a model_name — provide model_name for each entry."

    tool_context.state["retrieved_candidates"] = cleaned
    modality = tool_context.state.get("input_modality", "stereo")
    warn = "" if len(cleaned) == 4 else f"(NOTE: stored {len(cleaned)} candidates; the paper retrieves 4.) "
    return (
        f"STORED {len(cleaned)} retrieved candidate(s) into state['retrieved_candidates']. {warn}\n\n"
        + format_candidate_block(cleaned, modality)
    )


_ensure_data_split_tool = FunctionTool(func=ensure_data_split_fn)
_check_retriever_needed_tool = FunctionTool(func=check_retriever_needed_fn)
_web_search_tool = FunctionTool(func=web_search)
_store_retrieved_candidates_tool = FunctionTool(func=store_retrieved_candidates)

# ---------------------------------------------------------------------------
# Agent instruction
# ---------------------------------------------------------------------------

_RETRIEVER_INSTRUCTION = """You are A_retriever in the MLE-STAR AOI inspection pipeline.

Per MLE-STAR Section 3.1, you use web search as a tool to retrieve M=4 effective,
state-of-the-art model candidates for the current task. For each candidate you return a
DESCRIPTION (why it fits) and EXAMPLE CODE (how to instantiate it in PyTorch) — the
downstream coder may be unfamiliar with the model and needs the snippet to write runnable
code. You do NOT judge small-data fit or rank the models; filtering is empirical later
(each candidate is trained and scored). Just search, extract, and store.

---
## STEP 0 — Resume check
Call `check_retriever_needed_fn`.
- If it returns ALREADY_RETRIEVED or SCRIPTS_ALREADY_BUILT: stop immediately. Report that
  retrieval is skipped and do nothing else.
- If it returns RETRIEVAL_NEEDED: continue.

## STEP 1 — Ensure the data split (so you know the task)
Call `ensure_data_split_fn`. Note the reported `input_modality` (stereo or mono) and the
train size — you will use the modality in one of your search queries and to describe the task.

## STEP 2 — Run EXACTLY 4 web searches
Call `web_search` exactly four times, once per query below. Substitute the detected modality
into query 3 (use the literal word "stereo" or "mono"):
1. "best PyTorch pretrained model binary image classification small dataset 2024 2025"
2. "state of the art lightweight image classification PyTorch few hundred samples pretrained"
3. "PyTorch <modality> image defect detection industrial inspection pretrained backbone"
4. "EfficientNet ViT ConvNeXT small data image classification PyTorch example code"

Read every result. If a query returns SEARCH_UNAVAILABLE / SEARCH_EMPTY, still proceed using
your knowledge of CURRENT small-data backbones — never fall back to a legacy ResNet18-only set.

## STEP 3 — Extract 4 candidates
From the search results, identify 4 DISTINCT, modern, pretrained PyTorch architectures that
suit binary industrial defect detection (pass/fail) on a few-hundred-sample dataset. For each:
- model_name: the architecture name (e.g. "EfficientNet-B0", "ConvNeXt-Tiny", "ViT-Tiny", "DeiT-Small").
- description: 1-2 sentences on why it suits a small-data binary defect-detection task.
- example_code: a minimal PyTorch snippet that loads/instantiates the model with pretrained
  weights and adapts the head for binary output (use torchvision or timm as the search suggests).
Prefer diversity across the 4 (different families/sizes). Do not return four near-identical models.

## STEP 4 — Store
Call `store_retrieved_candidates` with a JSON array of your 4 objects
(keys: model_name, description, example_code). The tool writes state['retrieved_candidates']
and echoes the formatted menu.

## Final response
Briefly list the 4 retrieved model names. Do not paste full code in your final text.
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

retriever_agent = LlmAgent(
    name="retriever_agent",
    model=config.MODEL_PRO,
    description=(
        "A_retriever (MLE-STAR Section 3.1): runs M=4 web searches to retrieve 4 "
        "state-of-the-art PyTorch model candidates {description, example code} for the "
        "current AOI task and stores them in state['retrieved_candidates']."
    ),
    instruction=_RETRIEVER_INSTRUCTION,
    tools=[
        _check_retriever_needed_tool,
        _ensure_data_split_tool,
        _web_search_tool,
        _store_retrieved_candidates_tool,
    ],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)

# Convenience alias: the paper's A_retriever entry point is the agent itself.
retrieve_candidate_models = retriever_agent

__all__ = ["retriever_agent", "retrieve_candidate_models", "web_search", "store_retrieved_candidates"]
