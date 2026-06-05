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
# Fetch a few extra so the authority/diversity re-rank (see _prioritize_and_trim)
# has something to choose from before we trim back down to _MAX_RESULTS.
_FETCH_RESULTS = 9
# Per-result page text (Tavily raw_content) surfaced to the LLM so it writes
# example code from REAL current pages, not just titles/snippets from memory.
_RAW_CONTENT_CHARS = 700
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Authoritative sources to float to the top of every result list (#5). Primary
# literature / official docs / code beat random blog and video results. Ordered
# by preference; matching is substring-on-domain so subdomains also match.
_AUTHORITATIVE_DOMAINS = (
    "arxiv.org",
    "paperswithcode.com",
    "openreview.net",
    "proceedings.mlr.press",
    "pytorch.org",
    "docs.pytorch.org",
    "huggingface.co",
    "github.com",
    "pmc.ncbi.nlm.nih.gov",
    "nature.com",
    "sciencedirect.com",
    "ieeexplore.ieee.org",
    "mdpi.com",
    "springer.com",
)


def _domain(url: str) -> str:
    m = re.match(r"\s*https?://([^/]+)", url or "")
    return m.group(1).lower().removeprefix("www.") if m else ""


def _authority_rank(url: str) -> int:
    """Lower is better. Authoritative domains rank by their list position; everything
    else ranks last (so blogs/videos sink below papers and official docs)."""
    d = _domain(url)
    for i, auth in enumerate(_AUTHORITATIVE_DOMAINS):
        if d == auth or d.endswith("." + auth) or auth in d:
            return i
    return len(_AUTHORITATIVE_DOMAINS)


def _prioritize_and_trim(results: list[dict], limit: int = _MAX_RESULTS) -> list[dict]:
    """Re-rank one query's results: authoritative sources first (#5), then one per
    domain before repeats so the kept set spans distinct sources (#6). Nothing is
    discarded until the final trim to `limit`, so content is never lost — only reordered."""
    ranked = sorted(results, key=lambda r: _authority_rank(r.get("url", "")))
    seen: set[str] = set()
    first_per_domain: list[dict] = []
    repeats: list[dict] = []
    for r in ranked:
        d = _domain(r.get("url", ""))
        if d and d not in seen:
            seen.add(d)
            first_per_domain.append(r)
        else:
            repeats.append(r)
    return (first_per_domain + repeats)[:limit]

# ---------------------------------------------------------------------------
# Pluggable web search (real API -> keyless DuckDuckGo -> graceful signal)
# ---------------------------------------------------------------------------

def _search_tavily(query: str, api_key: str) -> list[dict]:
    # search_depth="advanced" → better-ranked results; include_raw_content=True →
    # the actual page text, so the LLM can ground example code in real pages (#2).
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": _FETCH_RESULTS,
            "search_depth": "advanced",
            "include_raw_content": True,
        },
        timeout=_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
            "raw": (r.get("raw_content") or "")[:_RAW_CONTENT_CHARS],
        }
        for r in data.get("results", [])
    ]


def _search_serper(query: str, api_key: str) -> list[dict]:
    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": _FETCH_RESULTS},
        timeout=_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in data.get("organic", [])[:_FETCH_RESULTS]:
        out.append({
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "url": r.get("link", ""),
            "raw": "",  # Serper does not return full page content
        })
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
        results.append({"title": _strip_tags(m.group(2)), "snippet": "", "url": m.group(1), "raw": ""})
        if len(results) >= _FETCH_RESULTS:
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
            "small-data image-classification backbones (e.g. EfficientNet-B1/B2, MobileNetV3, "
            "ResNet-50, DINOv2/CLIP frozen) — do NOT use ConvNeXt or DeiT/ViT variants "
            "(known probe failures on this dataset) and do NOT default to ResNet18-only."
        )

    if not results:
        return (
            f"SEARCH_EMPTY for query {query!r}. No results returned. Use your knowledge "
            "of current small-data image-classification backbones; avoid a legacy "
            "ResNet18-only choice."
        )

    # Authority-first, source-diverse re-rank, then trim to the display count.
    results = _prioritize_and_trim(results, _MAX_RESULTS)
    logger.info("web_search via %s for %r — %d result(s) after re-rank", source, query, len(results))

    lines = [f"SEARCH RESULTS ({source}) for: {query}"]
    for i, r in enumerate(results, 1):
        snippet = r.get("snippet", "").strip()
        raw = r.get("raw", "").strip()
        lines.append(f"[{i}] {r.get('title', '').strip()}")
        if snippet:
            lines.append(f"    {snippet}")
        if raw:
            # Collapse whitespace so the page excerpt stays compact in the prompt.
            raw = re.sub(r"\s+", " ", raw)
            lines.append(f"    PAGE EXCERPT: {raw}")
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
        entry = {
            "model_name": name,
            "description": str(c.get("description", "")).strip(),
            "example_code": str(c.get("example_code", "")).strip(),
        }
        # Optional capacity hint (e.g. "~5M") so the coder can prefer the
        # lowest-capacity option on this few-hundred-sample task.
        param_count = str(c.get("param_count", "")).strip()
        if param_count:
            entry["param_count"] = param_count
        cleaned.append(entry)
    if not cleaned:
        return "ERROR: no candidate had a model_name — provide model_name for each entry."

    tool_context.state["retrieved_candidates"] = cleaned
    modality = tool_context.state.get("input_modality", "stereo")
    warn = "" if len(cleaned) == 4 else f"(NOTE: stored {len(cleaned)} candidates; the paper retrieves 4.) "
    return (
        f"STORED {len(cleaned)} retrieved candidate(s) into state['retrieved_candidates']. {warn}\n\n"
        + format_candidate_block(cleaned, modality)
    )


def store_technique_hints(tool_context, hints_json: str) -> str:
    """Persist 3-5 ADVISORY small-data technique ideas (augmentation, loss, calibration).

    Pass a JSON array of 3-5 short strings, each a single technique idea (e.g.
    "MixUp augmentation to regularize a few-hundred-sample set" or "temperature
    scaling for probability calibration"). These are stored as plain strings in
    ``state['retrieved_technique_hints']`` — they are NOT model candidates and do
    NOT use the {model_name, description, example_code} schema. They are advisory
    context for the Phase 2 planner; any strategy derived from them still passes
    the KNOWN_FAILED_STRATEGY_FINGERPRINTS gate before it can be acted on.
    """
    try:
        hints = json.loads(hints_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return (
            f"ERROR: hints_json is not valid JSON ({exc}). Pass a JSON list of 3-5 "
            "short technique-idea strings."
        )
    if not isinstance(hints, list) or not hints:
        return "ERROR: expected a non-empty JSON list of technique-idea strings."

    cleaned = []
    for h in hints:
        text = str(h).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        return "ERROR: no non-empty technique idea found — provide 3-5 short strings."

    tool_context.state["retrieved_technique_hints"] = cleaned
    return (
        f"STORED {len(cleaned)} technique hint(s) into state['retrieved_technique_hints'] "
        "(advisory only — not model candidates).\n- " + "\n- ".join(cleaned)
    )


_ensure_data_split_tool = FunctionTool(func=ensure_data_split_fn)
_check_retriever_needed_tool = FunctionTool(func=check_retriever_needed_fn)
_web_search_tool = FunctionTool(func=web_search)
_store_retrieved_candidates_tool = FunctionTool(func=store_retrieved_candidates)
_store_technique_hints_tool = FunctionTool(func=store_technique_hints)

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
Call `ensure_data_split_fn`. From its output, record the TASK PROFILE you will search with:
- TRAIN_SIZE — the number of training samples (this is a SMALL dataset; the exact count matters).
- CLASS_BALANCE — the NG (defect) vs G (good) counts, i.e. how imbalanced the task is.
- MODALITY — "stereo" or "mono".
You will bake these real numbers into your queries below — generic queries waste the search.

## STEP 2 — Run EXACTLY 4 web searches (DATA-AWARE)
Call `web_search` exactly four times. Build each query from the TASK PROFILE you just recorded —
substitute the ACTUAL train size, class-imbalance wording, and modality. Use these as templates:
1. "pretrained PyTorch image classification <TRAIN_SIZE> training samples transfer learning overfitting prevention"
2. "DINOv2 SigLIP CLIP frozen foundation model features linear probe small dataset binary defect pass/fail classification"
3. "PyTorch <MODALITY> surface defect detection industrial inspection pretrained backbone small data DINOv2 vision foundation model"
4. "LoRA AdaptFormer PEFT parameter-efficient fine-tuning vision transformer <TRAIN_SIZE> samples timm torchvision example code"

Read every result, INCLUDING the `PAGE EXCERPT:` lines — those are real text from the source page,
so prefer them over your own memory when writing example code (APIs change). If a query returns
SEARCH_UNAVAILABLE / SEARCH_EMPTY, still proceed using your knowledge of CURRENT small-data
backbones — never fall back to a legacy ResNet18-only set.

## STEP 3 — Extract 4 candidates (DIVERSE ON THE AXIS THAT MATTERS)
From the search results, identify 4 modern, pretrained PyTorch architectures that suit binary
industrial defect detection (pass/fail) on a dataset of only ~TRAIN_SIZE samples. For each:
- model_name: the architecture name (e.g. "EfficientNet-B0", "ConvNeXt-Tiny", "DINOv2-ViT-S/14",
  "SigLIP-ViT-B", "CLIP-ViT-B/32").
- description: 1-2 sentences on why it suits THIS small, imbalanced binary defect task. For a
  frozen foundation backbone, note the adaptation method (frozen features + linear/MLP probe, or
  parameter-efficient fine-tuning such as LoRA / AdaptFormer / PEFT) that keeps trainable params
  tiny on ~TRAIN_SIZE samples.
- example_code: a minimal PyTorch snippet that loads/instantiates the model with pretrained
  weights and adapts the head for binary output (use torchvision, timm, transformers, or a PEFT
  library as the pages suggest).
Diversity rule — on a few-hundred-sample task, model capacity / data-hunger is the axis that
matters, not just the family name. So your 4 MUST:
- include at least ONE lightweight option (<10M params, e.g. EfficientNet-B0/B1/B2 / MobileNetV3-Large / ResNet-50);
- include at least ONE FROZEN self-supervised / vision-language FOUNDATION backbone (e.g. DINOv2,
  SigLIP, or CLIP) used as a frozen feature extractor or adapted with LoRA / AdaptFormer / PEFT —
  these transfer strongly from very few labels and are a core small-data lever, not just classic CNNs;
- span a range of sizes/adaptation strategies rather than four similarly-large fully-fine-tuned models.
Do not return four near-identical models, and do not return an all-classic-CNN set.
HARD EXCLUSIONS — do NOT suggest these models under any circumstances, they have been
empirically proven to fail the pre-training probe on this dataset every time:
- ConvNeXt-Tiny (and any ConvNeXt variant) — outputs constant probability for all samples,
  catastrophic overkill, prob_gap ≈ 0. Root cause: 9-channel stem adaptation breaks ConvNeXt's
  LayerNorm-based stem; the model never learns any G/NG separation.
- DeiT-Small (and any DeiT/ViT variant requiring patch-embedding adaptation) — outputs constant
  low probability for all samples, recall collapses to 0. Root cause: the patch embedding is a
  linear projection (not a conv), so the /3-repeat initialization cannot be applied cleanly;
  the model produces no signal even after probe epochs.
Good replacement options: EfficientNet-B1/B2/B3 (same family as B0, more capacity),
MobileNetV3-Large (lightweight, robust on small datasets), ResNet-50 with partial unfreeze,
or a FROZEN DINOv2/CLIP/SigLIP feature extractor (these work as frozen backbones with no
input-layer adaptation needed).

## STEP 4 — Store
Call `store_retrieved_candidates` with a JSON array of your 4 objects
(keys: model_name, description, example_code). The tool writes state['retrieved_candidates']
and echoes the formatted menu.

## STEP 5 — Retrieve small-data TECHNIQUE hints (one more search)
Call `web_search` exactly ONE more time with a small-data *technique* query (not a
model query), built from the TASK PROFILE — for example:
"small dataset overfitting prevention techniques augmentation loss calibration
binary defect classification <TRAIN_SIZE> samples".
Read the results (including PAGE EXCERPT lines) and extract 3-5 concrete technique
ideas spanning augmentation, loss design, and probability calibration (e.g. MixUp/CutMix,
focal/asymmetric loss, temperature scaling, label smoothing, test-time augmentation).
These are TECHNIQUES, not architectures — do NOT turn them into model candidates.
Call `store_technique_hints` with a JSON array of 3-5 short strings. The tool writes
state['retrieved_technique_hints'] for the Phase 2 planner as advisory context only.

## Final response
Briefly list the 4 retrieved model names and the 3-5 technique hints. Do not paste
full code in your final text.
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
        _store_technique_hints_tool,
    ],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)

# Convenience alias: the paper's A_retriever entry point is the agent itself.
retrieve_candidate_models = retriever_agent

__all__ = [
    "retriever_agent",
    "retrieve_candidate_models",
    "web_search",
    "store_retrieved_candidates",
    "store_technique_hints",
]
