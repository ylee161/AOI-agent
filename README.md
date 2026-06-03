# MLE-STAR AOI Agent

An autonomous, LLM-driven pipeline that designs, trains, refines, and ensembles
image-classification models for **AOI (Automated Optical Inspection)** defect
detection. Point it at a folder of inspected boards — images plus a per-lot Excel
sheet of pass/fail results — and it iterates through four phases
(initialisation → refinement → ensemble → submission) to produce a model that
meets configurable acceptance targets (miss rate, NG recall, overkill rate,
accuracy).

It is built on the [Google ADK](https://github.com/google/adk-python) agent
framework and runs its LLM reasoning on DeepSeek via LiteLLM. The design follows
the **MLE-STAR** method — the original paper is included in this repo
([MLE-STAR-paper.pdf](MLE-STAR-paper.pdf), with a short
[summary](MLE-STAR-paper-summary.md)).

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r mle_star_agent/requirements.txt

# 2. Secrets — at minimum a DeepSeek API key
cp .env.example .env        # then edit .env and set DEEPSEEK_API_KEY

# 3. Point at your data + verify it loads (see the detailed guide below)
python test_data_split.py

# 4. Run the pipeline (ADK convention: root_agent is exposed as `mle_star_root`)
adk run mle_star_agent      # interactive
# or: adk web               # browser UI; pick "mle_star_root"
```

### Convenience scripts

Two optional wrappers are included:

```bash
./run.sh           # retry-wrapped `adk run` — auto-restarts on transient
                   #   API errors and resumes from checkpoints. Ctrl-C to stop.
./start_web.sh     # launches the `adk web` UI and opens your browser
./start_web.sh 8080  # ...on a custom port
```

Both auto-detect a project-local `.venv/bin/adk`, falling back to `adk` on your
`PATH`. For a one-word launcher, add a shell alias, e.g.
`alias aoi='bash /path/to/AOI-agent/run.sh'`.

## Configuring it for your dataset

All dataset-specific knobs live at the top of
[mle_star_agent/config.py](mle_star_agent/config.py) — a new dataset needs only a
handful of edits (dataset glob, label vocabulary, board-grouping regex) and the
`.env` keys. The full walkthrough, including the expected folder/label layout,
mono-vs-stereo auto-detection, and acceptance-target settings, is in
**[mle_star_agent/README.md](mle_star_agent/README.md)**.

## What's in this repo

| Path | Purpose |
|------|---------|
| `mle_star_agent/` | The agent package (four phases, shared utilities, guards). |
| `mle_star_agent/config.py` | The single file you edit to point at new data and tune targets. |
| `mle_star_agent/requirements.txt` | Python dependencies. |
| `test_data_split.py` | Pre-flight check: confirms your data loads, splits, and groups correctly. |
| `.env.example` | Template for the required/optional API keys. |
| `MLE-STAR-paper.pdf` | The original MLE-STAR paper this pipeline is based on. |

> **Bring your own data.** No dataset, trained weights, or run checkpoints are
> shipped here — they are intentionally git-ignored. Configure `DATASET_GLOB` in
> `config.py` to point at your own lot folders.

## Requirements

- Python 3.11+
- A **DeepSeek API key** (required; the agent refuses to start without it)
- Optionally a Tavily or Serper key for higher-quality literature/backbone search
  (falls back to keyless DuckDuckGo otherwise)
