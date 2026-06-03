# MLE-STAR AOI Agent

An autonomous, LLM-driven pipeline that designs, trains, refines, and ensembles
image-classification models for **AOI (Automated Optical Inspection)** defect
detection. Given a folder of inspected boards — PNG images plus a per-lot Excel
sheet of pass/fail results — it iterates through four phases (initialisation →
refinement → ensemble → submission) to produce a model that meets configurable
acceptance targets (miss rate, NG recall, overkill rate, accuracy).

It is built on the [Google ADK](https://github.com/google/adk-python) agent
framework and runs its LLM reasoning on DeepSeek via LiteLLM.

---

## What you need

- **Python 3.11+** and the dependencies in [requirements.txt](requirements.txt):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r mle_star_agent/requirements.txt
  ```
- **A DeepSeek API key** (required) and optionally a web-search key — see
  [.env setup](#4-secrets-env) below.
- **Your dataset**, laid out as one folder per lot:
  ```
  <your lot folder>/
    ├── *.png          # cell images. Stereo: paired "<row>-<col>_L_*.png" / "_R_*.png".
    │                  #   Mono: one "<row>-<col>_*.png" per cell. Auto-detected.
    └── *.xlsx         # one sheet with Row, Column, and a result column
    │                  #   (TestResult / Result / Label) per cell.
  ```
  Stereo vs mono is detected automatically from the `_L_`/`_R_` filename infix.

---

## Point it at your data — 4 config edits

All dataset-specific knobs live at the top of
[config.py](config.py). A new dataset needs **only** these edits — no other
source file changes.

### 1. Dataset location — `DATASET_GLOB`
A glob (relative to the project root, or absolute) matching your lot folders:
```python
DATASET_GLOB = "[[]SUP046]*"        # default: bundled SUP046 lots
# e.g. DATASET_GLOB = "lots/*"      # all folders under ./lots
```

### 2. Label convention — `PASS_LABELS` / `FAIL_LABELS`
List every raw value your result column uses. Matching is case-insensitive;
internally everything maps to the canonical **G** (pass) / **NG** (fail) codes
the pipeline uses throughout. Defaults already cover G/NG, PASS/FAIL, OK, 0/1:
```python
FAIL_LABELS = {"fail", "ng", "1", "defect", "defective", "true", "positive"}
PASS_LABELS = {"pass", "ok", "g", "good", "0", "false", "negative"}
```
A value found in **neither** set raises a clear error at data load — it is never
silently dropped — so just add new wording here.

### 3. Board grouping — `BOARD_CODE_PATTERN`
Samples from the same physical board are kept together across the train/val/test
split to prevent leakage. This regex is matched against each lot-folder name and
the match becomes the board's group key:
```python
BOARD_CODE_PATTERN = r"VHB[A-Z0-9]+"     # default: SUP046 "VHB…" codes
BOARD_CODE_STRIP_SUFFIX_DIGITS = 2       # strip lot-sequence digits; 0 to disable
# Use r".+" to treat each lot folder as its own board.
```

### 4. Secrets — `.env`
Copy the template and fill in keys:
```bash
cp .env.example .env
```
- `DEEPSEEK_API_KEY` — **required**; the agent refuses to start without it.
- `TAVILY_API_KEY` *or* `SERPER_API_KEY` — optional. The retriever prefers Tavily,
  then Serper, then a keyless DuckDuckGo fallback (lower quality). Set one for
  reliable backbone/literature search.

---

## Verify your setup

Before a full run, confirm your data loads and splits correctly:
```bash
python test_data_split.py
```
This prints sample counts, the train/val/test sizes, class balance, and the
board groups it detected, and writes `checkpoints/data_split_grouped.json`. If
your label or folder config is wrong, this is where it fails loudly.

## Run the pipeline

The agent exposes `root_agent` following the ADK convention, so run it with the
ADK CLI from the directory that contains the `mle_star_agent` package:
```bash
adk run mle_star_agent        # interactive
# or
adk web                       # browser UI; pick "mle_star_root"
```
Progress is checkpointed under `checkpoints/`, so a re-run resumes rather than
restarting. Acceptance targets and loop caps are configurable in
[config.py](config.py) (see the `OVERKILL_*`, `NG_RECALL_*`, `MISS_RATE_*`,
`ACCURACY_*`, and `*_LOOP_MAX` settings).

---

## How a different dataset maps in

| Your data                         | What to set                                  |
|-----------------------------------|----------------------------------------------|
| Folders named `Run_42_BRDxyz`     | `DATASET_GLOB="Run_*"`, `BOARD_CODE_PATTERN=r"BRD[a-z]+"` |
| Result column says `PASS`/`FAIL`  | already covered by defaults                  |
| Result column says `accept`/`scrap` | add them to `PASS_LABELS` / `FAIL_LABELS`  |
| One image per cell (no L/R)       | nothing — mono is auto-detected              |
| Each lot is its own board         | `BOARD_CODE_PATTERN=r".+"`, strip digits `0` |

> Note: the bundled project root folder name contains a trailing space
> (`"AOI agent "`). If you relocate the project, a path without the trailing
> space is fine — just keep `DATASET_GLOB` pointing at your lot folders.
