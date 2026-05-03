# How to Run Each Framework

Every framework is installed inside the project's local virtualenv at
`~/Projects/llm-agents-survey/.venv`. They share the same harness in
`benchmark/harness.py`, so the run pattern is identical — only the
`--framework` flag changes.

## 0. One-time setup

```bash
cd ~/Projects/llm-agents-survey
source .venv/bin/activate          # activates the project venv

# OpenAI key must be set (used by every framework's backbone)
export OPENAI_API_KEY=sk-...        # or put it in .env, harness loads it

# CrewAI is chatty by default; suppress its remote tracing
export CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true

# So the harness can find the package
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Confirm the four frameworks are present:

```bash
pip list | grep -iE "crewai|autogen|langgraph|openai" | sort
# Expected:
# autogen-agentchat 0.7.5
# autogen-core      0.7.5
# autogen-ext       0.7.5
# crewai            1.9.3
# langchain-openai  0.3.23
# langgraph         1.0.1
# openai            1.83.0
```

## 1. Universal harness command

Every framework speaks to the same CLI:

```bash
python -m benchmark.harness --framework <fw> --task <category> [--n N]
```

- `<fw>` — `crewai` | `autogen` | `langgraph` | `metagpt`
- `<category>` — `gsm8k` | `tool_use` | `collab`
- `--n N` — limit to first N tasks (omit or `0` = all)

Each call writes one JSON per task to `benchmark/results/`.

---

## 2. CrewAI

**What it does** — Two roles per reasoning task (`Math Solver` ➜ `Answer Verifier`),
single tool-using agent for tool tasks, and a `PM ➜ Engineer ➜ Reviewer` crew
for the collab task. Code: [benchmark/runners/run_crewai.py](benchmark/runners/run_crewai.py).

```bash
# Smoke test (one task)
python -m benchmark.harness --framework crewai --task gsm8k --n 1

# Run a full category
python -m benchmark.harness --framework crewai --task gsm8k       # 15 problems
python -m benchmark.harness --framework crewai --task tool_use    # 10 problems
python -m benchmark.harness --framework crewai --task collab      # 1 task
```

> Telemetry note: CrewAI tries to phone home; the env vars above silence it.

---

## 3. AutoGen (Microsoft autogen-agentchat 0.7)

**What it does** — Uses the new async API: `AssistantAgent` +
`RoundRobinGroupChat` with a `TextMentionTermination("TERMINATE")`. Tool tasks
expose `calculator` and `web_search` via `FunctionTool`. Collab uses a `PM`
+ `Engineer` round-robin chat. Code: [benchmark/runners/run_autogen.py](benchmark/runners/run_autogen.py).

```bash
python -m benchmark.harness --framework autogen --task gsm8k --n 1
python -m benchmark.harness --framework autogen --task gsm8k
python -m benchmark.harness --framework autogen --task tool_use
python -m benchmark.harness --framework autogen --task collab
```

> If you see "No module named 'autogen'", you have the legacy 0.2 import
> path in mind — this project uses the v0.4+ split packages
> (`autogen_agentchat`, `autogen_core`, `autogen_ext`).

---

## 4. LangGraph

**What it does** — Uses `langgraph.prebuilt.create_react_agent` with a
`ChatOpenAI` model. Tool tasks register `calc_tool` and `search_tool`
(`@langchain_core.tools.tool`). Collab uses the same single-graph agent and
expects the model to emit `FILE: ...` blocks the harness materializes. Code:
[benchmark/runners/run_langgraph.py](benchmark/runners/run_langgraph.py).

```bash
python -m benchmark.harness --framework langgraph --task gsm8k --n 1
python -m benchmark.harness --framework langgraph --task gsm8k
python -m benchmark.harness --framework langgraph --task tool_use
python -m benchmark.harness --framework langgraph --task collab
```

---

## 5. MetaGPT-style SOP

**What it does** — A faithful PM ➜ Architect ➜ Engineer assembly line
implemented directly on `openai.OpenAI()` chat completions, mirroring
MetaGPT's published SOP paradigm. Tool tasks use OpenAI's native
function-calling API (max 3 tool-call iterations). The official `metagpt`
PyPI package was *not* installed because of its heavy/conflicting
dependencies; the paper documents this honestly. Code:
[benchmark/runners/run_metagpt.py](benchmark/runners/run_metagpt.py).

```bash
python -m benchmark.harness --framework metagpt --task gsm8k --n 1
python -m benchmark.harness --framework metagpt --task gsm8k
python -m benchmark.harness --framework metagpt --task tool_use
python -m benchmark.harness --framework metagpt --task collab
```

---

## 6. Run the entire benchmark in one shot

This re-runs every (framework × category) combination — 104 task runs total —
and then regenerates all four PDF figures plus the summary CSVs.

```bash
bash run_all.sh
# Resumable: it skips nothing, so wipe results/ first if you want a clean rerun:
# rm -f benchmark/results/*.json
```

Outputs land in:

| Path                                        | What it is                    |
|---------------------------------------------|-------------------------------|
| `benchmark/results/<fw>__<cat>__<task>.json`| One JSON per run (raw)        |
| `benchmark/results_summary.csv`             | Per-framework aggregate table |
| `benchmark/results_per_category.csv`        | Per-(fw, category) table      |
| `paper/figures/fig_accuracy.pdf`            | Plot: success rate            |
| `paper/figures/fig_cost_pareto.pdf`         | Plot: cost vs. accuracy       |
| `paper/figures/fig_latency_box.pdf`         | Plot: latency distribution    |
| `paper/figures/fig_tokens.pdf`              | Plot: input/output tokens     |

Regenerate just the figures from existing JSON results:

```bash
python benchmark/analyze.py
```

---

## 7. Inspecting one run

A quick one-liner to read the JSON of a specific run:

```bash
python -m json.tool benchmark/results/crewai__gsm8k__g1.json
```

Each result file contains: `framework`, `task_id`, `task_category`, `success`,
`output` (truncated), `latency_s`, `prompt_tokens`, `completion_tokens`,
`cost_usd`, `error` (if any), and `extra` (e.g. `rubric_breakdown` for the
collab task).

---

## 8. Recompiling the paper

```bash
cd paper
tectonic main.tex                  # one-shot, downloads packages on demand
# or, if you have a full TeX install:
# pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Outputs `paper/main.pdf`. The submission-named copies sit at the project
root: `MSDS25004_FinalPaper.pdf` and `MSDS25004_FinalPaper.zip`.

---

## 9. Common issues

| Symptom                                              | Fix                                                                 |
|------------------------------------------------------|---------------------------------------------------------------------|
| `ModuleNotFoundError: openai`                        | `source .venv/bin/activate` first                                   |
| `OpenAIError: api_key client option not set`         | `export OPENAI_API_KEY=sk-...` or write `.env` in project root      |
| CrewAI prints `Failed to resolve telemetry.crewai…`  | Harmless. Kill it with `export CREWAI_DISABLE_TELEMETRY=true`.      |
| `AttributeError: openai has no Beta`                 | OpenAI SDK got upgraded; `pip install "openai==1.83.0"`             |
| LangGraph import errors                              | `pip install -U langgraph langchain-openai langchain-community`     |
| `tectonic: command not found`                        | `brew install tectonic`                                             |
| Want pdflatex instead                                | `brew install --cask basictex` (needs sudo for installer)           |

---

## 10. Estimated cost & time

A full sweep on `gpt-4o-mini` is ~$0.02 of API spend and ~10 minutes wall
clock. Individual smoke tests (`--n 1`) are ~$0.0001 and finish in seconds.
