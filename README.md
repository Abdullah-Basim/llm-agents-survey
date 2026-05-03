# An Empirical Survey and Comparative Benchmark of LLM-Based Autonomous Agent Frameworks

Reproducible benchmark + paper artifact for the ITU MS-DS Research Methodology
term project (Spring 2026). The harness drives **CrewAI**, **AutoGen**,
**LangGraph**, and a **MetaGPT-style SOP** pipeline through identical tasks
using the same backbone model (`gpt-4o-mini`, temperature 0).

## Layout

```
paper/         # IEEE double-column LaTeX source + figures + bibliography
benchmark/
  harness.py     # unified runner contract + metrics + grading
  tasks/         # 15 GSM8K problems, 10 tool-use tasks, 1 collab spec
  runners/       # one runner per framework
  analyze.py     # produces all figures + summary tables
  results/       # raw per-run JSON files
run_all.sh     # full sweep
```

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

bash run_all.sh                # 4 frameworks x 3 task categories
python benchmark/analyze.py    # writes paper/figures/*.pdf and results_summary.csv
```

## Reproducing the paper

```bash
cd paper
tectonic main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What is being compared

| Axis            | CrewAI         | AutoGen        | LangGraph        | MetaGPT*       |
|-----------------|----------------|----------------|------------------|----------------|
| Reasoning       | ReAct/role     | ReAct + chat   | Prebuilt ReAct   | ReAct/CoT/SOP  |
| Planning        | Sequential     | Free-form      | State graph      | SOP pipeline   |
| Memory          | Per-task ctx   | Conversation   | Graph state      | Per-stage ctx  |
| Tool use        | Class-based    | Function reg.  | LangChain tools  | OpenAI tools   |
| Collaboration   | Roles + delg.  | Group chat     | Single graph     | PM/Arch/Eng    |

*MetaGPT-style: the official `metagpt` PyPI package was not installed inside the
20-hour project budget; we instead implement the published PM → Architect →
Engineer SOP pipeline directly on the OpenAI SDK as a faithful proxy.

## Tasks

- **GSM8K subset** — 15 grade-school math problems; correctness via final-number
  extraction.
- **Tool-use** — 10 hand-authored questions exercising calculator and web-search
  tools.
- **Multi-agent collaboration** — implement `wordstats.py` plus a README; graded
  on a 5-point functional rubric.

## License

Code: MIT. The paper, figures, and tables are licensed CC-BY-4.0.
