"""CrewAI runner — role-based multi-agent orchestration."""
from __future__ import annotations

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import Field

from ._common import calculator, web_search, count_tokens, grade_collab, make_workspace, extract_file_writes, extract_code_blocks


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate a math expression. Pass a Python expression like '2+2' or 'sqrt(1521)'."
    def _run(self, expression: str) -> str:
        return calculator(expression)


class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web and return short snippets."
    def _run(self, query: str) -> str:
        return web_search(query)


def _llm_id(cfg) -> str:
    # CrewAI uses LiteLLM under the hood; "openai/gpt-4o-mini" or "gpt-4o-mini" both work.
    m = cfg.model
    return m if "/" in m else f"openai/{m}"


def run(task: dict, cfg) -> dict:
    cat = task["category"]
    os.environ.setdefault("OPENAI_MODEL_NAME", cfg.model)

    if cat == "gsm8k":
        solver = Agent(
            role="Math Solver",
            goal="Solve grade-school math word problems precisely.",
            backstory="An expert tutor who shows brief working then states the final number.",
            llm=_llm_id(cfg), allow_delegation=False, verbose=False,
        )
        verifier = Agent(
            role="Answer Verifier",
            goal="Read the solver's work and produce the final numeric answer only.",
            backstory="Outputs only 'Final answer: <number>' on a single line.",
            llm=_llm_id(cfg), allow_delegation=False, verbose=False,
        )
        t1 = Task(description=f"Solve: {task['question']}\nShow brief working, then 'Final answer: <number>'.",
                  expected_output="Step-by-step plus final numeric answer.", agent=solver)
        t2 = Task(description="Read the previous solution and emit only 'Final answer: <number>'.",
                  expected_output="A line: Final answer: <number>", agent=verifier, context=[t1])
        crew = Crew(agents=[solver, verifier], tasks=[t1, t2], process=Process.sequential, verbose=False)
        out = crew.kickoff()
        text = str(out)
        ptoks, ctoks = _crew_usage(crew, text, task)
        return {"output": text, "prompt_tokens": ptoks, "completion_tokens": ctoks}

    if cat == "tool_use":
        agent = Agent(
            role="Tool-using Researcher",
            goal="Use the calculator or web_search tool to answer the user's question.",
            backstory="Picks the right tool, runs it, and reports results crisply.",
            llm=_llm_id(cfg), allow_delegation=False, verbose=False,
            tools=[CalculatorTool(), SearchTool()],
        )
        t = Task(description=f"{task['question']}\nEnd with 'Final answer: <answer>'.",
                 expected_output="Final answer line.", agent=agent)
        crew = Crew(agents=[agent], tasks=[t], process=Process.sequential, verbose=False)
        out = crew.kickoff()
        text = str(out)
        ptoks, ctoks = _crew_usage(crew, text, task)
        return {"output": text, "prompt_tokens": ptoks, "completion_tokens": ctoks}

    if cat == "collab":
        ws = make_workspace("crewai_collab")
        spec = task["spec"]
        pm = Agent(role="Product Manager", goal="Clarify the spec and pass it to engineering.",
                   backstory="Concise PM. Outputs a 4-bullet plan.",
                   llm=_llm_id(cfg), allow_delegation=False, verbose=False)
        eng = Agent(role="Senior Python Engineer",
                    goal="Implement wordstats.py and README.md per the plan.",
                    backstory="Writes clean stdlib-only Python with doctests.",
                    llm=_llm_id(cfg), allow_delegation=False, verbose=False)
        rev = Agent(role="Reviewer",
                    goal="Emit final two files in 'FILE:' blocks for materialization.",
                    backstory="Outputs exactly:\nFILE: wordstats.py\n```python\n<code>\n```\nFILE: README.md\n```\n<text>\n```",
                    llm=_llm_id(cfg), allow_delegation=False, verbose=False)
        t1 = Task(description=f"Plan from spec:\n{spec}", expected_output="4 bullets.", agent=pm)
        t2 = Task(description="Implement the module per the plan.", expected_output="Code + readme draft.",
                  agent=eng, context=[t1])
        t3 = Task(description="Output the two files in FILE: format only. No commentary.",
                  expected_output="FILE: wordstats.py ... FILE: README.md ...",
                  agent=rev, context=[t2])
        crew = Crew(agents=[pm, eng, rev], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)
        out = crew.kickoff()
        text = str(out)
        _materialize(text, ws)
        score, breakdown = grade_collab(ws)
        ptoks, ctoks = _crew_usage(crew, text, task)
        return {"output": text[:2000], "prompt_tokens": ptoks, "completion_tokens": ctoks,
                "rubric_score": score, "rubric_breakdown": breakdown}

    raise ValueError(f"Unknown category {cat}")


def _crew_usage(crew, text: str, task: dict) -> tuple[int, int]:
    """CrewAI exposes usage_metrics on Crew after kickoff()."""
    try:
        m = crew.usage_metrics
        if hasattr(m, "model_dump"):
            d = m.model_dump()
        elif isinstance(m, dict):
            d = m
        else:
            d = {}
        p = int(d.get("prompt_tokens", 0))
        c = int(d.get("completion_tokens", 0))
        if p or c:
            return p, c
    except Exception:
        pass
    # fallback estimate
    return count_tokens(task.get("question", "") + task.get("spec", "")), count_tokens(text)


def _materialize(text: str, ws: Path) -> None:
    files = extract_file_writes(text)
    if not files:
        blocks = extract_code_blocks(text)
        if blocks:
            files["wordstats.py"] = blocks[0]
            if len(blocks) > 1:
                files["README.md"] = blocks[1]
    for name, content in files.items():
        if "/" in name or ".." in name:
            continue
        (ws / name).write_text(content)
