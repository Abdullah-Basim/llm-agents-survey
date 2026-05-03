"""AutoGen runner — autogen-agentchat 0.7.x async API."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ._common import calculator, web_search, count_tokens, grade_collab, make_workspace, extract_file_writes, extract_code_blocks


def _client(cfg) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(model=cfg.model, temperature=cfg.temperature)


def run(task: dict, cfg) -> dict:
    return asyncio.run(_run_async(task, cfg))


async def _run_async(task: dict, cfg) -> dict:
    cat = task["category"]
    client = _client(cfg)

    if cat == "gsm8k":
        agent = AssistantAgent(
            name="solver",
            model_client=client,
            system_message="Solve grade-school math step by step. End with 'Final answer: <number>' and then 'TERMINATE'.",
        )
        team = RoundRobinGroupChat(
            participants=[agent],
            termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(3),
        )
        result = await team.run(task=task["question"])
        return _harvest(result, fallback_in=task["question"])

    if cat == "tool_use":
        calc_tool = FunctionTool(
            calculator,
            description="Evaluate a math expression like '2+2' or 'sqrt(1521)'.",
            name="calculator",
        )
        search = FunctionTool(
            web_search,
            description="Web search; returns short snippets.",
            name="web_search",
        )
        agent = AssistantAgent(
            name="solver",
            model_client=client,
            tools=[calc_tool, search],
            reflect_on_tool_use=True,
            system_message="Use the right tool. End with 'Final answer: <answer>' and 'TERMINATE'.",
        )
        team = RoundRobinGroupChat(
            participants=[agent],
            termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(8),
        )
        result = await team.run(task=task["question"])
        return _harvest(result, fallback_in=task["question"])

    if cat == "collab":
        ws = make_workspace("autogen_collab")
        spec = task["spec"]
        pm = AssistantAgent(
            name="pm", model_client=client,
            system_message="You are a Product Manager. Produce a 4-bullet plan from the spec, then say 'PLAN_DONE'.",
        )
        eng = AssistantAgent(
            name="engineer", model_client=client,
            system_message=(
                "You are a senior engineer. After PLAN_DONE, output exactly:\n"
                "FILE: wordstats.py\n```python\n<code>\n```\nFILE: README.md\n```\n<text>\n```\n"
                "Then say 'TERMINATE'."
            ),
        )
        team = RoundRobinGroupChat(
            participants=[pm, eng],
            termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(6),
        )
        result = await team.run(task=f"Spec:\n{spec}")
        out = _harvest(result, fallback_in=spec)
        _materialize(out["output"], ws)
        score, breakdown = grade_collab(ws)
        out["rubric_score"] = score
        out["rubric_breakdown"] = breakdown
        return out

    raise ValueError(f"Unknown category {cat}")


def _harvest(result, fallback_in: str = "") -> dict:
    msgs = getattr(result, "messages", None) or []
    text_chunks = []
    p = c = 0
    for m in msgs:
        content = getattr(m, "content", None)
        if isinstance(content, str):
            text_chunks.append(content)
        usage = getattr(m, "models_usage", None)
        if usage is not None:
            p += int(getattr(usage, "prompt_tokens", 0) or 0)
            c += int(getattr(usage, "completion_tokens", 0) or 0)
    text = text_chunks[-1] if text_chunks else ""
    if p == 0 and c == 0:
        p = count_tokens(fallback_in + " ".join(text_chunks[:-1]))
        c = count_tokens(text)
    return {"output": text, "prompt_tokens": p, "completion_tokens": c}


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
