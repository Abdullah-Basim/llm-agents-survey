"""LangGraph runner — uses prebuilt ReAct agent with calculator + web_search tools."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ._common import calculator, web_search, count_tokens, grade_collab, make_workspace, extract_file_writes, extract_code_blocks


@lc_tool
def calc_tool(expression: str) -> str:
    """Evaluate a math expression. Use Python syntax: '+', '-', '*', '/', '**', 'sqrt(x)'."""
    return calculator(expression)


@lc_tool
def search_tool(query: str) -> str:
    """Search the web for a factual query and return short snippets."""
    return web_search(query)


def _build_llm(cfg) -> ChatOpenAI:
    return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, timeout=cfg.timeout_s, max_tokens=cfg.max_tokens)


def run(task: dict, cfg) -> dict:
    cat = task["category"]
    llm = _build_llm(cfg)

    if cat == "gsm8k":
        prompt = (
            "Solve the following grade-school math problem step by step. "
            "End your answer with a line of the form 'Final answer: <number>'.\n\n"
            f"Problem: {task['question']}"
        )
        agent = create_react_agent(llm, tools=[])
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        msgs = result["messages"]
        text = msgs[-1].content if msgs else ""
        ptoks, ctoks = _aggregate_usage(msgs)
        return {"output": text, "prompt_tokens": ptoks, "completion_tokens": ctoks}

    if cat == "tool_use":
        prompt = (
            "You have a calculator and a web-search tool. Use the most appropriate tool to answer. "
            "End your answer with 'Final answer: <answer>'.\n\n"
            f"Question: {task['question']}"
        )
        agent = create_react_agent(llm, tools=[calc_tool, search_tool])
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        msgs = result["messages"]
        text = msgs[-1].content if msgs else ""
        ptoks, ctoks = _aggregate_usage(msgs)
        return {"output": text, "prompt_tokens": ptoks, "completion_tokens": ctoks}

    if cat == "collab":
        ws = make_workspace("langgraph_collab")
        spec = task["spec"]
        prompt = (
            "You are a software engineering team coordinated through a single ReAct agent. "
            "Produce both required files. Output them in the format:\n"
            "FILE: wordstats.py\n```python\n<code>\n```\n\n"
            "FILE: README.md\n```\n<text>\n```\n\n"
            f"SPEC:\n{spec}"
        )
        agent = create_react_agent(llm, tools=[])
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        msgs = result["messages"]
        text = msgs[-1].content if msgs else ""
        ptoks, ctoks = _aggregate_usage(msgs)
        _materialize(text, ws)
        score, breakdown = grade_collab(ws)
        return {"output": text[:2000], "prompt_tokens": ptoks, "completion_tokens": ctoks,
                "rubric_score": score, "rubric_breakdown": breakdown}

    raise ValueError(f"Unknown category {cat}")


def _aggregate_usage(messages) -> tuple[int, int]:
    ptoks = ctoks = 0
    for m in messages:
        meta = getattr(m, "response_metadata", None) or {}
        usage = meta.get("token_usage") or {}
        ptoks += int(usage.get("prompt_tokens", 0))
        ctoks += int(usage.get("completion_tokens", 0))
    if ptoks == 0 and ctoks == 0:
        # fallback estimate
        text_in = "".join(getattr(m, "content", "") or "" for m in messages[:-1])
        text_out = getattr(messages[-1], "content", "") if messages else ""
        ptoks = count_tokens(text_in)
        ctoks = count_tokens(text_out)
    return ptoks, ctoks


def _materialize(text: str, ws: Path) -> None:
    files = extract_file_writes(text)
    if not files:
        # try fallback: first python block → wordstats.py, second block → README.md
        blocks = extract_code_blocks(text)
        if blocks:
            files["wordstats.py"] = blocks[0]
            if len(blocks) > 1:
                files["README.md"] = blocks[1]
    for name, content in files.items():
        if "/" in name or ".." in name:
            continue
        (ws / name).write_text(content)
