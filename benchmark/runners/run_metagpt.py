"""MetaGPT-style runner.

The official `metagpt` PyPI package is heavy (chromadb + many transitive deps) and
often fails to resolve under modern Python. To stay reproducible inside the 20h
sprint we implement a faithful MetaGPT-style SOP pipeline directly on the OpenAI
SDK: PM -> Architect -> Engineer for collab tasks, single Engineer for solo
tasks. The framing follows MetaGPT's published assembly-line paradigm
(Hong et al., ICLR 2024). This is documented openly in the paper's
Experimental Setup section.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from ._common import calculator, web_search, count_tokens, grade_collab, make_workspace, extract_file_writes, extract_code_blocks


def _client() -> OpenAI:
    return OpenAI()


def _chat(messages: list[dict], cfg) -> tuple[str, int, int]:
    resp = _client().chat.completions.create(
        model=cfg.model, messages=messages,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.timeout_s,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    p = int(getattr(usage, "prompt_tokens", 0) or 0)
    c = int(getattr(usage, "completion_tokens", 0) or 0)
    if p == 0 and c == 0:
        p = sum(count_tokens(m.get("content","")) for m in messages)
        c = count_tokens(text)
    return text, p, c


def run(task: dict, cfg) -> dict:
    cat = task["category"]

    if cat == "gsm8k":
        sys = ("You are an SOP-driven engineering team. Role: Engineer. "
               "Solve grade-school math step by step. End with 'Final answer: <number>'.")
        text, p, c = _chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": task["question"]},
        ], cfg)
        return {"output": text, "prompt_tokens": p, "completion_tokens": c}

    if cat == "tool_use":
        # Mini ReAct loop (max 3 iterations) using OpenAI tool-calling API.
        tools = [
            {"type": "function", "function": {
                "name": "calculator",
                "description": "Evaluate a math expression like '2+2' or 'sqrt(1521)'.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }},
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Web search; returns short snippets.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            }},
        ]
        messages: list[dict] = [
            {"role": "system", "content": "You are an SOP engineer. Pick the right tool. End with 'Final answer: <answer>'."},
            {"role": "user", "content": task["question"]},
        ]
        client = _client()
        ptot = ctot = 0
        final_text = ""
        for _ in range(4):
            resp = client.chat.completions.create(
                model=cfg.model, messages=messages, tools=tools,
                tool_choice="auto", temperature=cfg.temperature,
                max_tokens=cfg.max_tokens, timeout=cfg.timeout_s,
            )
            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            ptot += int(getattr(usage, "prompt_tokens", 0) or 0)
            ctot += int(getattr(usage, "completion_tokens", 0) or 0)
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    name = tc.function.name
                    import json as _j
                    args = _j.loads(tc.function.arguments or "{}")
                    if name == "calculator":
                        result = calculator(args.get("expression", ""))
                    elif name == "web_search":
                        result = web_search(args.get("query", ""))
                    else:
                        result = "Unknown tool."
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            final_text = msg.content or ""
            break
        if ptot == 0 and ctot == 0:
            ptot = count_tokens(task["question"])
            ctot = count_tokens(final_text)
        return {"output": final_text, "prompt_tokens": ptot, "completion_tokens": ctot}

    if cat == "collab":
        ws = make_workspace("metagpt_collab")
        spec = task["spec"]
        # PM
        pm_text, p1, c1 = _chat([
            {"role": "system", "content": "You are a Product Manager (MetaGPT SOP). Produce a 4-bullet plan."},
            {"role": "user", "content": f"Spec:\n{spec}"},
        ], cfg)
        # Architect
        arch_text, p2, c2 = _chat([
            {"role": "system", "content": "You are an Architect. Given the plan, write a 3-bullet design including the key signature for analyze()."},
            {"role": "user", "content": f"Plan:\n{pm_text}\n\nSpec:\n{spec}"},
        ], cfg)
        # Engineer
        eng_text, p3, c3 = _chat([
            {"role": "system", "content": ("You are an Engineer. Output ONLY two files in the format below — no other commentary:\n"
                                            "FILE: wordstats.py\n```python\n<code>\n```\n"
                                            "FILE: README.md\n```\n<text>\n```")},
            {"role": "user", "content": f"Plan:\n{pm_text}\n\nDesign:\n{arch_text}\n\nSpec:\n{spec}"},
        ], cfg)
        _materialize(eng_text, ws)
        score, breakdown = grade_collab(ws)
        return {"output": eng_text[:2000], "prompt_tokens": p1 + p2 + p3, "completion_tokens": c1 + c2 + c3,
                "rubric_score": score, "rubric_breakdown": breakdown}

    raise ValueError(f"Unknown category {cat}")


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
