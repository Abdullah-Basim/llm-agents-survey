"""Unified benchmark harness for LLM agent frameworks.

Every framework runner exposes `run(task: dict, config: RunConfig) -> RunResult`.
The harness collects timing, token usage, and cost uniformly so figures are apples-to-apples.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# gpt-4o-mini pricing as of 2025: $0.150 / 1M input, $0.600 / 1M output.
PRICE_IN_PER_1M = 0.150
PRICE_OUT_PER_1M = 0.600

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class RunConfig:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: int = 180


@dataclass
class RunResult:
    framework: str
    task_id: str
    task_category: str
    success: bool
    output: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def usd_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * PRICE_IN_PER_1M / 1_000_000
        + completion_tokens * PRICE_OUT_PER_1M / 1_000_000
    )


# --- Task loading ---------------------------------------------------------------

def load_gsm8k() -> list[dict]:
    return json.loads((TASKS_DIR / "gsm8k_subset.json").read_text())

def load_tool_use() -> list[dict]:
    return json.loads((TASKS_DIR / "tool_use_tasks.json").read_text())

def load_collab_spec() -> str:
    return (TASKS_DIR / "collab_task.md").read_text()


# --- Answer checking ------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+\.?\d*")

def extract_final_number(text: str) -> str | None:
    """Extract the last number from text — works for GSM8K-style answers."""
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "")
    nums = _NUM_RE.findall(cleaned)
    if not nums:
        return None
    last = nums[-1]
    try:
        v = float(last)
        return str(int(v)) if v.is_integer() else f"{v:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return None


def gsm8k_correct(model_output: str, gold_answer: str) -> bool:
    pred = extract_final_number(model_output)
    gold = extract_final_number(gold_answer)
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return False


def tool_use_correct(model_output: str, expected: str) -> bool:
    """Permissive substring match for tool-use tasks: gold answer must appear in model output."""
    if not model_output or not expected:
        return False
    out = str(model_output).lower()
    for token in expected.lower().split("|"):
        if token.strip() in out:
            return True
    return False


# --- Persistence ----------------------------------------------------------------

def save_result(r: RunResult) -> Path:
    fp = RESULTS_DIR / f"{r.framework}__{r.task_category}__{r.task_id}.json"
    fp.write_text(r.to_json())
    return fp


def safe_run(
    framework: str,
    task: dict,
    runner_fn: Callable[[dict, RunConfig], dict],
    config: RunConfig,
) -> RunResult:
    """Runs a framework on a task. Catches all exceptions; records latency/usage even on failure."""
    t0 = time.time()
    try:
        out = runner_fn(task, config) or {}
        text = out.get("output", "")
        ptoks = int(out.get("prompt_tokens", 0))
        ctoks = int(out.get("completion_tokens", 0))
        extra = out.get("extra", {})
        elapsed = time.time() - t0

        category = task.get("category", "unknown")
        if category == "gsm8k":
            ok = gsm8k_correct(text, str(task.get("answer", "")))
        elif category == "tool_use":
            ok = tool_use_correct(text, str(task.get("expected", "")))
        elif category == "collab":
            ok = bool(out.get("rubric_score", 0) >= 3)
            extra["rubric_score"] = out.get("rubric_score", 0)
            extra["rubric_breakdown"] = out.get("rubric_breakdown", {})
        else:
            ok = False

        return RunResult(
            framework=framework,
            task_id=str(task.get("id", "?")),
            task_category=category,
            success=ok,
            output=text[:4000],
            latency_s=round(elapsed, 3),
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            cost_usd=round(usd_cost(ptoks, ctoks), 6),
            extra=extra,
        )
    except Exception as e:
        elapsed = time.time() - t0
        return RunResult(
            framework=framework,
            task_id=str(task.get("id", "?")),
            task_category=task.get("category", "unknown"),
            success=False,
            output="",
            latency_s=round(elapsed, 3),
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1000]}",
        )


# --- CLI driver ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=["crewai", "autogen", "langgraph", "metagpt"])
    parser.add_argument("--task", required=True, choices=["gsm8k", "tool_use", "collab"])
    parser.add_argument("--n", type=int, default=0, help="Limit number of tasks (0 = all)")
    args = parser.parse_args()

    from importlib import import_module
    mod = import_module(f"benchmark.runners.run_{args.framework}")

    if args.task == "gsm8k":
        tasks = load_gsm8k()
    elif args.task == "tool_use":
        tasks = load_tool_use()
    else:
        tasks = [{"id": "collab1", "category": "collab", "spec": load_collab_spec()}]

    if args.n > 0:
        tasks = tasks[: args.n]

    cfg = RunConfig()
    for task in tasks:
        result = safe_run(args.framework, task, mod.run, cfg)
        save_result(result)
        print(
            f"[{result.framework}] task={result.task_id} cat={result.task_category} "
            f"ok={result.success} lat={result.latency_s}s cost=${result.cost_usd:.4f}"
        )
