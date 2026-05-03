"""Helpers shared across runners: tools (calc/search), collab grader, token counter."""
from __future__ import annotations

import ast
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tiktoken

# ---------- Tools used by tool-use tasks ----------

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
            ast.FloorDiv: operator.floordiv}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        funcs = {"sqrt": math.sqrt, "abs": abs, "pow": pow, "round": round,
                 "min": min, "max": max, "log": math.log, "log10": math.log10,
                 "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp}
        fn = funcs.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        return fn(*[_safe_eval(a) for a in node.args])
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")

def calculator(expression: str) -> str:
    """Evaluate a math expression like '2+2', '1234 * 5678', 'sqrt(1521)'."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        v = _safe_eval(tree)
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return str(v)
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, max_results: int = 3) -> str:
    """Returns top-3 result snippets concatenated. Uses DuckDuckGo (no key)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return "No results."
        return " | ".join(f"{h.get('title','')}: {h.get('body','')}" for h in hits)
    except Exception as e:
        return f"Search error: {e}"


# ---------- Token counting ----------

_ENCODER = tiktoken.encoding_for_model("gpt-4o-mini") if hasattr(tiktoken, "encoding_for_model") else None

def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        if _ENCODER is None:
            return max(1, len(text) // 4)
        return len(_ENCODER.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------- Collab grader ----------

COLLAB_REFERENCE_TEXT = "the quick brown fox jumps over the lazy dog the fox is quick"

def grade_collab(workspace: Path) -> tuple[int, dict]:
    """Run the rubric. Returns (score 0-5, breakdown dict)."""
    breakdown = {
        "wordstats_exists": False,
        "analyze_defined": False,
        "returns_keys": False,
        "top3_correct": False,
        "readme_ok": False,
    }
    py = workspace / "wordstats.py"
    if py.exists():
        breakdown["wordstats_exists"] = True
        # try import + call
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 "import sys, importlib.util as u; "
                 f"spec=u.spec_from_file_location('w','{py}'); m=u.module_from_spec(spec); spec.loader.exec_module(m); "
                 f"out=m.analyze({COLLAB_REFERENCE_TEXT!r}); "
                 "import json; print(json.dumps(out, default=str))"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                import json as _json
                out = _json.loads(r.stdout.strip().splitlines()[-1])
                breakdown["analyze_defined"] = True
                expected_keys = {"word_count", "unique_words", "avg_word_length", "top_3"}
                if expected_keys.issubset(set(out.keys())):
                    breakdown["returns_keys"] = True
                top3 = out.get("top_3") or []
                top3_lower = [str(w).lower() for w in top3[:3]]
                # In the ref text, 'the' (3), 'fox' (2), 'quick' (2). Tie broken alphabetically.
                if top3_lower[:3] == ["the", "fox", "quick"]:
                    breakdown["top3_correct"] = True
        except Exception:
            pass
    readme = workspace / "README.md"
    if readme.exists():
        lines = [l for l in readme.read_text().splitlines() if l.strip()]
        if len(lines) >= 3:
            breakdown["readme_ok"] = True
    score = sum(1 for v in breakdown.values() if v)
    return score, breakdown


def make_workspace(prefix: str) -> Path:
    base = Path(__file__).resolve().parent.parent / "results" / "raw" / prefix
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------- Output extraction helpers ----------

def extract_code_blocks(text: str) -> list[str]:
    """Return code fences ```python ... ``` content."""
    return re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)


def extract_file_writes(text: str) -> dict:
    """Parse '## file: name.py\\n```...```' or 'FILE: name.py' style sections."""
    files = {}
    pattern = re.compile(
        r"(?:FILE|##\s*file|filename)\s*:\s*([\w\-./]+)\s*\n+```(?:\w+)?\n(.*?)```",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        files[m.group(1).strip()] = m.group(2)
    return files
