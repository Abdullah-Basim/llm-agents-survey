"""Generates all paper figures + the summary CSV from per-run JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT.parent / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

FRAMEWORKS = ["crewai", "autogen", "langgraph", "metagpt"]
CATEGORIES = ["gsm8k", "tool_use", "collab"]
CAT_LABEL = {"gsm8k": "Reasoning (GSM8K)", "tool_use": "Tool Use", "collab": "Collaboration"}
FW_LABEL = {"crewai": "CrewAI", "autogen": "AutoGen", "langgraph": "LangGraph", "metagpt": "MetaGPT*"}
FW_COLOR = {"crewai": "#1f77b4", "autogen": "#ff7f0e", "langgraph": "#2ca02c", "metagpt": "#d62728"}


def load_df() -> pd.DataFrame:
    rows = []
    for fp in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        rows.append(d)
    if not rows:
        raise SystemExit("No result JSON files found in benchmark/results/.")
    df = pd.DataFrame(rows)
    df["framework"] = df["framework"].astype(str)
    df["task_category"] = df["task_category"].astype(str)
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce").fillna(0.0)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).astype(int)
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce").fillna(0).astype(int)
    df["success"] = df["success"].astype(bool).astype(int)
    return df


def fig_accuracy(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="task_category", columns="framework", values="success", aggfunc="mean")
    pivot = pivot.reindex(index=CATEGORIES, columns=FRAMEWORKS).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(pivot.index))
    w = 0.18
    for i, fw in enumerate(FRAMEWORKS):
        vals = pivot[fw].values * 100
        ax.bar(x + (i - 1.5) * w, vals, w, label=FW_LABEL[fw], color=FW_COLOR[fw])
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABEL[c] for c in pivot.index])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-task success rate by framework (gpt-4o-mini backbone)")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_accuracy.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_cost_pareto(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    agg = df.groupby(["framework", "task_category"]).agg(
        cost=("cost_usd", "mean"),
        success=("success", "mean"),
    ).reset_index()
    markers = {"gsm8k": "o", "tool_use": "s", "collab": "^"}
    for _, row in agg.iterrows():
        ax.scatter(row["cost"], row["success"] * 100,
                   s=140, marker=markers[row["task_category"]],
                   color=FW_COLOR[row["framework"]],
                   edgecolor="black", linewidth=0.6,
                   label=f"{FW_LABEL[row['framework']]} ({CAT_LABEL[row['task_category']]})")
    ax.set_xlabel("Mean cost per task (USD)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Cost vs. accuracy Pareto view")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen, dedup_h, dedup_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); dedup_h.append(h); dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cost_pareto.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_cost_pareto.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_latency_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    data = [df[df["framework"] == fw]["latency_s"].values for fw in FRAMEWORKS]
    bp = ax.boxplot(data, labels=[FW_LABEL[fw] for fw in FRAMEWORKS], patch_artist=True, showfliers=True)
    for patch, fw in zip(bp["boxes"], FRAMEWORKS):
        patch.set_facecolor(FW_COLOR[fw]); patch.set_alpha(0.7)
    ax.set_ylabel("Latency per task (s)")
    ax.set_title("Wall-clock latency distribution by framework")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_latency_box.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_latency_box.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_tokens(df: pd.DataFrame) -> None:
    agg = df.groupby("framework").agg(
        p=("prompt_tokens", "mean"),
        c=("completion_tokens", "mean"),
    ).reindex(FRAMEWORKS).fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(FRAMEWORKS))
    ax.bar(x, agg["p"], color="#5b8def", label="Input tokens")
    ax.bar(x, agg["c"], bottom=agg["p"], color="#f59e0b", label="Output tokens")
    ax.set_xticks(x); ax.set_xticklabels([FW_LABEL[fw] for fw in FRAMEWORKS])
    ax.set_ylabel("Average tokens per task")
    ax.set_title("Token consumption per framework (mean across all tasks)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_tokens.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_tokens.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summary_csv(df: pd.DataFrame) -> None:
    out = df.groupby("framework").agg(
        accuracy=("success", "mean"),
        avg_latency_s=("latency_s", "mean"),
        avg_cost_usd=("cost_usd", "mean"),
        avg_input_tokens=("prompt_tokens", "mean"),
        avg_output_tokens=("completion_tokens", "mean"),
        n=("success", "count"),
    ).reindex(FRAMEWORKS)
    out["accuracy"] = (out["accuracy"] * 100).round(1)
    out["avg_latency_s"] = out["avg_latency_s"].round(2)
    out["avg_cost_usd"] = out["avg_cost_usd"].round(5)
    out["avg_input_tokens"] = out["avg_input_tokens"].round(0).astype(int)
    out["avg_output_tokens"] = out["avg_output_tokens"].round(0).astype(int)
    out.to_csv(ROOT / "results_summary.csv")
    print(out.to_string())

    # Per-task category breakdown for paper Table I
    tc = df.groupby(["framework", "task_category"]).agg(
        accuracy=("success", "mean"),
        avg_cost=("cost_usd", "mean"),
        avg_latency=("latency_s", "mean"),
    ).reset_index()
    tc["accuracy"] = (tc["accuracy"] * 100).round(1)
    tc["avg_cost"] = tc["avg_cost"].round(4)
    tc["avg_latency"] = tc["avg_latency"].round(1)
    tc.to_csv(ROOT / "results_per_category.csv", index=False)


def main() -> None:
    df = load_df()
    print(f"Loaded {len(df)} run records.")
    fig_accuracy(df)
    fig_cost_pareto(df)
    fig_latency_box(df)
    fig_tokens(df)
    summary_csv(df)
    print(f"Wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
