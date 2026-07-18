"""
Analysis & graphical representation layer.

Requirement 7 asks for graphical representation of data and analysis.
Rather than asking an LLM to "draw a chart" (unreliable and not really
possible), we compute derived metrics (growth, margins) in plain Python
from the verified FinancialFact table, and render charts with matplotlib
directly from that same verified data -- so the picture is guaranteed to
match the numbers.
"""
from __future__ import annotations

import io
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt

from app.extraction.pdf_extractor import FinancialFact


def facts_by_metric(facts: List[FinancialFact]) -> Dict[str, List[FinancialFact]]:
    grouped: Dict[str, List[FinancialFact]] = defaultdict(list)
    for f in facts:
        grouped[f.metric].append(f)
    return grouped


def compute_key_metrics(facts: List[FinancialFact]) -> dict:
    """
    Derived analytics per metric.

    For time-series metrics (company income statement / balance sheet
    line items across quarters/years), this includes period-over-period
    growth. For per-holding portfolio metrics (mutual fund market value,
    cost value, NAV, unit balance -- one value per scheme, not per time
    period) a growth % between consecutive table rows would compare two
    unrelated funds and is actively misleading, so those instead get
    portfolio-level stats (count, highest, lowest holding).
    """
    grouped = facts_by_metric(facts)
    summary = {}
    for metric, items in grouped.items():
        is_per_holding = metric.startswith("portfolio_")
        with_period = [f for f in items if f.period]
        no_period = [f for f in items if not f.period]

        if with_period:
            latest = with_period[-1]
            entry = {
                "latest_value": latest.value,
                "latest_period": latest.period,
                "unit": latest.unit,
                "history": [{"period": f.period, "value": f.value} for f in with_period],
            }
            if is_per_holding:
                values = [f.value for f in with_period]
                best = max(with_period, key=lambda f: f.value)
                worst = min(with_period, key=lambda f: f.value)
                entry["holding_count"] = len(with_period)
                entry["highest"] = {"period": best.period, "value": best.value}
                entry["lowest"] = {"period": worst.period, "value": worst.value}
            elif len(with_period) >= 2 and with_period[-2].value != 0:
                prev = with_period[-2]
                growth_pct = ((latest.value - prev.value) / abs(prev.value)) * 100
                entry["period_over_period_growth_pct"] = round(growth_pct, 2)
            summary[metric] = entry
        elif no_period:
            # Single aggregate value with no period breakdown (e.g. a
            # portfolio "Total" row, or a metric that only appears once).
            latest = no_period[-1]
            summary[metric] = {
                "latest_value": latest.value,
                "latest_period": None,
                "unit": latest.unit,
            }

    # Derived margins if the raw components are present
    if "revenue" in summary and "gross_profit" in summary:
        rev = summary["revenue"]["latest_value"]
        gp = summary["gross_profit"]["latest_value"]
        if rev:
            summary["gross_margin_pct_computed"] = {"latest_value": round(gp / rev * 100, 2)}
    if "revenue" in summary and "net_income" in summary:
        rev = summary["revenue"]["latest_value"]
        ni = summary["net_income"]["latest_value"]
        if rev:
            summary["net_margin_pct_computed"] = {"latest_value": round(ni / rev * 100, 2)}

    return summary


def _short_label(label: str, max_len: int = 32) -> str:
    return label if len(label) <= max_len else label[: max_len - 1].rstrip() + "…"


def render_metric_trend_chart(metric: str, facts: List[FinancialFact]) -> bytes:
    """
    Render a chart of a metric's value across periods (company statements)
    or across holdings (portfolio statements). Returns PNG bytes.

    Company-report metrics have a handful of short labels (Q1 FY24, FY24)
    so a vertical bar chart reads fine. Per-holding portfolio metrics can
    have a dozen-plus long scheme names, which would overlap and become
    unreadable as vertical axis labels -- those are rendered as a
    horizontal bar chart with truncated labels instead.
    """
    items = [f for f in facts if f.metric == metric and f.period]
    if not items:
        raise ValueError(f"No period-labelled data found for metric '{metric}'.")

    labels = [f.period for f in items]
    values = [f.value for f in items]
    title = metric.replace("_", " ").title()
    unit_label = items[0].unit or "Value"

    use_horizontal = len(items) > 6 or max(len(l) for l in labels) > 14

    if use_horizontal:
        short_labels = [_short_label(l) for l in labels]
        # Largest value at the top of the chart
        order = sorted(range(len(values)), key=lambda i: values[i])
        short_labels = [short_labels[i] for i in order]
        values_sorted = [values[i] for i in order]

        fig_height = max(3.5, 0.45 * len(items) + 1)
        fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=140)
        bars = ax.barh(short_labels, values_sorted, color="#2563eb")
        ax.set_title(f"{title} by Holding")
        ax.set_xlabel(unit_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, value in zip(bars, values_sorted):
            ax.annotate(
                f"{value:,.0f}",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
    else:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
        bars = ax.bar(labels, values, color="#2563eb")
        ax.set_title(f"{title} by Period")
        ax.set_ylabel(unit_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:,.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_multi_metric_comparison(metrics: List[str], facts: List[FinancialFact]) -> bytes:
    """Render a grouped bar chart comparing several metrics across shared periods."""
    grouped = facts_by_metric(facts)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    all_periods: List[str] = []
    for m in metrics:
        for f in grouped.get(m, []):
            if f.period and f.period not in all_periods:
                all_periods.append(f.period)

    width = 0.8 / max(len(metrics), 1)
    x = range(len(all_periods))

    for i, metric in enumerate(metrics):
        values_by_period = {f.period: f.value for f in grouped.get(metric, [])}
        values = [values_by_period.get(p, 0) for p in all_periods]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values, width=width, label=metric.replace("_", " ").title())

    ax.set_xticks([xi + width * (len(metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels(all_periods)
    ax.set_title("Metric Comparison")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
