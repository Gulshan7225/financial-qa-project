"""
Generates sample_reports/Acme_Quarterly_Report.pdf -- a synthetic but
realistic-looking quarterly financial report containing:
  - narrative text sections
  - a ruled income statement table (4 quarters)
  - a ruled balance sheet table
  - an embedded chart image (revenue trend, rendered with matplotlib)

This lets the project be demoed end-to-end without needing a real
confidential financial report.
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_PATH = Path(__file__).parent / "Acme_Quarterly_Report.pdf"

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20)
h2 = ParagraphStyle("H2", parent=styles["Heading2"])
body = styles["BodyText"]


def make_revenue_chart_image() -> Image:
    quarters = ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]
    revenue = [42.5, 47.1, 51.8, 58.3]  # in $M

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot(quarters, revenue, marker="o", linewidth=2, color="#1d4ed8")
    ax.fill_between(quarters, revenue, alpha=0.1, color="#1d4ed8")
    ax.set_title("Quarterly Revenue Trend (FY24, $M)")
    ax.set_ylabel("Revenue ($M)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for x, y in zip(quarters, revenue):
        ax.annotate(f"${y}M", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=5.5 * inch, height=2.9 * inch)


def income_statement_table() -> Table:
    data = [
        ["Line Item", "Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"],
        ["Total Revenue", "$42.5M", "$47.1M", "$51.8M", "$58.3M"],
        ["Cost of Goods Sold", "$18.2M", "$19.9M", "$21.4M", "$23.7M"],
        ["Gross Profit", "$24.3M", "$27.2M", "$30.4M", "$34.6M"],
        ["Operating Expenses", "$14.1M", "$15.0M", "$16.2M", "$17.8M"],
        ["Operating Income", "$10.2M", "$12.2M", "$14.2M", "$16.8M"],
        ["Net Income", "$7.8M", "$9.1M", "$10.9M", "$13.2M"],
        ["Diluted EPS", "$0.42", "$0.49", "$0.58", "$0.70"],
    ]
    table = Table(data, colWidths=[1.8 * inch] + [1.0 * inch] * 4)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return table


def balance_sheet_table() -> Table:
    data = [
        ["Line Item", "FY23", "FY24"],
        ["Cash and Cash Equivalents", "$32.1M", "$41.6M"],
        ["Total Assets", "$210.4M", "$248.9M"],
        ["Total Liabilities", "$96.7M", "$104.2M"],
        ["Total Equity", "$113.7M", "$144.7M"],
        ["Total Debt", "$40.0M", "$35.5M"],
    ]
    table = Table(data, colWidths=[2.4 * inch, 1.3 * inch, 1.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return table


def build():
    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph("Acme Industries Inc.", title_style))
    story.append(Paragraph("Fiscal Year 2024 Quarterly Financial Report", h2))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "This report summarizes Acme Industries' financial performance for fiscal year 2024 (FY24). "
            "The company delivered consistent quarter-over-quarter growth in revenue and profitability, "
            "driven by strong demand in its core industrial products segment and disciplined cost management.",
            body,
        )
    )
    story.append(Spacer(1, 16))

    story.append(Paragraph("Management Commentary", h2))
    story.append(
        Paragraph(
            "Total revenue grew from $42.5M in Q1 to $58.3M in Q4, representing a 37.2% increase across the "
            "fiscal year. Gross margin improved from 57.2% in Q1 to 59.3% in Q4 due to procurement efficiencies. "
            "Net income more than expanded on a similar trajectory, reflecting operating leverage as the business "
            "scaled. The Board declared no special dividends during the period.",
            body,
        )
    )
    story.append(Spacer(1, 16))

    story.append(Paragraph("Quarterly Revenue Trend", h2))
    story.append(make_revenue_chart_image())
    story.append(Spacer(1, 16))

    story.append(PageBreak())
    story.append(Paragraph("Income Statement (Quarterly, FY24)", h2))
    story.append(Spacer(1, 8))
    story.append(income_statement_table())
    story.append(Spacer(1, 20))

    story.append(Paragraph("Balance Sheet Highlights (FY23 vs FY24)", h2))
    story.append(Spacer(1, 8))
    story.append(balance_sheet_table())
    story.append(Spacer(1, 20))

    story.append(Paragraph("Outlook", h2))
    story.append(
        Paragraph(
            "Management expects continued revenue growth in the 10-15% range per quarter into FY25, supported "
            "by a healthy order backlog and expansion into two new regional markets. Total debt was reduced from "
            "$40.0M to $35.5M during FY24 as part of the company's deleveraging strategy.",
            body,
        )
    )

    doc.build(story)
    print(f"Sample report written to {OUT_PATH}")


if __name__ == "__main__":
    build()
