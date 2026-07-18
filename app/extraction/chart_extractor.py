"""
Chart / image extraction.

Financial reports commonly embed charts (bar/line graphs of revenue,
margins, etc.) as raster images rather than vector data, so there is no
ground-truth number to parse deterministically. Strategy:

  1. Crop every embedded image on the page using pdfplumber (deterministic,
     no LLM involved).
  2. If an LLM (vision-capable) is configured, ask it to describe the chart
     and extract approximate series values as JSON. This is explicitly
     tagged with confidence="low"/"medium" and is NEVER used to override a
     table-derived fact -- charts supplement narrative answers, they don't
     replace verified numbers.
  3. If no LLM is configured, we still surface the chart as an image asset
     plus a generic caption, so the pipeline degrades gracefully.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber

from app import config

logger = logging.getLogger("finqa.chart_extraction")


@dataclass
class ExtractedChart:
    chart_id: str
    page: int
    image_path: str
    description: str
    extracted_series: Optional[dict] = None
    confidence: str = "low"


def crop_page_images(file_path: str, output_dir: str) -> List[ExtractedChart]:
    """Crop every embedded image bounding box out of the PDF and save as PNG."""
    charts: List[ExtractedChart] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            for img in page.images:
                # Skip tiny images (likely logos/icons, not charts)
                width = img["x1"] - img["x0"]
                height = img["bottom"] - img["top"]
                if width < 80 or height < 80:
                    continue
                chart_id = str(uuid.uuid4())[:8]
                try:
                    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                    cropped = page.crop(bbox).to_image(resolution=200)
                    img_path = out_dir / f"chart_{chart_id}.png"
                    cropped.save(str(img_path))
                except Exception as exc:  # rendering can fail on odd PDFs
                    logger.warning("Could not crop image on page %s: %s", page_index, exc)
                    continue

                charts.append(
                    ExtractedChart(
                        chart_id=chart_id,
                        page=page_index,
                        image_path=str(img_path),
                        description="Chart image detected (not yet described).",
                        confidence="low",
                    )
                )
    return charts


def describe_chart_with_llm(chart: ExtractedChart) -> ExtractedChart:
    """
    Use a vision-capable LLM to turn the chart image into a structured
    description + best-effort data series. No-ops (returns chart unchanged)
    if no API key is configured, so the rest of the pipeline is unaffected.
    """
    if not config.ENABLE_LLM:
        chart.description = (
            "Chart image extracted from the report. Automatic chart-to-data "
            "description is disabled (no LLM API key configured); the image "
            "is available for manual review at the path above."
        )
        return chart

    try:
        import anthropic  # imported lazily so the base app has no hard dep

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        with open(chart.image_path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        prompt = (
            "This image is a chart cropped from a financial report. "
            "1) Write a 1-2 sentence caption describing what it shows. "
            "2) Extract the approximate data series as JSON with this shape: "
            '{"chart_type": "...", "x_label": "...", "y_label": "...", '
            '"series": [{"label": "...", "points": [{"x": "...", "y": number}]}]}. '
            "Only report numbers you can actually read off the chart (axis "
            "gridlines/labels/data labels) -- do not invent precision that "
            "isn't visually supported. Return your reply as: first line = "
            "caption, then a line '---JSON---', then the JSON object only."
        )

        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        raw_text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

        if "---JSON---" in raw_text:
            caption, json_part = raw_text.split("---JSON---", 1)
        else:
            caption, json_part = raw_text, "{}"

        chart.description = caption.strip() or "Chart described by LLM."
        try:
            json_str_match = re.search(r"\{.*\}", json_part, re.DOTALL)
            chart.extracted_series = json.loads(json_str_match.group(0)) if json_str_match else None
            chart.confidence = "medium" if chart.extracted_series else "low"
        except (json.JSONDecodeError, AttributeError):
            chart.extracted_series = None
            chart.confidence = "low"

    except Exception as exc:
        logger.warning("LLM chart description failed for %s: %s", chart.chart_id, exc)
        chart.description = (
            "Chart image extracted; automatic description failed. "
            "Image available for manual review."
        )

    return chart


def extract_charts(file_path: str, output_dir: str) -> List[ExtractedChart]:
    charts = crop_page_images(file_path, output_dir)
    return [describe_chart_with_llm(c) for c in charts]
