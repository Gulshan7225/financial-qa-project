"""
Central configuration for the Financial Report QA service.

All tunables live here so behaviour can be changed via environment
variables without touching code (12-factor style config).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Storage locations -----------------------------------------------
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---- Security ----------------------------------------------------------
# Simple shared-secret API key auth. Set FINQA_API_KEY in the environment
# to enable. If unset, auth is disabled (useful for local demo).
API_KEY = os.getenv("FINQA_API_KEY", "")
API_KEY_HEADER_NAME = "X-API-Key"

MAX_UPLOAD_MB = int(os.getenv("FINQA_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf"}
ALLOWED_EXTENSION = ".pdf"

# ---- LLM (optional enrichment layer) -----------------------------------
# The system is fully functional WITHOUT an LLM key: extraction is done
# with deterministic parsers (pdfplumber) and QA falls back to a
# retrieval + rule based engine. If ANTHROPIC_API_KEY is present, it is
# used to (a) describe/structure chart images and (b) produce a fluent,
# grounded natural-language answer on top of the retrieved, verified facts.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ENABLE_LLM = bool(ANTHROPIC_API_KEY)

# ---- Retrieval -----------------------------------------------------------
TOP_K_CHUNKS = int(os.getenv("FINQA_TOP_K", "5"))
