"""
Security & data-handling helpers.

Covers the "basic validation and secure processing" requirement:
  - real file-type verification (magic bytes, not just the extension/MIME
    header the client claims)
  - upload size limits
  - filename sanitisation (no path traversal)
  - optional shared-secret API key auth
  - safe, unguessable storage IDs
"""
from __future__ import annotations

import re
import uuid

from fastapi import Header, HTTPException, UploadFile, status

from app import config

PDF_MAGIC_BYTES = b"%PDF-"


def new_report_id() -> str:
    return uuid.uuid4().hex


def sanitize_filename(filename: str) -> str:
    """Strip path components and disallowed characters from a client-supplied filename."""
    name = filename.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:200] or "report.pdf"


async def validate_pdf_upload(file: UploadFile) -> bytes:
    """
    Read and validate an uploaded file. Raises HTTPException on any
    validation failure. Returns the raw bytes on success so the caller
    doesn't need to re-read the stream.
    """
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. Only application/pdf is accepted.",
        )

    filename = sanitize_filename(file.filename or "report.pdf")
    if not filename.lower().endswith(config.ALLOWED_EXTENSION):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf files are accepted.",
        )

    contents = await file.read()

    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {config.MAX_UPLOAD_MB}MB upload limit.",
        )

    if not contents.startswith(PDF_MAGIC_BYTES):
        # The client can claim any content-type/extension it likes; this
        # checks the actual file signature so a renamed .exe etc. is rejected.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid PDF (bad file signature).",
        )

    return contents


async def require_api_key(x_api_key: str = Header(default=None, alias=config.API_KEY_HEADER_NAME)):
    """
    FastAPI dependency enforcing the shared-secret API key when
    FINQA_API_KEY is set in the environment. No-ops (open access) when
    unset, so the demo works out of the box.
    """
    if not config.API_KEY:
        return
    if not x_api_key or x_api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")
