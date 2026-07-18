"""Tests for upload validation and filename sanitisation."""
import pytest

from app.security import sanitize_filename


def test_sanitize_filename_strips_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\evil.pdf") == "evil.pdf"


def test_sanitize_filename_strips_special_chars():
    assert sanitize_filename("my report (Q1)!.pdf") == "my_report__Q1__.pdf"


def test_sanitize_filename_handles_empty():
    assert sanitize_filename("") == "report.pdf"


@pytest.mark.asyncio
async def test_validate_pdf_upload_rejects_non_pdf_signature():
    from fastapi import HTTPException, UploadFile
    from io import BytesIO

    from app.security import validate_pdf_upload

    fake_file = UploadFile(filename="fake.pdf", file=BytesIO(b"NOT A PDF"))
    fake_file.headers = {"content-type": "application/pdf"}

    with pytest.raises(HTTPException) as exc_info:
        await validate_pdf_upload(fake_file)
    assert exc_info.value.status_code == 400
