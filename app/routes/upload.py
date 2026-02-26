from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.services.pdf_parser import extract_text_from_pdf
from app.services.supabase_service import store_clinical_note


router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/")
async def upload_note(
    file: Optional[UploadFile] = None,
    text: Optional[str] = Form(None),
):
    raw_text: str = ""
    file_name: str = ""

    if file is not None:
        file_bytes = await file.read()
        file_name = file.filename or "uploaded_file"

        if file_name.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_bytes)
        else:
            raw_text = file_bytes.decode("utf-8", errors="ignore")

    elif text is not None:
        raw_text = text
        file_name = "pasted_text"
    else:
        raise HTTPException(status_code=400, detail="No file or text provided")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text")

    result = store_clinical_note(file_name, raw_text)

    return {
        "note_id": result.get("id"),
        "file_name": file_name,
        "raw_text": raw_text,
    }
