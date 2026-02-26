from io import BytesIO

import pdfplumber


def extract_text_from_pdf(file_bytes: bytes) -> str:
    full_text_parts = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                full_text_parts.append(page_text.strip())

    return "\n\n".join(full_text_parts) if full_text_parts else ""
