from fastapi import APIRouter, HTTPException

from app.services.openai_service import (
    extract_entities,
    generate_embedding,
    generate_summary,
)
from app.services.supabase_service import (
    get_clinical_note,
    store_embedding,
    update_note_analysis,
)


router = APIRouter(prefix="/analyze", tags=["Analyze"])


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text by newlines into chunks of at most max_chars."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        add_len = (1 if current else 0) + len(line)
        if current_len + add_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += add_len

    if current:
        chunks.append("\n".join(current))

    return chunks


@router.post("/{note_id}")
async def analyze_note(note_id: str):
    note = get_clinical_note(note_id)
    if not note or not note.get("raw_text"):
        raise HTTPException(status_code=404, detail="Note not found")

    raw_text = note["raw_text"]

    entities = extract_entities(raw_text)
    summary = generate_summary(raw_text)
    update_note_analysis(note_id, entities, summary)

    chunks = chunk_text(raw_text, max_chars=500)
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        store_embedding(note_id, chunk, embedding)

    return {"note_id": note_id, "entities": entities, "summary": summary}
