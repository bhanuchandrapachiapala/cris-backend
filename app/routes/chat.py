from fastapi import APIRouter
from pydantic import BaseModel

from app.services.openai_service import chat_with_context, generate_embedding
from app.services.supabase_service import search_similar_chunks


router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    note_id: str
    question: str


@router.post("/")
async def chat(body: ChatRequest):
    query_embedding = generate_embedding(body.question)
    results = search_similar_chunks(query_embedding, body.note_id, top_k=3)

    if not results:
        return {"answer": "No relevant context found for this note."}

    context_chunks = [item.get("chunk_text", "") for item in results]
    context_chunks = [c for c in context_chunks if c]

    if not context_chunks:
        return {"answer": "No relevant context found for this note."}

    answer = chat_with_context(body.question, context_chunks)
    return {"answer": answer, "context_chunks": context_chunks}
