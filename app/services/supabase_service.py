from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from app.config import SUPABASE_KEY, SUPABASE_URL


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore[arg-type]


def store_clinical_note(file_name: str, raw_text: str) -> Dict[str, Any]:
    response = (
        supabase.table("clinical_notes")
        .insert({"file_name": file_name, "raw_text": raw_text})
        .execute()
    )
    data: Optional[List[Dict[str, Any]]] = getattr(response, "data", None)
    return data[0] if data else {}


def get_clinical_note(note_id: str) -> Dict[str, Any]:
    response = (
        supabase.table("clinical_notes")
        .select("*")
        .eq("id", note_id)
        .single()
        .execute()
    )
    data: Optional[Dict[str, Any]] = getattr(response, "data", None)
    return data or {}


def update_note_analysis(note_id: str, entities: Dict[str, Any], summary: str) -> None:
    (
        supabase.table("clinical_notes")
        .update({"entities": entities, "summary": summary})
        .eq("id", note_id)
        .execute()
    )


def store_embedding(note_id: str, chunk_text: str, embedding: List[float]) -> Dict[str, Any]:
    response = (
        supabase.table("embeddings")
        .insert(
            {
                "note_id": note_id,
                "chunk_text": chunk_text,
                "embedding": embedding,
            }
        )
        .execute()
    )
    data: Optional[List[Dict[str, Any]]] = getattr(response, "data", None)
    return data[0] if data else {}


def search_similar_chunks(
    query_embedding: List[float], note_id: str, top_k: int = 3
) -> List[Dict[str, Any]]:
    response = supabase.rpc(
        "match_embeddings",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "filter_note_id": note_id,
        },
    ).execute()

    data: Optional[List[Dict[str, Any]]] = getattr(response, "data", None)
    return data or []
