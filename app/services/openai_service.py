import hashlib
import json
import random
import re
import struct
from typing import Any, Dict, List

from openai import OpenAI

from app.config import GROQ_API_KEY


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

CHAT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_DIM = 1536


def _get_chat_response(system_message: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


def _fake_embedding(text: str) -> List[float]:
    """Deterministic 1536-dim vector from text (placeholder; Groq has no embeddings API)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = struct.unpack(">Q", h[:8])[0] ^ struct.unpack(">Q", h[8:16])[0]
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


def extract_entities(clinical_text: str) -> Dict[str, Any]:
    default_entities: Dict[str, Any] = {
        "diagnoses": [],
        "medications": [],
        "lab_values": [],
        "procedures": [],
    }

    system_message = (
        'You are a clinical NLP engine. Extract medical entities from the given clinical note. '
        'Return ONLY valid JSON with no markdown formatting, no backticks. '
        'Return this exact structure: '
        '{"diagnoses": [...], "medications": [...], "lab_values": [...], "procedures": [...]}'
    )

    content = _get_chat_response(system_message, clinical_text)

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return default_entities
        for key in default_entities.keys():
            if key not in parsed or not isinstance(parsed[key], list):
                parsed[key] = []
        return parsed
    except Exception:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return default_entities
            for key in default_entities.keys():
                if key not in parsed or not isinstance(parsed[key], list):
                    parsed[key] = []
            return parsed
        except Exception:
            return default_entities


def generate_summary(clinical_text: str) -> str:
    system_message = (
        "You are a clinical summarization engine. Summarize the following clinical note into a "
        "structured format with these sections: Chief Complaint, Diagnosis, Treatment Plan, "
        "Medications, Follow-up. Keep it concise."
    )
    return _get_chat_response(system_message, clinical_text)


def generate_embedding(text: str) -> List[float]:
    return _fake_embedding(text)


def chat_with_context(question: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    system_message = (
        "You are a helpful clinical assistant. Answer the question based ONLY on the provided "
        "clinical context. If the answer is not in the context, say so."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    return _get_chat_response(system_message, user_message)
