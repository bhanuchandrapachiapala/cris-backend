from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.upload import router as upload_router
from app.routes.analyze import router as analyze_router
from app.routes.chat import router as chat_router


app = FastAPI(
    title="CRIS - Clinical Records Intelligence System",
    description="AI-powered clinical document analysis with NER, summarization, and RAG-based Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "CRIS - Clinical Records Intelligence System is running"}
