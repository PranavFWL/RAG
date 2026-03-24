from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
from models import QueryRequest, QueryResponse, Source
from rag_pipeline import RAGPipeline
from chat import get_history, add_message, clear_history, build_prompt_with_history
import time
import uuid

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Starting up...")
    pipeline = RAGPipeline()
    yield
    print("Shutting down...")

app = FastAPI(
    lifespan=lifespan,
    title="RAG Chatbot API",
    description="Local RAG pipeline powered by Ollama phi3 + ChromaDB",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    try:
        doc_count = pipeline.collection.count()
        return {
            "status":    "healthy",
            "model":     "phi3",
            "documents": doc_count,
            "ollama":    "http://localhost:11434"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        start = time.time()
        result = pipeline.answer(
            question=req.question,
            top_k=req.top_k,
            min_score=req.min_score
        )
        latency = round(time.time() - start, 2)
        print(f"Query answered in {latency}s")
        sources = [Source(**s) for s in result["sources"]]
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.2

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        start = time.time()
        results = pipeline.retrieve(req.question, req.top_k, req.min_score)
        if not results:
            return {
                "answer":     "I don't know based on the documents.",
                "session_id": session_id,
                "sources":    [],
                "confidence": 0.0
            }
        context = "\n\n".join([r["content"] for r in results])
        prompt = build_prompt_with_history(context, req.question, session_id)
        response = pipeline.llm.invoke(prompt)
        answer = response.content
        add_message(session_id, "user", req.question)
        add_message(session_id, "assistant", answer)
        latency = round(time.time() - start, 2)
        print(f"Chat answered in {latency}s")
        return {
            "answer":     answer,
            "session_id": session_id,
            "sources":    [r["metadata"].get("source", "unknown") for r in results],
            "confidence": round(max([r["score"] for r in results]), 3),
            "latency":    latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    clear_history(session_id)
    return {"message": f"History cleared for session {session_id}"}

@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    return {"history": get_history(session_id)}

@app.post("/ingest")
async def ingest():
    try:
        pipeline.chroma.delete_collection("rag_docs")
        pipeline.collection = pipeline.chroma.get_or_create_collection("rag_docs")
        pipeline._load_documents()
        return {
            "status":  "success",
            "message": "Documents reingested",
            "count":   pipeline.collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
