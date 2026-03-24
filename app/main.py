from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from models import QueryRequest, QueryResponse, Source
from rag_pipeline import RAGPipeline
import time

# ── Lifespan (startup/shutdown) ────────────────────────────
# This runs ONCE when the server starts
# We load the RAG pipeline here so it's ready for all requests
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    global pipeline
    print("🚀 Starting up...")
    pipeline = RAGPipeline()  # loads models, connects ChromaDB
    yield
    # SHUTDOWN
    print("👋 Shutting down...")

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="RAG API",
    description="Local RAG pipeline powered by Ollama phi3 + ChromaDB",
    version="1.0.0"
)

# ── Endpoints ──────────────────────────────────────────────

@app.get("/")
async def root():
    """Just confirms the API is running"""
    return {
        "message": "RAG API is running!",
        "model":   "phi3",
        "status":  "ready"
    }

@app.get("/health")
async def health():
    """
    Health check endpoint
    In production this would check:
    - Is Ollama running?
    - Is ChromaDB connected?
    - How many documents are indexed?
    """
    try:
        doc_count = pipeline.collection.count()
        return {
            "status":     "healthy",
            "model":      "phi3",
            "documents":  doc_count,
            "ollama_url": "http://localhost:11434"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Main RAG endpoint
    
    Send a question → get answer + sources + confidence score
    
    Example request:
    {
        "question": "What is the refund policy?",
        "top_k": 5,
        "min_score": 0.2
    }
    """
    try:
        # Track response time (good for monitoring)
        start = time.time()

        result = pipeline.answer(
            question=req.question,
            top_k=req.top_k,
            min_score=req.min_score
        )

        latency = round(time.time() - start, 2)
        print(f"⏱️  Query answered in {latency}s")

        # Build response
        sources = [Source(**s) for s in result["sources"]]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            confidence=result["confidence"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest():
    """
    Re-ingests all documents from data/ folder
    
    Use this when you add new documents:
    1. Drop new PDF/text files in data/pdf or data/text_files
    2. Call POST /ingest
    3. New documents are now searchable
    """
    try:
        # Clear existing collection
        pipeline.chroma.delete_collection("rag_docs")
        pipeline.collection = pipeline.chroma.get_or_create_collection("rag_docs")

        # Reload everything
        pipeline._load_documents()

        return {
            "status":  "success",
            "message": "Documents reingested",
            "count":   pipeline.collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
