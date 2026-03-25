# RAG Chatbot — Local AI Document Assistant

A production-ready Retrieval-Augmented Generation (RAG) chatbot powered by **Ollama (phi3)**, **ChromaDB**, and **FastAPI**. Runs completely offline — no API keys, no cloud, no cost.

---

## What it does

Upload any documents (PDF, TXT) and ask questions about them. The system retrieves the most relevant sections and generates accurate answers using a local LLM.
```
User Question → Hybrid Search → Relevant Chunks → Ollama phi3 → Answer
```

---

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                    │
├─────────────┬───────────────────┬───────────────────┤
│   /chat     │    /query         │    /ingest        │
│  (stream)   │   (standard)      │  (add new docs)   │
└──────┬──────┴─────────┬─────────┴────────┬──────────┘
       │                │                  │
       ▼                ▼                  ▼
┌─────────────────────────────────────────────────────┐
│              RAG Pipeline                           │
│                                                     │
│  Hybrid Retrieval:                                  │
│  Dense Search (ChromaDB) + Sparse Search (BM25)     │
│  60% semantic + 40% keyword = best results          │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ChromaDB Vector Store      Ollama phi3 LLM
   (all-MiniLM-L6-v2)        (runs locally)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama phi3 (local, offline) |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Hybrid Search | BM25 + Dense Vector Search |
| API Framework | FastAPI |
| Document Loading | LangChain + PyMuPDF |
| Frontend | HTML/CSS/JS (served by FastAPI) |
| Containerization | Docker + Docker Compose |

---

## Project Structure
```
RAG/
├── app/
│   ├── main.py           # FastAPI endpoints
│   ├── rag_pipeline.py   # Core RAG logic
│   ├── models.py         # Pydantic schemas
│   ├── chat.py           # Conversation memory
│   └── static/
│       └── index.html    # Chatbot UI
├── data/
│   ├── pdf/              # PDF documents
│   └── text_files/       # Text documents
├── notebook/
│   └── document.ipynb    # RAG prototype notebook
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- phi3 model downloaded

### Step 1 — Clone the repo
```bash
git clone https://github.com/PranavFWL/RAG.git
cd RAG
```

### Step 2 — Install Ollama and pull model
```bash
ollama pull phi3
ollama serve
```

### Step 3 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Setup environment variables
```bash
cp .env.example .env
# Edit .env if needed
```

### Step 6 — Add your documents
```
Drop PDF or TXT files into:
data/pdf/         ← for PDF files
data/text_files/  ← for text files
```

### Step 7 — Run the app
```bash
cd app
uvicorn main:app --reload
```

### Step 8 — Open browser
```
http://localhost:8000        → Chatbot UI
http://localhost:8000/docs   → API Documentation
http://localhost:8000/health → System Health
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Chatbot UI |
| GET | `/health` | System health check |
| POST | `/chat` | Chat with memory |
| POST | `/chat/stream` | Streaming chat (word by word) |
| GET | `/chat/{session_id}/history` | Get conversation history |
| DELETE | `/chat/{session_id}` | Clear conversation |
| POST | `/query` | Single RAG query |
| POST | `/ingest` | Re-ingest all documents |

---

## Key Features

### 1. Hybrid Search
Combines dense vector search (semantic) with BM25 sparse search (keyword matching):
```
Final Score = 60% Dense Score + 40% BM25 Score
```
This ensures both semantic similarity and keyword matches are captured.

### 2. Streaming Responses
Responses stream word by word like ChatGPT using Server-Sent Events (SSE).

### 3. Conversational Memory
Each session maintains conversation history (last 3 exchanges) for context-aware answers.

### 4. Local & Private
- No API keys required
- No data sent to cloud
- Runs completely on your machine
- Ideal for sensitive/regulated environments (banking, healthcare)

### 5. Dynamic Document Ingestion
Add new documents at runtime without restarting:
1. Drop files in `data/pdf/` or `data/text_files/`
2. Call `POST /ingest`
3. New documents immediately searchable

---

## Docker Deployment
```bash
# Build image
docker build -t rag-api .

# Run with docker-compose
docker-compose up
```

---

## Evaluation

RAG pipeline evaluated using RAGAS metrics:

| Metric | Score |
|---|---|
| Faithfulness | TBD |
| Answer Relevancy | TBD |
| Context Recall | TBD |

*Run `python app/evaluate.py` to generate evaluation scores*
