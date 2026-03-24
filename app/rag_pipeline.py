import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any

load_dotenv()

# ── paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR  = BASE_DIR / "data" / "pdf"
TXT_DIR  = BASE_DIR / "data" / "text_files"
DB_DIR   = BASE_DIR / "data" / "vector_store"

class RAGPipeline:
    """
    One class that handles everything:
    load docs → chunk → embed → store → retrieve → answer
    """

    def __init__(self):
        print("🚀 Starting RAG Pipeline...")

        # Step 1 - LLM (local Ollama)
        self.llm = ChatOllama(
            model="phi3",
            base_url="http://localhost:11434"
        )
        print("✅ Ollama phi3 connected")

        # Step 2 - Embedding model (local, no API needed)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")

        # Step 3 - ChromaDB vector store
        self.chroma = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.chroma.get_or_create_collection("rag_docs")
        print("✅ ChromaDB connected")

        # Step 4 - Load and index documents
        self._load_documents()
        print("✅ RAG Pipeline ready!\n")

    # ── Document Loading ───────────────────────────────────
    def _load_documents(self):
        """Load PDFs and text files, chunk them, store in ChromaDB"""

        # Skip if already indexed
        if self.collection.count() > 0:
            print(f"📚 Found {self.collection.count()} chunks already indexed, skipping reload")
            return

        docs = []

        # Load PDFs
        if PDF_DIR.exists():
            pdf_loader = DirectoryLoader(
                str(PDF_DIR),
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader
            )
            docs.extend(pdf_loader.load())
            print(f"📄 Loaded {len(docs)} PDF pages")

        # Load text files
        if TXT_DIR.exists():
            for txt_file in TXT_DIR.glob("*.txt"):
                txt_loader = TextLoader(str(txt_file), encoding="utf-8")
                docs.extend(txt_loader.load())
            print(f"📝 Loaded text files")

        if not docs:
            print("⚠️ No documents found!")
            return

        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        print(f"✂️  Split into {len(chunks)} chunks")

        # Embed and store
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print(f"💾 Stored {len(chunks)} chunks in ChromaDB")

    # ── Retrieval ──────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.2) -> List[Dict]:
        """Find most relevant chunks for a query"""

        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            # Convert distance to similarity score
            score = 1 - results["distances"][0][i]

            if score >= min_score:
                retrieved.append({
                    "content":  results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score":    round(score, 3)
                })

        return retrieved

    # ── Answer Generation ──────────────────────────────────
    def answer(self, question: str, top_k: int = 5, min_score: float = 0.2) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve → build prompt → generate answer"""

        # Retrieve relevant chunks
        results = self.retrieve(question, top_k, min_score)

        if not results:
            return {
                "answer":     "No relevant information found in documents.",
                "sources":    [],
                "confidence": 0.0
            }

        # Build context from retrieved chunks
        context = "\n\n".join([r["content"] for r in results])

        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:"""

        # Generate answer with Ollama
        response = self.llm.invoke(prompt)

        # Build sources list
        sources = [{
            "source":  r["metadata"].get("source", "unknown"),
            "page":    str(r["metadata"].get("page", "unknown")),
            "score":   r["score"],
            "preview": r["content"][:200] + "..."
        } for r in results]

        return {
            "answer":     response.content,
            "sources":    sources,
            "confidence": round(max([r["score"] for r in results]), 3)
        }
