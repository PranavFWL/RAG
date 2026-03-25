import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import chromadb
from typing import List, Dict, Any
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.llm = ChatOllama(
            model="phi3",
            base_url=ollama_host
        )
        print("✅ Ollama phi3 connected")

        # Step 2 - Embedding model (local, no API needed)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("✅ Embedding model loaded")

        # Step 3 - ChromaDB vector store
        self.chroma = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.chroma.get_or_create_collection("rag_docs")
        print("✅ ChromaDB connected")

        # Step 4 - Load documents and build indexes
        self._load_documents()
        self._build_bm25()
        print("✅ RAG Pipeline ready!\n")

    # ── Document Loading ───────────────────────────────────
    def _load_documents(self):
        """Load PDFs and text files, chunk them, store in ChromaDB"""

        # Check if all source files are indexed
        existing = self.collection.get(include=['metadatas'])
        indexed_sources = set([
            Path(m.get('source', '')).name 
            for m in existing['metadatas']
        ])
        available_sources = set([
            f.name for f in PDF_DIR.iterdir() 
            if f.suffix == '.pdf'
        ] + [
            f.name for f in TXT_DIR.iterdir() 
            if f.suffix == '.txt'
        ] if TXT_DIR.exists() else [
            f.name for f in PDF_DIR.iterdir() 
            if f.suffix == '.pdf'
        ])

        print(f"📚 Indexed: {indexed_sources}")
        print(f"📂 Available: {available_sources}")

        if indexed_sources >= available_sources and self.collection.count() > 0:
            print(f"📚 All files already indexed ({self.collection.count()} chunks), skipping reload")
            return
        else:
            missing = available_sources - indexed_sources
            print(f"🔄 Missing files detected: {missing}, reindexing...")
            self.chroma.delete_collection("rag_docs")
            self.collection = self.chroma.get_or_create_collection("rag_docs")

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

        # Fix metadata - ensure source is always set
        metadatas = []
        for c in chunks:
            meta = dict(c.metadata)
            if 'source' not in meta or not meta['source']:
                meta['source'] = 'unknown'
            # Convert all values to strings (ChromaDB requirement)
            meta = {k: str(v) for k, v in meta.items()}
            metadatas.append(meta)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

        print(f"💾 Stored {len(chunks)} chunks in ChromaDB")

    def _build_bm25(self):
        """Build BM25 index from stored chunks"""
        results = self.collection.get(include=["documents"])
        self.corpus = results["documents"]
        tokenized = [re.findall(r'\w+', doc.lower()) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✅ BM25 index built with {len(self.corpus)} documents")

    # ── Retrieval ──────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        Hybrid retrieval = Dense (ChromaDB) + Sparse (BM25)
        Combines semantic search with keyword search
        """

        # ── Dense retrieval (semantic) ─────────────────────
        query_embedding = self.embedder.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        dense_scores = {}
        for i in range(len(dense_results["documents"][0])):
            doc = dense_results["documents"][0][i]
            distance = dense_results["distances"][0][i]
            score = 1 / (1 + distance)
            dense_scores[doc] = {
                "score":    score,
                "metadata": dense_results["metadatas"][0][i],
                "content":  doc
            }

        # ── Sparse retrieval (BM25 keyword) ───────────────
        tokenized_query = re.findall(r'\w+', query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = bm25_scores / max_bm25

        # ── Combine scores (RRF fusion) ────────────────────
        combined = {}
        for i, doc in enumerate(self.corpus):
            dense_score = dense_scores.get(doc, {}).get("score", 0)
            sparse_score = float(bm25_normalized[i])

            # 60% dense + 40% sparse
            final_score = (0.6 * dense_score) + (0.4 * sparse_score)

            if final_score >= min_score:
                metadata = self.collection.get(
                    where={"source": doc[:50]}
                ) if doc not in dense_scores else dense_scores.get(doc, {}).get("metadata", {})

                combined[doc] = {
                    "content":  doc,
                    "score":    round(final_score, 3),
                    "metadata": dense_scores.get(doc, {}).get("metadata", {})
                }

        # Sort by score and return top_k
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        return sorted_results


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
        prompt = f"""You are a precise and helpful assistant for document analysis.

        INSTRUCTIONS:
        - Answer ONLY using the provided context
        - Be concise and direct
        - If asked for details, use bullet points
        - Always mention which document the answer comes from
        - If answer is not in context, say exactly: "This information is not available in the provided documents."
        - Never make up information

        CONTEXT FROM DOCUMENTS:
        {context}

        QUESTION: {question}

        ANSWER (be concise, cite document name):"""


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
