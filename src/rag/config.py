"""Central configuration. Every value can be overridden through environment variables or a .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# --- Ingestion ---
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "data"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# --- Embeddings (local, no API key needed) ---
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Vector DB (Chroma, persisted on disk) ---
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", PROJECT_ROOT / "vector_db"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "pdf_documents")

# --- Retrieval ---
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# --- LLM (any provider supported by langchain's init_chat_model) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or os.getenv("RAG_LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("RAG_LLM_MODEL", "claude-opus-5-5")
# Generic key; if unset, each provider falls back to its own env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)
LLM_API_KEY = os.getenv("LLM_API_KEY")
