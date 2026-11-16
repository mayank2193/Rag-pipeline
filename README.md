# PDF RAG pipeline (LangChain)

**Ingestion:** PDF → `PyPDFLoader` → text cleanup → `RecursiveCharacterTextSplitter` → HuggingFace embeddings (`all-MiniLM-L6-v2`, local) → Chroma vector DB (`./vector_db`)

**Retrieval:** user query → Chroma similarity search (top-k) → prompt with context → LLM (Claude via `langchain-anthropic`) → answer + sources

## Setup

```bash
uv sync
cp .env.example .env   # set LLM_PROVIDER, LLM_MODEL, LLM_API_KEY
```

## Usage

```bash
uv run rag ingest                 # ingest every PDF in ./data (or pass a file/folder)
uv run rag search "query"         # vector search only, no LLM
uv run rag ask "your question"    # retrieve + generate
uv run rag chat                   # interactive prompt loop
```

Re-ingesting is idempotent: chunk IDs are deterministic, so existing chunks are updated rather than duplicated.

## Code

| File | Role |
|---|---|
| `src/rag/config.py` | Settings (all overridable via env / `.env`) |
| `src/rag/embeddings.py` | Shared embedding model + Chroma handle |
| `src/rag/ingestion.py` | Load → parse → chunk → embed → store |
| `src/rag/retrieval.py` | Retriever + LCEL RAG chain |
| `src/rag/__init__.py` | CLI |

## Switching LLM

The LLM is created with LangChain's `init_chat_model`, so any provider works. Install its package and set:

```bash
RAG_LLM_PROVIDER=groq RAG_LLM_MODEL=llama-3.3-70b-versatile   # needs langchain-groq + GROQ_API_KEY
RAG_LLM_PROVIDER=openai RAG_LLM_MODEL=gpt-4o-mini             # needs langchain-openai + OPENAI_API_KEY
```
