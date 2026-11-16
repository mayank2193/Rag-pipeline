"""Data ingestion pipeline: PDF -> load -> parse/clean -> chunk -> embed -> Chroma vector DB."""

import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.embeddings import get_vector_store


# 1. Data ingestion: read PDFs
def load_pdfs(source: Path) -> list[Document]:
    """Load one PDF or every PDF under a directory. Returns one Document per page."""
    source = Path(source)
    pdf_files = [source] if source.is_file() else sorted(source.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found at {source}")

    documents: list[Document] = []
    for pdf in pdf_files:
        pages = PyPDFLoader(str(pdf)).load()
        for page in pages:
            page.metadata["source_file"] = pdf.name
        documents.extend(pages)
        print(f"[ingest] Loaded {len(pages):>3} pages from {pdf.name}")
    return documents


# 2. Data parsing: clean extracted text, drop empty pages
def parse_documents(documents: list[Document]) -> list[Document]:
    parsed = []
    for doc in documents:
        text = doc.page_content
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
        text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines
        text = text.strip()
        if text:
            parsed.append(Document(page_content=text, metadata=doc.metadata))
    print(f"[ingest] Parsed {len(parsed)} non-empty pages (dropped {len(documents) - len(parsed)})")
    return parsed


# 3. Chunking
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def _chunk_id(chunk: Document) -> str:
    """Deterministic ID so re-ingesting the same PDF updates chunks instead of duplicating them."""
    key = f"{chunk.metadata.get('source_file')}|{chunk.metadata.get('page')}|{chunk.metadata.get('start_index')}"
    return hashlib.sha256(key.encode()).hexdigest()


# 4 + 5. Embed and store in vector DB
def store_chunks(chunks: list[Document]) -> int:
    store = get_vector_store()
    ids = [_chunk_id(c) for c in chunks]
    print(f"[ingest] Embedding with {config.EMBEDDING_MODEL} and writing to Chroma at {config.CHROMA_DIR} ...")
    store.add_documents(chunks, ids=ids)  # embeds + upserts
    total = store._collection.count()
    print(f"[ingest] Stored {len(chunks)} chunks. Collection '{config.COLLECTION_NAME}' now holds {total}.")
    return total


def run_ingestion(source: Path | None = None) -> int:
    source = Path(source) if source else config.DATA_DIR
    docs = load_pdfs(source)
    docs = parse_documents(docs)
    chunks = split_documents(docs)
    return store_chunks(chunks)
