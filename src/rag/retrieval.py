"""Retrieval pipeline: user query -> vector search in Chroma -> prompt with context -> LLM -> answer."""

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag import config
from rag.embeddings import get_vector_store

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context \
from the user's documents.

Rules:
- Answer only from the context. If the context does not contain the answer, say you don't know.
- Cite the sources you used as [file, page N].
- Be concise and well structured.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source_file', 'unknown')}, page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


def get_retriever(k: int = config.TOP_K):
    store = get_vector_store()
    if store._collection.count() == 0:
        raise RuntimeError("Vector DB is empty. Run the ingestion pipeline first: `uv run rag ingest`")
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_llm():
    kwargs = {"max_tokens": 4096}
    if config.LLM_API_KEY:
        kwargs["api_key"] = config.LLM_API_KEY
    return init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, **kwargs)


def build_rag_chain(k: int = config.TOP_K):
    """LCEL chain: {question} -> retrieve -> prompt -> LLM -> string.

    Returns a chain that outputs {"answer": str, "sources": list[Document]}.
    """
    retriever = get_retriever(k)
    llm = get_llm()

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=answer_chain)
        | (lambda x: {"answer": x["answer"], "sources": x["docs"]})
    )


def ask(question: str, k: int = config.TOP_K) -> dict:
    return build_rag_chain(k).invoke(question)


def search(question: str, k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """Raw vector search (no LLM) - useful for checking what the retriever finds."""
    return get_vector_store().similarity_search_with_relevance_scores(question, k=k)
