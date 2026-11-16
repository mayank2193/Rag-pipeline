"""PDF RAG pipeline CLI.

    uv run rag ingest [PATH]        # PDF file or folder (default: ./data)
    uv run rag search "query"       # vector search only, no LLM
    uv run rag ask "question"       # retrieval + LLM answer
    uv run rag chat                 # interactive prompt loop
"""

import argparse


def _print_sources(sources) -> None:
    print("\nSources:")
    for d in sources:
        print(f"  - {d.metadata.get('source_file')} p.{d.metadata.get('page', 0) + 1}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="PDF RAG pipeline (LangChain + Chroma)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Run the data ingestion pipeline")
    p_ingest.add_argument("path", nargs="?", help="PDF file or directory (default: ./data)")

    for name, help_ in [("search", "Vector search only"), ("ask", "Retrieve context and answer with the LLM")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("query")
        p.add_argument("-k", type=int, default=None, help="Number of chunks to retrieve")

    p_chat = sub.add_parser("chat", help="Interactive Q&A loop")
    p_chat.add_argument("-k", type=int, default=None)

    args = parser.parse_args()

    # Imports are lazy so `rag --help` stays fast.
    from rag import config

    k = getattr(args, "k", None) or config.TOP_K

    if args.cmd == "ingest":
        from rag.ingestion import run_ingestion

        run_ingestion(args.path)

    elif args.cmd == "search":
        from rag.retrieval import search

        for doc, score in search(args.query, k):
            print(f"\n[{score:.3f}] {doc.metadata.get('source_file')} p.{doc.metadata.get('page', 0) + 1}")
            print(doc.page_content[:400].replace("\n", " ") + " ...")

    elif args.cmd == "ask":
        from rag.retrieval import ask

        result = ask(args.query, k)
        print(result["answer"])
        _print_sources(result["sources"])

    elif args.cmd == "chat":
        from rag.retrieval import build_rag_chain

        chain = build_rag_chain(k)
        print(f"RAG chat ({config.LLM_PROVIDER}:{config.LLM_MODEL}). Type 'exit' to quit.")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit", "q"}:
                break
            if not q:
                continue
            result = chain.invoke(q)
            print(f"\nAssistant: {result['answer']}")
            _print_sources(result["sources"])
