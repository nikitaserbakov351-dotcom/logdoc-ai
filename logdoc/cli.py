"""CLI LogDoc AI: индексация документов и вопрос-ответ с цитатами.

Примеры:
    python -m logdoc.cli ingest ./samples --index index.json
    python -m logdoc.cli ask "кто подписал акт по заявке 152" --index index.json
    python -m logdoc.cli ask "условия хранения" --backend gemini
"""
import argparse
import sys
from pathlib import Path

from logdoc.answer import Answer, extractive_answer, gemini_answer
from logdoc.chunking import chunk_pages
from logdoc.ingest import extract_pages, iter_documents
from logdoc.store import VectorStore, resolve_embedder


def cmd_ingest(args) -> None:
    source = Path(args.source)
    files = iter_documents(source) if source.is_dir() else ([source] if source.is_file() else [])
    if not files:
        sys.exit(f"Нет поддерживаемых документов (.pdf/.txt/.md) в: {source}")

    chunks = []
    empty_text_files = []
    for file in files:
        pages = extract_pages(file)
        file_chunks = chunk_pages(pages, max_chars=args.max_chars, overlap=args.overlap)
        if not file_chunks and file.suffix.lower() == ".pdf":
            empty_text_files.append(file.name)
        chunks.extend(file_chunks)
        print(f"[logdoc] {file.name}: {len(pages)} стр. → {len(file_chunks)} фрагментов")

    if empty_text_files:
        print(
            "[logdoc] ВНИМАНИЕ: в PDF нет текстового слоя (сканы?): " + ", ".join(empty_text_files)
        )

    embedder = resolve_embedder(verbose=True)
    store = VectorStore.build(chunks, embedder)
    store.save(args.index)
    print(f"[logdoc] индекс сохранён: {args.index} ({len(store.chunks)} фрагментов)")


def cmd_ask(args) -> None:
    store = VectorStore.load(args.index)
    embedder = resolve_embedder(verbose=True)
    hits = store.search(args.query, embedder, k=args.k)

    if args.backend == "gemini":
        answer = gemini_answer(args.query, hits, model=args.model)
    else:
        answer = extractive_answer(args.query, hits)

    print()
    print(answer.text)
    if answer.sources:
        print("\nИсточники:")
        for source in answer.sources:
            print(f"  [{source['n']}] {source['file']}, стр. {source['page']} (score {source['score']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logdoc", description="RAG-ассистент по документам")
    parser.add_argument("--index", default="logdoc_index.json", help="файл индекса (по умолчанию logdoc_index.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="проиндексировать документ или каталог")
    p_ingest.add_argument("source", help="каталог или файл (.pdf/.txt/.md)")
    p_ingest.add_argument("--max-chars", type=int, default=800, help="максимальная длина фрагмента")
    p_ingest.add_argument("--overlap", type=int, default=120, help="перекрытие фрагментов")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="задать вопрос по проиндексированным документам")
    p_ask.add_argument("query", help="вопрос")
    p_ask.add_argument("-k", type=int, default=5, help="сколько фрагментов извлекать")
    p_ask.add_argument("--backend", choices=["extractive", "gemini"], default="extractive",
                       help="extractive — оффлайн-цитаты; gemini — генерация ответа (нужен GEMINI_API_KEY)")
    p_ask.add_argument("--model", default="gemini-3.5-flash-lite", help="модель Gemini для backend=gemini")
    p_ask.set_defaults(func=cmd_ask)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
