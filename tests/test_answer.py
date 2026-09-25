"""Тесты ответов: извлекательный режим, контекст, структура источников."""
from logdoc.answer import Answer, build_context, extractive_answer
from logdoc.chunking import Chunk
from logdoc.store import Hit


def make_hits():
    return [
        Hit(Chunk(file="akt.txt", page=1, text="Акт подписал Рустам Валеев"), 0.82),
        Hit(Chunk(file="dogovor.txt", page=4, text="Пеня 0,5% за каждый день просрочки"), 0.61),
    ]


def test_build_context_numbered_with_pages():
    context = build_context(make_hits())
    assert "[1] akt.txt, стр. 1:" in context
    assert "[2] dogovor.txt, стр. 4:" in context


def test_extractive_answer_contains_citations():
    answer = extractive_answer("кто подписал акт?", make_hits())
    assert isinstance(answer, Answer)
    assert "[1]" in answer.text and "[2]" in answer.text
    assert answer.sources[0]["file"] == "akt.txt"
    assert answer.sources[0]["page"] == 1


def test_extractive_answer_empty_hits():
    answer = extractive_answer("вопрос", [])
    assert answer.sources == []
    assert "не найдено" in answer.text.lower() or "Индекс пуст" in answer.text


def test_long_snippets_truncated():
    hits = [Hit(Chunk(file="f.txt", page=1, text="х" * 1000), 0.9)]
    answer = extractive_answer("q", hits)
    assert "…" in answer.text
    assert len(answer.sources[0]["snippet"]) == 200
