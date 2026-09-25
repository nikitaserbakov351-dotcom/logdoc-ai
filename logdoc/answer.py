"""Формирование ответа на вопрос по найденным фрагментам.

Два режима:
- extractive (по умолчанию, полностью оффлайн) — возвращает наиболее
  релевантные фрагменты с цитатами. Честный «поиск без галлюцинаций».
- gemini — передаёт фрагменты как контекст в Google Gemini и просит
  сформулировать ответ, ссылаясь на источники [1], [2]…
"""
import os
from dataclasses import dataclass, field

from logdoc.store import Hit

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

SYSTEM_INSTRUCTION = (
    "Ты — ассистент по корпоративным документам логистики. "
    "Отвечай ТОЛЬКО на основе переданного контекста, ничего не выдумывай. "
    "На каждый факт ссылайся на источник в квадратных скобках, например [1]. "
    "Если в контексте нет данных для ответа — так и скажи: «В документах ответа нет»."
)


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)


def build_context(hits: list[Hit]) -> str:
    """Нумерованный контекст: [1] файл, стр. N: текст фрагмента."""
    return "\n\n".join(
        f"[{i}] {hit.chunk.file}, стр. {hit.chunk.page}:\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )


def extractive_answer(query: str, hits: list[Hit]) -> Answer:
    """Оффлайн-ответ: релевантные фрагменты как есть, без генерации."""
    if not hits:
        return Answer("Индекс пуст или ничего не найдено.", sources=[])

    lines = ["По документам наиболее релевантные фрагменты:"]
    for i, hit in enumerate(hits, start=1):
        snippet = hit.chunk.text if len(hit.chunk.text) <= 300 else hit.chunk.text[:300] + "…"
        lines.append(f"[{i}] {hit.chunk.file}, стр. {hit.chunk.page} — «{snippet}»")
    return Answer("\n\n".join(lines), sources=[_source(i, hit) for i, hit in enumerate(hits, start=1)])


def gemini_answer(query: str, hits: list[Hit], api_key: str | None = None, model: str = "gemini-3.5-flash-lite") -> Answer:
    import httpx  # сетевая зависимость нужна только в этом режиме

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Не задан GEMINI_API_KEY — используйте backend extractive или добавьте ключ в .env")

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": f"Контекст:\n\n{build_context(hits)}\n\nВопрос: {query}"}]}],
        "generationConfig": {"temperature": 0.2},
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(GEMINI_URL.format(model=model), params={"key": key}, json=payload)
        data = response.json()

    if "candidates" not in data:
        message = data.get("error", {}).get("message", "неизвестная ошибка")
        raise RuntimeError(f"Ошибка Gemini API: {message}")

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return Answer(text, sources=[_source(i, hit) for i, hit in enumerate(hits, start=1)])


def _source(number: int, hit: Hit) -> dict:
    return {
        "n": number,
        "file": hit.chunk.file,
        "page": hit.chunk.page,
        "score": round(hit.score, 4),
        "snippet": hit.chunk.text[:200],
    }
