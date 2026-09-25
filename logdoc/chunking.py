"""Разбиение страниц на перекрывающиеся фрагменты (чанки).

Границы слов стараются не рвать: если фрагмент обрезан посреди страницы,
хвост обрезается по последнему пробелу. Перекрытие сохраняет смысл
на стыках — фразу, разрезанную пополам, всё ещё можно найти по любой половине.
"""
from dataclasses import dataclass

from logdoc.ingest import PageText


@dataclass
class Chunk:
    file: str
    page: int
    text: str


def chunk_pages(pages: list[PageText], max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append(Chunk(page.file, page.page, text))
            continue

        start = 0
        while start < len(text):
            piece = text[start:start + max_chars]
            is_tail = start + max_chars >= len(text)
            if not is_tail:
                cut = piece.rfind(" ")
                if cut > overlap:
                    piece = piece[:cut]
            chunk = piece.strip()
            if chunk:
                chunks.append(Chunk(page.file, page.page, chunk))
            start += max(len(chunk) - overlap, 1)
    return chunks
