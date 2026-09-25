"""Извлечение текста из документов с сохранением номеров страниц.

Поддержка MVP: текстовые PDF (pypdf), .txt, .md.
Сканированные PDF без текстового слоя определяются на этапе загрузки —
OCR подключается опционально (см. «Развитие проекта» в README).
"""
from dataclasses import dataclass
from pathlib import Path

import pypdf

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass
class PageText:
    """Текст одной страницы документа."""

    file: str
    page: int  # нумерация с 1 — так цитируются бумажные документы
    text: str


def extract_pages(path: Path) -> list[PageText]:
    """Возвращает постраничный текст файла; неподдерживаемый формат — пустой список."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [PageText(path.name, 1, text)]
    return []


def _extract_pdf(path: Path) -> list[PageText]:
    reader = pypdf.PdfReader(str(path))
    return [
        PageText(path.name, number, (page.extract_text() or "").strip())
        for number, page in enumerate(reader.pages, start=1)
    ]


def iter_documents(directory: Path) -> list[Path]:
    """Собирает поддерживаемые файлы из каталога (без рекурсии)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
