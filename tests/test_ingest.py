"""Тесты загрузки документов: TXT/MD, неподдерживаемые форматы, сканы-заглушки."""
import pytest

from logdoc.ingest import extract_pages, iter_documents


def test_txt_extracted_with_page_one(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("Транспортная накладная № 1", encoding="utf-8")
    pages = extract_pages(file)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert "накладная" in pages[0].text


def test_md_supported(tmp_path):
    file = tmp_path / "doc.md"
    file.write_text("# Заголовок", encoding="utf-8")
    assert extract_pages(file)[0].text == "# Заголовок"


def test_unsupported_suffix_returns_empty(tmp_path):
    file = tmp_path / "table.xlsx"
    file.write_bytes(b"bin")
    assert extract_pages(file) == []


def test_pdf_without_text_layer_detected(tmp_path):
    """PDF без текстового слоя (скан-заглушка) — страницы извлекаются, но пустые."""
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    file = tmp_path / "scan.pdf"
    with open(file, "wb") as f:
        writer.write(f)
    pages = extract_pages(file)
    assert len(pages) == 1
    assert pages[0].text == ""


def test_iter_documents_filters_supported(tmp_path):
    (tmp_path / "a.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "c.exe").write_bytes(b"bin")
    names = [f.name for f in iter_documents(tmp_path)]
    assert names == ["a.txt", "b.pdf"]
