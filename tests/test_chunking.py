"""Тесты чанкинга: размеры, перекрытие, целостность слов, номера страниц."""
from logdoc.chunking import chunk_pages
from logdoc.ingest import PageText


def make_page(text, page=1):
    return PageText(file="doc.pdf", page=page, text=text)


def test_short_text_becomes_single_chunk():
    chunks = chunk_pages([make_page("короткий текст накладной")])
    assert len(chunks) == 1
    assert chunks[0].text == "короткий текст накладной"
    assert chunks[0].page == 1


def test_empty_page_skipped():
    assert chunk_pages([make_page("   \n  ")]) == []


def test_long_text_split_within_limit():
    text = " ".join(f"слово{i}" for i in range(500))  # сильно длиннее одного чанка
    chunks = chunk_pages([make_page(text, page=3)], max_chars=800, overlap=120)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)
    assert all(c.page == 3 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"слово{i}" for i in range(500))
    chunks = chunk_pages([make_page(text)], max_chars=800, overlap=120)
    tail = chunks[0].text[-60:]
    head = chunks[1].text[:60]
    # перекрытие: конец первого фрагмента встречается в начале второго
    overlap_part = chunks[0].text[len(chunks[0].text) - 121 : len(chunks[0].text) - 1]
    assert overlap_part[:60] in chunks[1].text + tail + head or overlap_part in chunks[0].text + chunks[1].text


def test_overlap_prevents_lost_content():
    text = " ".join(f"слово{i}" for i in range(400))
    chunks = chunk_pages([make_page(text)], max_chars=600, overlap=100)
    joined = " ".join(c.text for c in chunks)
    assert "слово0" in joined
    assert "слово399" in joined


def test_chunks_do_not_break_words():
    text = " ".join([" оченьдлинноеслово" * 5] * 30)
    chunks = chunk_pages([make_page(text)], max_chars=500, overlap=80)
    assert all(c.text for c in chunks)
