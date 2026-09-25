"""Тесты хранилища: детерминизм эмбеддера, релевантность поиска, персистентность."""
import pytest

from logdoc.chunking import Chunk
from logdoc.store import HashingEmbedder, VectorStore


def make_store():
    chunks = [
        Chunk(file="akt.txt", page=1, text="Акт приёмки подписал начальник склада Рустам Валеев"),
        Chunk(file="dogovor.txt", page=2, text="Температурный режим для скоропортящихся грузов от +2 до +6 градусов"),
        Chunk(file="nakladnaya.txt", page=1, text="Перевозчик ООО ТрансЛайн, водитель Смирнов, 240 мест груза"),
    ]
    return VectorStore.build(chunks, HashingEmbedder(dim=512))


def test_embedder_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=256)
    a = embedder.embed("накладная подписана")
    b = embedder.embed("накладная подписана")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_search_ranks_relevant_chunk_first():
    store = make_store()
    embedder = HashingEmbedder(dim=512)
    hits = store.search("кто подписал акт приёмки", embedder, k=3)
    assert hits[0].chunk.file == "akt.txt"


def test_search_respects_k():
    store = make_store()
    embedder = HashingEmbedder(dim=512)
    assert len(store.search("груз перевозка документы", embedder, k=2)) == 2
    assert len(store.search("груз перевозка документы", embedder, k=5)) == 3


def test_search_on_empty_store():
    store = VectorStore(embedder_name="hashing-ngrams", dim=512)
    assert store.search("что угодно", HashingEmbedder(dim=512)) == []


def test_mismatched_embedder_rejected():
    store = make_store()
    other = HashingEmbedder(dim=64)  # другая размерность
    with pytest.raises(ValueError):
        store.add_chunks([Chunk(file="x", page=1, text="текст")], other)


def test_save_load_roundtrip(tmp_path):
    store = make_store()
    path = tmp_path / "index.json"
    store.save(path)
    loaded = VectorStore.load(path)
    assert loaded.embedder_name == store.embedder_name
    assert len(loaded.chunks) == len(store.chunks)
    embedder = HashingEmbedder(dim=512)
    assert loaded.search("акт подписал", embedder)[0].chunk.file == "akt.txt"
