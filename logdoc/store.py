"""Векторное хранилище на чистом Python: эмбеддинг + косинусный поиск.

MVP-философия: никакого обязательного ML-стека. Дефолтный HashingEmbedder
(хэширование символьных n-грамм и слов) детерминирован и не требует ничего,
кроме стандартной библиотеки, — поиск с цитатами работает сразу.
SentenceTransformerEmbedder подключается опционально для семантического поиска
(см. README): качество выше, но нужен установленный sentence-transformers.
"""
import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from logdoc.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HashingEmbedder:
    """Текст → нормированный разреженный вектор фиксированной размерности.

    Признаки: слова и символьные n-граммы; каждое хэшируется в одну из
    dim корзин (hashing trick). Работает для русского и английского.
    """

    name = "hashing-ngrams"

    def __init__(self, dim: int = 1024, ngram_sizes: tuple[int, ...] = (3, 4)):
        self.dim = dim
        self.ngram_sizes = ngram_sizes

    def _features(self, text: str) -> list[str]:
        features: list[str] = []
        for token in _TOKEN_RE.findall(text.lower()):
            features.append("w:" + token)
            padded = f" {token} "
            for size in self.ngram_sizes:
                for i in range(len(padded) - size + 1):
                    features.append(f"g:{padded[i:i + size]}")
        return features

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            bucket = int(hashlib.md5(feature.encode("utf-8")).hexdigest()[:8], 16)
            vector[bucket % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


class SentenceTransformerEmbedder:
    """Семантические эмбеддинги через sentence-transformers (опциональная зависимость).

    Для русского языка по умолчанию берётся компактная cointegrated/rubert-tiny2
    (~120 МБ): её хватает для поиска по документам, а моделируется всё локально.
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        from sentence_transformers import SentenceTransformer  # импорт только при использовании

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    embedder_name: str
    dim: int
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "VectorStore":
        store = cls(embedder_name=embedder.name, dim=embedder.dim)
        store.add_chunks(chunks, embedder)
        return store

    def add_chunks(self, chunks: list[Chunk], embedder) -> None:
        if embedder.name != self.embedder_name or embedder.dim != self.dim:
            raise ValueError("эмбеддер индекса не совпадает с текущим — пересоздайте индекс")
        for chunk in chunks:
            self.chunks.append(chunk)
            self.vectors.append(embedder.embed(chunk.text))

    def search(self, query: str, embedder, k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        query_vector = embedder.embed(query)
        scored = [
            Hit(chunk, cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def save(self, path) -> None:
        payload = {
            "embedder_name": self.embedder_name,
            "dim": self.dim,
            "chunks": [chunk.__dict__ for chunk in self.chunks],
            "vectors": self.vectors,
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "VectorStore":
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        store = cls(embedder_name=payload["embedder_name"], dim=payload["dim"])
        store.chunks = [Chunk(**c) for c in payload["chunks"]]
        store.vectors = payload["vectors"]
        return store
