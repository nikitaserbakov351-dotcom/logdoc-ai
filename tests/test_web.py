"""Тесты веб-интерфейса: статус, вопрос-ответ через API, HTML-страница."""
import pytest
from fastapi.testclient import TestClient

from logdoc import web
from logdoc.chunking import Chunk
from logdoc.store import HashingEmbedder, VectorStore

CHUNKS = [
    Chunk(file="akt.txt", page=1, text="Акт приёмки подписал начальник склада Рустам Валеев"),
    Chunk(file="dogovor.txt", page=2, text="Пеня 0,5% за каждый день просрочки доставки"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    index = tmp_path_factory.mktemp("idx") / "index.json"
    VectorStore.build(CHUNKS, HashingEmbedder(dim=512)).save(index)
    web.os.environ["LOGDOC_INDEX"] = str(index)
    web.reset_store()
    with TestClient(web.app) as c:
        yield c


def test_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LogDoc AI" in response.text
    assert "text/html" in response.headers["content-type"]


def test_status_reports_index(client):
    data = client.get("/api/status").json()
    assert data["chunks"] == 2
    assert data["embedder"] == "hashing-ngrams"
    assert data["index"].endswith("index.json")


def test_ask_returns_answer_with_sources(client):
    response = client.post("/api/ask", json={"query": "кто подписал акт", "k": 2, "backend": "extractive"})
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "extractive"
    assert "[1]" in data["text"]
    assert data["sources"][0]["file"] == "akt.txt"


def test_ask_validates_request(client):
    bad = client.post("/api/ask", json={"query": "", "backend": "extractive"})
    assert bad.status_code == 422
    bad_backend = client.post("/api/ask", json={"query": "вопрос", "backend": "chatgpt"})
    assert bad_backend.status_code == 422


def test_ask_unknown_backend_error_is_http(client):
    # gemini без ключа → 502 с понятным текстом, а не падение сервера
    web.os.environ.pop("GEMINI_API_KEY", None)
    response = client.post("/api/ask", json={"query": "кто подписал акт", "backend": "gemini"})
    assert response.status_code == 502
    assert "GEMINI_API_KEY" in response.json()["detail"]
