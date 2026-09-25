"""Веб-интерфейс LogDoc AI: FastAPI-бэкенд и одностраничный UI.

Запуск:
    LOGDOC_INDEX=logdoc_index.json uvicorn logdoc.web:app --port 8100
или
    python -m logdoc.web [--index logdoc_index.json] [--port 8100]

UI доступен на http://127.0.0.1:8100
"""
import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from logdoc.answer import Answer, extractive_answer, gemini_answer
from logdoc.store import VectorStore, resolve_embedder

app = FastAPI(title="LogDoc AI")

_cached = {"path": None, "store": None, "embedder": None}


def get_index_path() -> Path:
    return Path(os.getenv("LOGDOC_INDEX", "logdoc_index.json"))


def get_store():
    """Ленивая загрузка индекса; пересоздаётся, если путь файла изменился."""
    path = get_index_path()
    if _cached["store"] is None or _cached["path"] != path:
        if not path.exists():
            raise FileNotFoundError(f"Индекс не найден: {path}. Сначала выполните: python -m logdoc.cli ingest ./samples")
        store = VectorStore.load(path)
        _cached["path"] = path
        _cached["store"] = store
        _cached["embedder"] = resolve_embedder()
    return _cached["store"], _cached["embedder"]


def reset_store() -> None:
    """Сброс кеша индекса (используется тестами)."""
    _cached.update({"path": None, "store": None, "embedder": None})


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    backend: str = Field(default="extractive", pattern="^(extractive|gemini)$")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/status")
def status():
    store, embedder = get_store()
    return {"chunks": len(store.chunks), "embedder": embedder.name, "index": str(get_index_path())}


@app.post("/api/ask")
def ask(request: AskRequest):
    store, embedder = get_store()
    hits = store.search(request.query, embedder, k=request.k)
    try:
        if request.backend == "gemini":
            answer = gemini_answer(request.query, hits)
        else:
            answer = extractive_answer(request.query, hits)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"backend": request.backend, **answer.__dict__}


HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LogDoc AI — вопрос-ответ по документам</title>
<style>
  :root {
    --bg: #070b14; --card: rgba(255,255,255,.04); --border: rgba(255,255,255,.1);
    --text: #e8edf6; --muted: #97a3b8; --accent: #38bdf8; --green: #34d399;
    --gradient: linear-gradient(135deg, #22d3ee, #3b82f6 55%, #8b5cf6);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', -apple-system, Roboto, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh;
    background-image:
      linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
    background-size: 44px 44px;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 40px 20px 80px; }
  header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .logo { width: 40px; height: 40px; border-radius: 12px; background: var(--gradient);
          display: grid; place-items: center; font-size: 20px; }
  h1 { font-size: 1.5rem; letter-spacing: -.3px; }
  .sub { color: var(--muted); margin-bottom: 6px; font-size: .95rem; }
  .status { color: var(--muted); font-size: .8rem; margin-bottom: 26px; }
  .status b { color: var(--green); }
  .search { display: flex; gap: 10px; }
  input[type=text] {
    flex: 1; font-size: 1rem; color: var(--text); background: rgba(255,255,255,.05);
    border: 1px solid var(--border); border-radius: 14px; padding: 15px 18px; outline: none;
  }
  input[type=text]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(56,189,248,.15); }
  button {
    font-size: .95rem; font-weight: 600; color: #fff; background: var(--gradient);
    border: none; border-radius: 14px; padding: 0 26px; cursor: pointer;
  }
  button:disabled { opacity: .55; cursor: wait; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 30px; }
  .chip {
    font-size: .8rem; color: var(--muted); background: rgba(255,255,255,.04);
    border: 1px solid var(--border); border-radius: 999px; padding: 7px 14px; cursor: pointer;
  }
  .chip:hover { color: var(--text); border-color: rgba(56,189,248,.5); }
  .opts { display: flex; align-items: center; gap: 8px; margin: -18px 0 24px; font-size: .82rem; color: var(--muted); }
  select { color: var(--text); background: rgba(255,255,255,.05); border: 1px solid var(--border);
           border-radius: 8px; padding: 5px 8px; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 22px 24px; margin-bottom: 14px; display: none; white-space: pre-wrap;
  }
  .card.visible { display: block; }
  .card h3 { font-size: .85rem; text-transform: uppercase; letter-spacing: .08em;
             color: var(--accent); margin-bottom: 10px; }
  .src { border-left: 3px solid; border-image: var(--gradient) 1; padding: 4px 0 4px 16px; margin-bottom: 6px; }
  .src-meta { font-size: .8rem; color: var(--muted); margin-bottom: 4px; }
  .src-meta b { color: var(--text); }
  .src-meta .score { float: right; }
  mark { background: rgba(56,189,248,.25); color: #bfe9ff; border-radius: 3px; padding: 0 2px; }
  .err { color: #f87171; }
  footer { margin-top: 40px; color: var(--muted); font-size: .78rem; text-align: center; }
  footer a { color: var(--accent); text-decoration: none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🔎</div>
    <div>
      <h1>LogDoc AI</h1>
      <div class="sub">вопрос-ответ по документам — с цитатами, без галлюцинаций</div>
    </div>
  </header>
  <div class="status" id="status">загрузка индекса…</div>

  <div class="search">
    <input type="text" id="query" placeholder="Например: кто подписал акт приёмки по заявке 152?"
           autocomplete="off">
    <button id="ask">Спросить</button>
  </div>
  <div class="chips">
    <span class="chip">кто подписал акт приёмки по заявке 152?</span>
    <span class="chip">какая пеня за просрочку доставки?</span>
    <span class="chip">температурный режим для скоропортящихся грузов</span>
  </div>
  <div class="opts">
    режим ответа:
    <select id="backend">
      <option value="extractive">extractive — цитаты, оффлайн</option>
      <option value="gemini">gemini — генерация (нужен GEMINI_API_KEY)</option>
    </select>
  </div>

  <div class="card" id="answer"><h3>Ответ</h3><div id="answer-text"></div></div>
  <div class="card" id="sources"><h3>Источники</h3><div id="sources-list"></div></div>
  <div class="card" id="error"><h3>Ошибка</h3><div class="err" id="error-text"></div></div>

  <footer>LogDoc AI · оффлайн RAG по документам · <a href="https://github.com/nikitaserbakov351-dotcom/logdoc-ai">репозиторий</a></footer>
</div>

<script>
const $ = (id) => document.getElementById(id);

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function highlight(text, query) {
  let safe = escapeHtml(text);
  const terms = [...new Set(query.toLowerCase().split(/\\s+/).filter(t => t.length > 2))]
    .sort((a, b) => b.length - a.length)
    .map(t => t.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'));
  if (!terms.length) return safe;
  return safe.replace(new RegExp('(' + terms.join('|') + ')', 'gi'), '<mark>$1</mark>');
}

function showCards(ids) {
  document.querySelectorAll('.card').forEach(c => c.classList.remove('visible'));
  ids.forEach(id => $(id).classList.add('visible'));
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    $('status').innerHTML = `индекс: <b>${s.chunks}</b> фрагментов · эмбеддер: ${s.embedder} · ${s.index}`;
  } catch {
    $('status').innerHTML = '<span class="err">индекс не загружен — выполните: python -m logdoc.cli ingest ./samples</span>';
  }
}

async function ask(query) {
  $('ask').disabled = true;
  showCards(['answer']);
  $('answer-text').textContent = 'Ищу в документах…';
  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, k: 5, backend: $('backend').value})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'ошибка запроса');

    $('answer-text').innerHTML = highlight(data.text, query);
    $('sources-list').innerHTML = data.sources.map(s => `
      <div class="src">
        <div class="src-meta"><span class="score">score ${s.score}</span>
          [${s.n}] <b>${escapeHtml(s.file)}</b>, стр. ${s.page}</div>
        <div>${highlight(s.snippet, query)}</div>
      </div>`).join('');
    showCards(['answer', 'sources']);
  } catch (e) {
    showCards(['error']);
    $('error-text').textContent = e.message;
  } finally {
    $('ask').disabled = false;
  }
}

$('ask').onclick = () => { const q = $('query').value.trim(); if (q) ask(q); };
$('query').addEventListener('keydown', e => { if (e.key === 'Enter') $('ask').onclick(); });
document.querySelectorAll('.chip').forEach(c => c.onclick = () => { $('query').value = c.textContent; ask(c.textContent); });

loadStatus();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Веб-интерфейс LogDoc AI")
    parser.add_argument("--index", default=os.getenv("LOGDOC_INDEX", "logdoc_index.json"))
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    os.environ["LOGDOC_INDEX"] = args.index
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
