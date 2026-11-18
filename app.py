"""Flask app: chat UI + JSON API around the Robot Advisor graph."""
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

from advisor.catalog import Catalog, price_text
from advisor.graph import GroqLLM, build_graph

load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
catalog = Catalog()
sessions: dict[str, dict] = {}   # in-memory: {session_id: {"history": [...], "lead": {...}}}
_graph = None

# Public demo protection: each visitor IP gets at most RATE_LIMIT messages per hour.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", 40))
_hits: dict[str, list[float]] = {}


def rate_limited(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _hits.get(ip, []) if now - t < 3600]
    if len(recent) >= RATE_LIMIT:
        _hits[ip] = recent
        return True
    _hits[ip] = recent + [now]
    return False


def get_graph():
    global _graph
    if _graph is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None
        llm = GroqLLM(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), api_key=key)
        _graph = build_graph(llm, catalog)
    return _graph


FEATURED = ["bellabot-pro", "agibot-x2-ultra", "agibot-d1-edu-pro", "agibot-c5"]


@app.get("/")
def index():
    return render_template("index.html", featured=[catalog.by_id[i] for i in FEATURED])


@app.get("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon", max_age=86400)


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    msg = (body.get("message") or "").strip()[:2000]
    sid = body.get("session_id") or str(uuid.uuid4())
    if not msg:
        return jsonify(error="Empty message"), 400
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "?").split(",")[0].strip()
    if rate_limited(ip):
        return jsonify(error="Demo limit reached for this hour. Please try again later."), 429
    graph = get_graph()
    if graph is None:
        return jsonify(error="GROQ_API_KEY is not set. Add it to the .env file and restart."), 503

    sess = sessions.setdefault(sid, {"history": [], "lead": {}})
    try:
        out = graph.invoke({"user_msg": msg, "history": sess["history"], "lead": sess["lead"]})
    except Exception as e:  # LLM/network failure: tell the user plainly instead of a 500 page
        return jsonify(error=f"The AI service failed: {type(e).__name__}. Please try again."), 502

    sess["history"] += [{"role": "user", "text": msg}, {"role": "assistant", "text": out["reply"]}]
    sess["lead"] = out["lead"]
    cards = [{"name": r["name"], "category": r["category"], "price": price_text(r), "url": r["url"],
              "buy_usd": r["buy_usd"], "lease_usd_month": r["lease_usd_month"], "image": r.get("image")}
             for r in (catalog.all_by_id[i] for i in out.get("recommended_ids", []) if i in catalog.all_by_id)]
    return jsonify(session_id=sid, reply=out["reply"], status=out["status"], cards=cards,
                   trace=out["trace"], lead={k: v for k, v in out["lead"].items() if k != "off_topic"})


@app.post("/api/reset")
def reset():
    sessions.pop((request.get_json(force=True) or {}).get("session_id"), None)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5055)), debug=False)
