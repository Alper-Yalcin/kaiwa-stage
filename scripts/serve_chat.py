"""
Canlı sohbet sunucusu: web/ sayfası + gerçek model (TTS) + Ollama (cevap).

    python scripts/serve_chat.py                # http://localhost:8000
    python scripts/serve_chat.py --port 8080 --llm qwen3:4b --device cuda

Uç noktalar:
    GET  /api/health          {ok, llm, characters}
    POST /api/chat            {text, history, me, them} -> {user:{ja,tr,emotion,audio}, reply:{...}}
    POST /api/tts             {text, character, style?, style_weight?} -> audio/mpeg
    GET  /chat-audio/<dosya>  üretilen mp3'ler (outputs/chat/)
    GET  /                    web/ (statik)

Sayfa sunucu yoksa (örn. statik hosting) otomatik olarak hazır-cümle moduna düşer.
TTS modeli süreç başında bir kez yüklenir; sentez tek kilit altında çalışır (model
iş parçacığı güvenli değil). Ollama modeli 30 dk bellekte tutulur (keep_alive).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException                       # noqa: E402
from fastapi.responses import FileResponse, JSONResponse         # noqa: E402
from fastapi.staticfiles import StaticFiles                      # noqa: E402
from pydantic import BaseModel                                   # noqa: E402
from starlette.concurrency import run_in_threadpool             # noqa: E402

from japonca_tts import chat as chatmod                          # noqa: E402
from japonca_tts.characters import load_character, list_characters   # noqa: E402
from japonca_tts.synth import synthesize_wav, get_model         # noqa: E402
from japonca_tts.export import wav_to_mp3                        # noqa: E402

CHAT_DIR = ROOT / "outputs" / "chat"
WEB_DIR = ROOT / "web"
BOT_NAMES = {"yuki": "ユキ", "hana": "ハナ", "kenta": "ケンタ", "takeshi": "タケシ"}
EMO_WEIGHT = 0.7

app = FastAPI(title="Kaiwa Stage")
_tts_lock = threading.Lock()
_cfg = {"device": "cuda", "llm": chatmod.DEFAULT_MODEL}


class ChatReq(BaseModel):
    text: str
    history: list[dict] = []
    me: str = "yuki"
    them: str = "takeshi"


class TTSReq(BaseModel):
    text: str
    character: str = "yuki"
    style: str | None = None
    style_weight: float | None = None


def _tts(text: str, character: str, emotion: str | None, tag: str) -> str:
    """Sentez + mp3; /chat-audio altındaki yolu döndürür."""
    style = None if emotion in (None, "neutral") else emotion
    weight = None if style is None else EMO_WEIGHT
    name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{tag}"
    wav = CHAT_DIR / f"{name}.wav"
    with _tts_lock:
        synthesize_wav(text, load_character(character), wav, device=_cfg["device"],
                       style=style, style_weight=weight)
    mp3 = wav_to_mp3(wav, CHAT_DIR / f"{name}.mp3")
    wav.unlink(missing_ok=True)
    return f"/chat-audio/{mp3.name}"


@app.get("/api/health")
def health():
    ok, msg = chatmod.ollama_available(_cfg["llm"])
    return {"ok": True, "llm": _cfg["llm"] if ok else None, "llm_error": None if ok else msg,
            "characters": [c for c in list_characters() if c in BOT_NAMES]}


@app.post("/api/chat")
async def chat(req: ChatReq):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "boş mesaj")
    if req.me not in BOT_NAMES or req.them not in BOT_NAMES:
        raise HTTPException(400, "bilinmeyen ses")
    try:
        data = await run_in_threadpool(chatmod.chat_turn, text, req.history,
                                       BOT_NAMES[req.them], _cfg["llm"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"LLM hatası: {e}") from e
    u_audio = await run_in_threadpool(_tts, data["user_ja"], req.me, data["user_emotion"], "u")
    r_audio = await run_in_threadpool(_tts, data["reply_ja"], req.them, data["reply_emotion"], "r")
    return {
        "user": {"ja": data["user_ja"], "romaji": data["user_romaji"], "tr": data["user_tr"],
                 "emotion": data["user_emotion"], "audio": u_audio},
        "reply": {"ja": data["reply_ja"], "romaji": data["reply_romaji"], "tr": data["reply_tr"],
                  "emotion": data["reply_emotion"], "audio": r_audio},
        "suggestions": data["suggestions"],
    }


@app.post("/api/tts")
async def tts(req: TTSReq):
    if req.character not in list_characters():
        raise HTTPException(400, "bilinmeyen karakter")
    url = await run_in_threadpool(_tts, req.text, req.character, req.style, "t")
    return FileResponse(CHAT_DIR / Path(url).name, media_type="audio/mpeg")


@app.get("/chat-audio/{name}")
def chat_audio(name: str):
    p = CHAT_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/mpeg")


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


@app.exception_handler(Exception)
async def _err(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm", default=chatmod.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    _cfg.update(device=args.device, llm=args.llm)
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHAT_DIR.glob("*"):          # önceki oturumun klipleri
        old.unlink(missing_ok=True)

    print("[tts] model yükleniyor...")
    get_model(None, args.device)
    ok, msg = chatmod.ollama_available(args.llm)
    print(f"[llm] {'OK: ' + msg if ok else 'UYARI: ' + msg}")
    print(f"[web] http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
