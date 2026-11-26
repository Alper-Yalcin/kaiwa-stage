"""
Sohbet botu: yerel Ollama (qwen3) ile Japonca konuşma partneri.

Kullanıcı ne yazarsa yazsın (romaji, Türkçe, İngilizce, Japonca) model şunları
JSON olarak döndürür:
    user_ja      kullanıcının mesajı doğal Japonca yazımıyla (romaji -> kana/kanji; başka dil -> çeviri)
    user_tr      kullanıcı mesajının Türkçesi
    user_emotion neutral | happy | sad | surprise | anger | fear | disgust
    reply_ja     botun kısa Japonca cevabı (1-2 cümle; TTS için)
    reply_tr     cevabın Türkçesi
    reply_emotion (aynı küme)

Ollama'nın yapılandırılmış çıktı (JSON schema) özelliği kullanılır; qwen3'ün
düşünme modu kapatılır (think=false), aksi halde cevap gecikir ve içerik kirlenir.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("CHAT_MODEL", "qwen3:8b")
EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger", "fear", "disgust"]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_ja": {"type": "string"},
        "user_tr": {"type": "string"},
        "user_emotion": {"type": "string", "enum": EMOTIONS},
        "reply_ja": {"type": "string"},
        "reply_tr": {"type": "string"},
        "reply_emotion": {"type": "string", "enum": EMOTIONS},
        "suggestions": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "object",
                      "properties": {"ja": {"type": "string"}, "tr": {"type": "string"}},
                      "required": ["ja", "tr"]},
        },
    },
    "required": ["user_ja", "user_tr", "user_emotion", "reply_ja", "reply_tr", "reply_emotion", "suggestions"],
}

SYSTEM_PROMPT = """あなたは「{bot_name}」という名前の、親しみやすい日本語の会話パートナーです。
相手は日本語を勉強しているトルコ人です。相手はローマ字・トルコ語・英語・日本語のどれで書くか分かりません。

必ず次のJSONだけを返してください（説明や前置きは禁止）:
- user_ja: 相手のメッセージを自然な日本語の表記に直したもの（ローマ字なら漢字かな交じりに、他の言語なら日本語に訳す）。
- user_tr: user_ja のトルコ語訳。
- user_emotion: 相手の感情。neutral / happy / sad / surprise / anger / fear / disgust のいずれか。
- reply_ja: あなたの返事。自然で短い日本語（1〜2文、最大60文字）。質問には「〜ですか？」のように必ず「？」を付ける。会話を続ける。
- reply_tr: reply_ja のトルコ語訳。
- reply_emotion: あなたの返事の感情（同じ選択肢）。
- suggestions: 相手が次に言えそうな返事を3つ（相手は日本語初心者。それぞれ短い自然な日本語 ja とトルコ語訳 tr）。あなたの返事に自然につながるもの。

丁寧語（です・ます）で話してください。難しい単語は避け、初級〜中級の日本語にしてください。"""


def _messages(user_text: str, history: list[dict[str, str]], bot_name: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(bot_name=bot_name)}]
    for h in history[-8:]:                      # son 8 tur bağlam
        if h.get("user_ja"):
            msgs.append({"role": "user", "content": h["user_ja"]})
        if h.get("reply_ja"):
            msgs.append({"role": "assistant", "content": h["reply_ja"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def chat_turn(user_text: str, history: list[dict[str, str]] | None = None,
              bot_name: str = "ユキ", model: str | None = None, timeout: float = 120.0) -> dict[str, str]:
    """Bir sohbet turu: kullanıcı metni -> JSON (user_ja, user_tr, ..., reply_ja, ...)."""
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": _messages(user_text, history or [], bot_name),
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "keep_alive": "30m",
        "options": {"temperature": 0.7, "num_predict": 300},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
    content = r.json()["message"]["content"]
    data = json.loads(content)
    for k in ("user_emotion", "reply_emotion"):
        if data.get(k) not in EMOTIONS:
            data[k] = "neutral"
    for k in ("user_ja", "reply_ja"):
        data[k] = (data.get(k) or "").strip()
    if not data["reply_ja"]:
        data["reply_ja"] = "すみません、もう一度言ってください。"
        data["reply_tr"] = "Pardon, bir daha söyler misiniz?"
    if not data["user_ja"]:
        data["user_ja"] = user_text
    sugg = [x for x in (data.get("suggestions") or []) if isinstance(x, dict) and x.get("ja")]
    data["suggestions"] = [{"ja": x["ja"].strip(), "tr": (x.get("tr") or "").strip(),
                            "romaji": romaji(x["ja"])} for x in sugg[:3]]
    data["user_romaji"] = romaji(data["user_ja"])
    data["reply_romaji"] = romaji(data["reply_ja"])
    return data


_kakasi = None
_PARTICLES = {"は": "wa", "へ": "e", "を": "o"}
_ATTACH_JOSHI = {"て", "で", "ば", "たり", "ながら", "ちゃ", "じゃ"}   # önceki fiile yapışan edatlar (te-form vb.)
_PUNCT = {"、": ",", "。": ".", "！": "!", "？": "?", "，": ",", "．": "."}
_VOWEL_ROW = {  # uzun ünlü işareti ー için: önceki kananın ünlüsü -> ders-kitabı romaji (ō -> ou, ē -> ei)
    "a": "ア", "i": "イ", "u": "ウ", "e": "イ", "o": "ウ"}


def _kata_to_hepburn(kata: str) -> str:
    """Katakana okunuş -> romaji. Aksan işaretleri atılır, ー önceki ünlüye göre açılır."""
    global _kakasi
    import pykakasi
    if _kakasi is None:
        _kakasi = pykakasi.kakasi()
    kata = kata.replace("’", "").replace("'", "")
    out = []
    for ch in kata:
        if ch == "ー" and out:
            prev = "".join(x["hepburn"] for x in _kakasi.convert(out[-1]))
            out.append(_VOWEL_ROW.get(prev[-1:] if prev else "u", "ウ"))
        else:
            out.append(ch)
    return "".join(x["hepburn"] for x in _kakasi.convert("".join(out)))


def romaji(text: str) -> str:
    """Japonca -> romaji okunuş satırı. Okuma Open JTalk'tan (bağlama göre doğru:
    今日は -> kyou wa, こんにちは -> konnichiwa), romanizasyon pykakasi ile.
    Kelime sınırlarında boşluk; çekim ekleri ve te-form edatları fiile yapışır;
    は/へ/を edatları wa/e/o."""
    if not text:
        return ""
    try:
        import pyopenjtalk
    except ImportError:
        return ""
    groups: list[list[str]] = []        # her grup: katakana parçaları (birleşik okunur)
    words: list[str] = []               # groups ile aynı uzunlukta; None = katakana grubu, str = hazır romaji
    prev_pos = ""
    for d in pyopenjtalk.run_frontend(text):
        surf, pron, pos = d["string"], d.get("pron", ""), d.get("pos", "")
        if surf in _PUNCT or pos.startswith("記号"):
            groups.append([]); words.append(_PUNCT.get(surf, surf)); prev_pos = pos; continue
        if pos.startswith("助詞") and surf in _PARTICLES:
            groups.append([]); words.append(_PARTICLES[surf]); prev_pos = pos; continue
        attach = groups and words[-1] is None and (
            (pos.startswith("助動詞") and prev_pos.startswith(("動詞", "助動詞")))
            or (pos.startswith("助詞") and surf in _ATTACH_JOSHI
                and prev_pos.startswith(("動詞", "形容詞", "助動詞"))))
        if attach:
            groups[-1].append(pron)
        else:
            groups.append([pron]); words.append(None)
        prev_pos = pos
    out = []
    for g, w in zip(groups, words):
        out.append(w if w is not None else _kata_to_hepburn("".join(g)))
    s = " ".join(x for x in out if x)
    for a, b in ((" ,", ","), (" .", "."), (" !", "!"), (" ?", "?")):
        s = s.replace(a, b)
    return " ".join(s.split())


def ollama_available(model: str | None = None) -> tuple[bool, str]:
    """(çalışıyor mu, mesaj)."""
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        want = model or DEFAULT_MODEL
        if want not in names:
            return False, f"Ollama çalışıyor ama '{want}' modeli yok (ollama pull {want}); mevcut: {names}"
        return True, want
    except Exception as e:  # noqa: BLE001
        return False, f"Ollama'ya ulaşılamadı ({OLLAMA_URL}): {e}"
