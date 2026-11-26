"""Kalite kontrol yardımcıları (Aşama 6 / MVP 5).

CLI girişleri scripts/qc_whisper.py ve scripts/qc_batch.py içindedir. Bu
modül torch IMPORT ETMEZ (Whisper/ctranslate2 ile torch-CUDA aynı süreçte
cuDNN çakışması yapıyor); yalnızca pyopenjtalk + saf Python.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

import re

from .text_frontend import to_phonemes, normalize

_KANA_PUNCT = re.compile(r"[。、，．,.!！?？…・「」『』（）()\s'’\-−ー]+$|[。、，．,.!！?？…・「」『』（）()\s'’]")


def cer(ref: str, hyp: str) -> float:
    """Karakter hata oranı: Levenshtein(ref, hyp) / len(ref)."""
    r, h = list(ref), list(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(h)] / max(1, len(r))


def reading_cer(reference_text: str, hypothesis_text: str) -> float:
    """İki metni katakana okunuşa indirgeyip CER döndürür.

    Yüzey metni doğrudan karşılaştırılmaz: Whisper'ın kanji seçimi farklı
    olabilir (ör. 「有難う」/「ありがとう」), okunuş aynıysa hata sayılmaz.
    Noktalama/boşluk her iki taraftan atılır (Whisper'ın eklediği 。 tek
    kelimede %20 CER yapıyordu; sesle ilgisi yok).
    """
    ref = _KANA_PUNCT.sub("", to_phonemes(normalize(reference_text), kana=True))
    hyp = _KANA_PUNCT.sub("", to_phonemes(normalize(hypothesis_text), kana=True))
    return cer(ref, hyp)


def wav_stats(path: Path) -> dict:
    """WAV bütünlüğü: var mı, boyut, süre. Bozuk/0-bayt ise ok=False."""
    p = Path(path)
    info = {"exists": p.exists(), "size_bytes": 0, "duration_sec": 0.0, "ok": False,
            "error": ""}
    if not p.exists():
        info["error"] = "dosya yok"
        return info
    info["size_bytes"] = p.stat().st_size
    if info["size_bytes"] == 0:
        info["error"] = "0 bayt"
        return info
    try:
        with wave.open(str(p), "rb") as w:
            frames, sr = w.getnframes(), w.getframerate()
        info["duration_sec"] = round(frames / sr, 3) if sr else 0.0
        if info["duration_sec"] < 0.2:
            info["error"] = "çok kısa (<0.2s)"
            return info
        info["ok"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = f"okunamadı: {e}"
    return info


def file_stats(path: Optional[str]) -> dict:
    """Herhangi bir dosya (MP3) için var/boyut kontrolü."""
    if not path:
        return {"exists": False, "size_bytes": 0, "ok": False}
    p = Path(path)
    size = p.stat().st_size if p.exists() else 0
    return {"exists": p.exists(), "size_bytes": size, "ok": size > 0}
