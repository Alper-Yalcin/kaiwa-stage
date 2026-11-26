"""
Üretilen sesin kalite kontrolü (Aşama 6).

Whisper ile transkript çıkarır, HEM transkripti HEM orijinal metni
pyopenjtalk ile OKUNUŞA (kana) indirger ve okunuş seviyesinde CER
hesaplar. Yüzey metni doğrudan karşılaştırılmaz (kanji çeşitliliği
yanlış-pozitif üretir).

Kullanım:
    python scripts/qc_whisper.py --wav outputs/wav/hello.wav --text "こんにちは。"
"""
from __future__ import annotations

import argparse


def to_kana(text: str) -> str:
    """Metni pyopenjtalk ile katakana okunuşa indirger."""
    import pyopenjtalk
    return pyopenjtalk.g2p(text, kana=True)


def cer(ref: str, hyp: str) -> float:
    """Karakter hata oranı (Levenshtein / ref uzunluğu)."""
    import numpy as np
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", required=True, help="Orijinal Japonca metin")
    ap.add_argument("--model-size", default="large-v3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="CPU önerilir: GPU'yu SBV2 tutuyorsa ctranslate2 cuDNN çakışabilir")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    compute = "int8" if args.device == "cpu" else "float16"
    model = WhisperModel(args.model_size, device=args.device, compute_type=compute)
    segments, _ = model.transcribe(args.wav, language="ja")
    hyp_text = "".join(s.text for s in segments).strip()

    ref_kana = to_kana(args.text)
    hyp_kana = to_kana(hyp_text)
    score = cer(ref_kana, hyp_kana)

    print(f"orijinal      : {args.text}")
    print(f"whisper       : {hyp_text}")
    print(f"ref (kana)    : {ref_kana}")
    print(f"hyp (kana)    : {hyp_kana}")
    print(f"CER (okunuş)  : {score:.3f}  ({'OK' if score < 0.1 else 'GÖZDEN GEÇİR'})")


if __name__ == "__main__":
    main()
