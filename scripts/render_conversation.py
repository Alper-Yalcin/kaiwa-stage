"""
web/dialog.json'daki sohbeti iki karakterle tek bir kayıt (MP3) olarak üretir.

Her tur: kullanıcı cümlesi (--me sesi) -> kısa duraklama -> cevap (--them sesi)
-> tur arası duraklama. dialog.json'daki cümle bazlı "style" (duygu) alanı
modelde varsa uygulanır; yoksa karakterin kendi stili kullanılır.

Kullanım:
    python scripts/render_conversation.py --me bright_female --them deep_narrator --out conversation_v2.mp3
    python scripts/render_conversation.py --me bright_female --them deep_narrator --model-dir models/finetuned/jvs_jvnv16
    python scripts/render_conversation.py ... --turns t01,t02,t03      # alt küme
    python scripts/render_conversation.py ... --no-style --no-rise     # v1 davranışı (duygu yok, soru yükseltme yok)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.characters import load_character   # noqa: E402
from japonca_tts.synth import synthesize_wav         # noqa: E402
from japonca_tts.export import wav_to_mp3            # noqa: E402
from japonca_tts.config import OUTPUTS               # noqa: E402

DIALOG = ROOT / "web" / "dialog.json"
GAP_REPLY = 0.55    # soru -> cevap arası (s)
GAP_TURN = 1.0      # tur arası (s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", required=True)
    ap.add_argument("--them", required=True)
    ap.add_argument("--out", default="conversation.mp3")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--turns", default=None, help="Virgülle ayrılmış tur id'leri")
    ap.add_argument("--no-style", action="store_true")
    ap.add_argument("--no-rise", action="store_true", help="Soru son-mora yükseltmesini kapat (v1 davranışı)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    turns = json.loads(DIALOG.read_text(encoding="utf-8"))["turns"]
    if args.turns:
        keep = set(args.turns.split(","))
        turns = [t for t in turns if t["id"] in keep]
    chars = {"me": load_character(args.me), "them": load_character(args.them)}
    stem = Path(args.out).stem
    wav_dir = OUTPUTS["wav"] / "conversation" / stem
    wav_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[np.ndarray] = []
    sr = 44100
    for t in turns:
        for side, key, gap in (("me", "user", GAP_REPLY), ("them", "reply", GAP_TURN)):
            u = t[key]
            style = None if args.no_style else u.get("style")
            weight = None if args.no_style else u.get("style_weight")
            wav = synthesize_wav(u["ja"], chars[side], wav_dir / f"{t['id']}_{side}.wav",
                                 model_dir=args.model_dir, device=args.device,
                                 style=style, style_weight=weight,
                                 question_rise=False if args.no_rise else None)
            y, sr = sf.read(str(wav), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            pieces.append(y)
            pieces.append(np.zeros(int(sr * gap), dtype=np.float32))
            print(f"[{t['id']}] {side:4} {style or chars[side].style_id:9} {u['ja']}")

    full = np.concatenate(pieces[:-1])   # sondaki boşluğu at
    full_wav = wav_dir / f"{stem}.wav"
    sf.write(str(full_wav), full, sr)
    mp3 = wav_to_mp3(full_wav, OUTPUTS["mp3"] / args.out)
    print(f"[ok] {mp3}  ({len(full) / sr:.1f} s, {len(turns)} tur)")


if __name__ == "__main__":
    main()
