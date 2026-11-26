"""
Chat demosu için sesleri üretir ve web/manifest.js'i yazar.

web/dialog.json'daki her tur (kullanıcı cümlesi + cevap), dialog.json'daki
"characters" listesindeki her karakterle seslendirilir; sonuç web/audio/<karakter>/<tur>_{u,r}.mp3 olur. Site statiktir:
web/index.html + manifest.js + audio/ olduğu gibi bir web sunucusuna kopyalanır,
model hiç dışarı çıkmaz.

Kullanım:
    python scripts/build_web.py                         # eksik olanları üret
    python scripts/build_web.py --force                 # hepsini yeniden üret
    python scripts/build_web.py --characters bright_female,deep_narrator
    python scripts/build_web.py --manifest-only         # ses üretme, sadece manifest.js

Var olan MP3'ler atlanır; dialog.json'a tur eklemek ucuzdur.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.characters import load_character, list_characters  # noqa: E402
from japonca_tts.config import OUTPUTS                              # noqa: E402

WEB = ROOT / "web"
DIALOG = WEB / "dialog.json"
AUDIO = WEB / "audio"
MANIFEST = WEB / "manifest.js"
WAV_TMP = OUTPUTS["wav"] / "web"

SKIP_CHARACTERS: set[str] = set()


def rel_audio(char: str, turn_id: str, side: str) -> str:
    return f"audio/{char}/{turn_id}_{side}.mp3"


def write_manifest(turns: list[dict], chars: list[str]) -> None:
    from japonca_tts.chat import romaji
    for t in turns:
        for key in ("user", "reply"):
            t[key]["romaji"] = romaji(t[key]["ja"])
    data = {
        "characters": [
            {"name": c, "description": load_character(c).description} for c in chars
        ],
        "turns": turns,
        "audio_pattern": "audio/{character}/{turn}_{side}.mp3",
    }
    js = "// scripts/build_web.py üretti; elle düzenleme.\nwindow.CHAT_DATA = "
    js += json.dumps(data, ensure_ascii=False, indent=1) + ";\n"
    MANIFEST.write_text(js, encoding="utf-8")
    print(f"[ok] {MANIFEST.relative_to(ROOT)}  ({len(chars)} karakter, {len(turns)} tur)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--characters", default=None,
                    help="Virgülle ayrılmış karakter listesi (varsayılan: hepsi)")
    ap.add_argument("--force", action="store_true", help="Var olan MP3'leri yeniden üret")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    dialog = json.loads(DIALOG.read_text(encoding="utf-8"))
    turns = dialog["turns"]
    # Öncelik: --characters > dialog.json "characters" > tüm karakterler
    chars = ([c.strip() for c in args.characters.split(",")] if args.characters
             else dialog.get("characters")
             or [c for c in list_characters() if c not in SKIP_CHARACTERS])

    if args.manifest_only:
        write_manifest(turns, chars)
        return

    from japonca_tts.synth import synthesize_wav   # model importu geç: --manifest-only hızlı olsun
    from japonca_tts.export import wav_to_mp3

    jobs = []
    for c in chars:
        for t in turns:
            for side, key in (("u", "user"), ("r", "reply")):
                mp3 = WEB / rel_audio(c, t["id"], side)
                if mp3.exists() and not args.force:
                    continue
                jobs.append((c, t["id"], side, t[key], mp3))
    print(f"[plan] {len(jobs)} klip üretilecek "
          f"({len(chars)} karakter × {len(turns)} tur × 2; var olanlar atlandı)")

    t0 = time.time()
    fails = 0
    for i, (c, tid, side, utt, mp3) in enumerate(jobs, 1):
        text = utt["ja"]
        wav = WAV_TMP / c / f"{tid}_{side}.wav"
        try:
            synthesize_wav(text, load_character(c), wav,
                           model_dir=args.model_dir, device=args.device,
                           style=utt.get("style"), style_weight=utt.get("style_weight"))
            wav_to_mp3(wav, mp3)
            print(f"[{i}/{len(jobs)}] {c:18} {tid}_{side}  {text}")
        except Exception as e:                      # bir klip düşerse kalanı sürsün
            fails += 1
            print(f"[{i}/{len(jobs)}] {c:18} {tid}_{side}  HATA: {e}")

    write_manifest(turns, chars)
    print(f"[done] {len(jobs) - fails}/{len(jobs)} klip, {time.time() - t0:.0f} s"
          + (f", {fails} hata" if fails else ""))


if __name__ == "__main__":
    main()
