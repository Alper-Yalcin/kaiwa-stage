"""
Tek bir karakterle Japonca metni seslendirir ve MP3 üretir (Aşama 5).

Akış:
    metin -> dil ön-yüzü -> karakter preset -> SBV2 inference (WAV)
          -> ffmpeg ile MP3 + loudness normalize -> log

Kullanım:
    python scripts/generate_voice.py --text "こんにちは。" --character chill_male --out hello.mp3
    python scripts/generate_voice.py --list            # karakterleri ve model bilgisini yaz
    python scripts/generate_voice.py --text "..." --character calm_female --out x.mp3         --model-dir third_party/Style-Bert-VITS2/model_assets/jvs_multi12   # belirli model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.characters import load_character, list_characters  # noqa: E402
from japonca_tts.synth import synthesize_wav, model_info   # noqa: E402
from japonca_tts.export import wav_to_mp3                   # noqa: E402
from japonca_tts.config import OUTPUTS                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--character")
    ap.add_argument("--out", help="Çıktı MP3 dosya adı")
    ap.add_argument("--model-dir", default=None,
                    help="Model klasörü (varsayılan: models/finetuned > pretrained)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--list", action="store_true", help="Karakterleri + modeli listele")
    args = ap.parse_args()

    if args.list:
        info = model_info(args.model_dir, args.device)
        print(f"[model] {info['model_dir']}  ckpt={info['checkpoint']}")
        print(f"[model] konusmacilar: {info['speakers']}")
        print(f"[model] stiller     : {info['styles']}")
        for n in list_characters():
            c = load_character(n)
            ok = "" if c.speaker_id in info["speakers"] or c.speaker_id.isdigit() else "  (!) modelde yok"
            print(f"  {n:18} {c.speaker_id:7} {c.style_id:9} {c.description}{ok}")
        return
    if not (args.text and args.character and args.out):
        ap.error("--text, --character ve --out gerekli (ya da --list)")

    char = load_character(args.character)
    wav_path = OUTPUTS["wav"] / (Path(args.out).stem + ".wav")
    mp3_path = OUTPUTS["mp3"] / args.out

    synthesize_wav(text=args.text, character=char, out_wav=wav_path,
                   model_dir=args.model_dir, device=args.device)
    wav_to_mp3(wav_path, mp3_path)
    print(f"[ok] {mp3_path}")


if __name__ == "__main__":
    main()
