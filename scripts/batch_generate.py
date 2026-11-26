"""
Toplu üretim (MVP 5): N cümle × M karakter -> WAV + MP3 + üretim manifest'i.

Girdi cümle kaynağı (--texts):
    * .json : {"sentences": [{"id": "...", "text": "..."}, ...]}   (tests/test_sentences.json)
    * .csv  : id,text sütunları
    * .txt  : satır başına bir cümle (id = t001, t002, ...)

Çıktı:
    outputs/wav/<run>/<character>/<id>.wav
    outputs/mp3/<run>/<character>/<id>.mp3
    outputs/logs/<run>_manifest.csv   -> qc_batch.py'nin girdisi
    outputs/logs/<run>_run.json       -> model sürümü + karakter parametreleri

Başarısız üretimler manifest'te status=FAIL ile işaretlenir, script devam eder.
QC AYRI süreçte çalışır (bkz. run_pipeline.py / qc_batch.py).

Kullanım:
    python scripts/batch_generate.py --texts tests/test_sentences.json \
        --characters chill_male excited_female --run mvp4
    python scripts/batch_generate.py --texts cumleler.txt --characters all --run demo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Windows konsolu (cp1254) Japonca basamıyor; UTF-8'e zorla.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.characters import load_character, list_characters   # noqa: E402
from japonca_tts.config import OUTPUTS                              # noqa: E402
from japonca_tts.export import wav_to_mp3                            # noqa: E402
from japonca_tts.qc import wav_stats, file_stats                     # noqa: E402
from japonca_tts.synth import synthesize_wav, model_info             # noqa: E402

MANIFEST_FIELDS = ["run", "character", "id", "text", "wav_path", "mp3_path",
                   "status", "error", "duration_sec", "wav_bytes", "mp3_bytes",
                   "synth_sec"]


def load_texts(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["sentences"] if isinstance(data, dict) else data
        return [{"id": r["id"], "text": r["text"]} for r in rows]
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            return [{"id": r["id"], "text": r["text"]} for r in csv.DictReader(f)]
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [{"id": f"t{i+1:03d}", "text": t} for i, t in enumerate(lines)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="json/csv/txt cümle listesi")
    ap.add_argument("--characters", nargs="+", required=True,
                    help="Karakter adları ya da 'all'")
    ap.add_argument("--run", default=None, help="Koşu adı (varsayılan: zaman damgası)")
    ap.add_argument("--model-dir", default=None,
                    help="Model klasörü (varsayılan: models/finetuned > pretrained)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-mp3", action="store_true", help="Sadece WAV üret")
    ap.add_argument("--no-reading-check", action="store_true",
                    help="Ön-uçuş okunuş denetimini (OJT vs Sudachi) atla")
    args = ap.parse_args()

    run = args.run or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    texts = load_texts(Path(args.texts))
    chars = list_characters() if args.characters == ["all"] else args.characters
    characters = [load_character(c) for c in chars]

    info = model_info(args.model_dir, args.device)
    print(f"[model] {info['model_dir']}  ckpt={info['checkpoint']}")
    print(f"[model] konusmacilar={len(info['speakers'])} stiller={info['styles']}")
    print(f"[run] {run}: {len(texts)} cumle x {len(characters)} karakter")

    OUTPUTS["logs"].mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUTS["logs"] / f"{run}_manifest.csv"

    # Ön-uçuş: okunuş anlaşmazlıkları (Open JTalk sözlük hataları için uyarı)
    if not args.no_reading_check:
        sys.path.insert(0, str(ROOT / "scripts"))
        from check_reading import analyze
        warn_lines = []
        for t in texts:
            for x in analyze(t["text"]):
                warn_lines.append(f"{t['id']}	{x['word']}	OJT={x['ojt']}	Sudachi={x['sudachi']}	{t['text']}")
        wpath = OUTPUTS["logs"] / f"{run}_reading_warnings.tsv"
        header = "id\tkelime\tOJT\tSudachi\tmetin\n"
        wpath.write_text(header + "".join(l + "\n" for l in warn_lines), encoding="utf-8")
        if warn_lines:
            print(f"[okunus] {len(warn_lines)} anlasmazlik (kontrol et, gerekirse configs/lexicon.yaml): {wpath}")
            for l in warn_lines[:10]:
                print("   ", l.replace("	", "  "))
        else:
            print("[okunus] anlasmazlik yok")
    rows: list[dict] = []
    t_run = time.time()

    for ch in characters:
        for t in texts:
            wav = OUTPUTS["wav"] / run / ch.name / f"{t['id']}.wav"
            mp3 = OUTPUTS["mp3"] / run / ch.name / f"{t['id']}.mp3"
            row = {"run": run, "character": ch.name, "id": t["id"], "text": t["text"],
                   "wav_path": str(wav), "mp3_path": "" if args.no_mp3 else str(mp3),
                   "status": "OK", "error": "", "duration_sec": 0.0,
                   "wav_bytes": 0, "mp3_bytes": 0, "synth_sec": 0.0}
            t0 = time.time()
            try:
                synthesize_wav(text=t["text"], character=ch, out_wav=wav,
                               model_dir=args.model_dir, device=args.device)
                ws = wav_stats(wav)
                row["duration_sec"], row["wav_bytes"] = ws["duration_sec"], ws["size_bytes"]
                if not ws["ok"]:
                    raise RuntimeError(f"WAV bozuk: {ws['error']}")
                if not args.no_mp3:
                    wav_to_mp3(wav, mp3)
                    ms = file_stats(str(mp3))
                    row["mp3_bytes"] = ms["size_bytes"]
                    if not ms["ok"]:
                        raise RuntimeError("MP3 0 bayt / yazılamadı")
            except Exception as e:  # noqa: BLE001
                row["status"], row["error"] = "FAIL", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            row["synth_sec"] = round(time.time() - t0, 2)
            rows.append(row)
            print(f"[{row['status']:4}] {ch.name:18} {t['id']:6} {row['duration_sec']:5.2f}s"
                  f"  ({row['synth_sec']}s)  {row['error']}")

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    run_meta = {
        "run": run,
        "started": datetime.fromtimestamp(t_run).isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t_run, 1),
        "model": info,
        "device": args.device,
        "texts_source": str(args.texts),
        "n_texts": len(texts),
        "characters": {c.name: asdict(c) for c in characters},
        "n_ok": sum(r["status"] == "OK" for r in rows),
        "n_fail": sum(r["status"] == "FAIL" for r in rows),
        "manifest": str(manifest_path),
    }
    run_json = OUTPUTS["logs"] / f"{run}_run.json"
    run_json.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[ok] {run_meta['n_ok']}/{len(rows)} uretildi, {run_meta['n_fail']} basarisiz")
    print(f"[manifest] {manifest_path}")
    print(f"[run-meta] {run_json}")
    print(f"QC icin: python scripts/qc_batch.py {manifest_path}")
    sys.exit(0 if run_meta["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
