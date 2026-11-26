"""
Toplu QC (Aşama 6 / MVP 5): bir üretim manifest'indeki her satır için
  1) dosya bütünlüğü (WAV var mı, 0 bayt mı, süre; MP3 var mı)
  2) Whisper transkripti -> okunuş-seviyesi CER (eşik varsayılan 0.15)
hesaplar; satır bazlı rapor CSV'si + özet JSON yazar.

Manifest: batch_generate.py çıktısı (run, character, id, text, wav_path, mp3_path,
status, ...) ya da eski mvp1 formatı (id, wav_path, text).

torch IMPORT ETMEZ: ctranslate2 (Whisper) ile torch-CUDA aynı process'te
cuDNN çakışması yaptığı için QC üretimden ayrı süreçte çalışır. Whisper CPU'da.

KISA MADDELER (<= 4 mora: 猫, 手, はい): Whisper 0.3-0.5 s'lik tek klipleri
okuyamıyor, halüsinasyon üretiyor (ölçüldü: 手 -> "ご視聴ありがとうございました").
Sesi bağlam içinde verince (aynı karakterin kısa kelimeleri 0.6 s aralıkla tek
klipte) doğru tanıyor. Bu yüzden kısa maddeler tek tek FAIL sayılmaz:
qc_status=SHORT + tek-klip transkripti bilgi amaçlı; karakter başına
"concat_cer" (birleştirilmiş klip CER'i) özet JSON'a yazılır ve eşik 0.30'u
aşarsa karakter için uyarı basılır.

Çıktı:
    outputs/logs/<run>_qc.csv     satır bazlı: CER, transkript, durum
    outputs/logs/<run>_qc.json    özet: karakter başına ortalama CER, geçen/kalan

Kullanım:
    python scripts/qc_batch.py outputs/logs/mvp4_manifest.csv
    python scripts/qc_batch.py <manifest.csv> --threshold 0.15 --model large-v3
    python scripts/qc_batch.py <manifest.csv> --no-asr      # sadece dosya bütünlüğü
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Windows konsolu (cp1254) Japonca basamıyor; UTF-8'e zorla.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.qc import reading_cer, wav_stats, file_stats  # noqa: E402
from japonca_tts.text_frontend import to_phonemes, normalize      # noqa: E402

SHORT_MORA = 4
SHORT_CONCAT_THRESHOLD = 0.30


def mora_count(text: str) -> int:
    kana = to_phonemes(normalize(text), kana=True)
    return sum(1 for ch in kana if ch not in "ァィゥェォャュョ、。！？，．・ 　'")


def transcribe(asr, wav_path: str, prompt: str | None = None) -> str:
    segments, _ = asr.transcribe(wav_path, language="ja", initial_prompt=prompt,
                                 condition_on_previous_text=False)
    return "".join(s.text for s in segments).strip()


def concat_cer(asr, rows: list[dict], gap_sec: float = 0.6) -> float | None:
    """Kısa maddeleri tek klipte birleştirip bağlamlı CER hesaplar."""
    import tempfile
    import numpy as np
    import soundfile as sf
    parts, refs, sr = [], [], 44100
    for r in rows:
        a, sr = sf.read(r["wav_path"])
        parts += [np.asarray(a, dtype=np.float32), np.zeros(int(sr * gap_sec), np.float32)]
        refs.append(r["text"])
    if not parts:
        return None
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, np.concatenate(parts), sr)
    hyp = transcribe(asr, tmp, prompt="単語を一つずつ、読点で区切って読み上げます。例：花、海、月、雨。")
    Path(tmp).unlink(missing_ok=True)
    return round(reading_cer("、".join(refs), hyp), 4)

REPORT_FIELDS = ["run", "character", "id", "text", "transcript", "cer", "qc_status",
                 "reason", "duration_sec", "wav_bytes", "mp3_bytes", "wav_path", "mp3_path"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--no-asr", action="store_true", help="Whisper'ı atla")
    ap.add_argument("--short-mora", type=int, default=SHORT_MORA,
                    help="Bu mora sayısı ve altı 'kısa' sayılır (tek tek FAIL edilmez)")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "logs"))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    run = rows[0].get("run") if rows and rows[0].get("run") else manifest.stem.replace("_manifest", "")

    asr = None
    if not args.no_asr:
        from faster_whisper import WhisperModel
        asr = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"{'karakter':18} {'id':6} {'CER':>6}  {'durum':6}  metin")
    print("-" * 80)
    report: list[dict] = []
    for row in rows:
        rep = {"run": run, "character": row.get("character", "-"), "id": row["id"],
               "text": row["text"], "transcript": "", "cer": "", "qc_status": "OK",
               "reason": "", "wav_path": row["wav_path"], "mp3_path": row.get("mp3_path", "")}

        # 1) Üretim zaten başarısızsa
        if row.get("status", "OK") == "FAIL":
            rep.update(qc_status="FAIL", reason=f"uretim: {row.get('error', '')}",
                       duration_sec=0, wav_bytes=0, mp3_bytes=0)
            report.append(rep)
            continue

        # 2) Dosya bütünlüğü
        ws = wav_stats(Path(row["wav_path"]))
        ms = file_stats(row.get("mp3_path") or None)
        rep.update(duration_sec=ws["duration_sec"], wav_bytes=ws["size_bytes"],
                   mp3_bytes=ms["size_bytes"])
        if not ws["ok"]:
            rep.update(qc_status="FAIL", reason=f"wav: {ws['error']}")
        elif row.get("mp3_path") and not ms["ok"]:
            rep.update(qc_status="FAIL", reason="mp3 yok / 0 bayt")

        # 3) ASR + CER
        if rep["qc_status"] == "OK" and asr is not None:
            hyp = transcribe(asr, row["wav_path"])
            score = reading_cer(row["text"], hyp)
            rep.update(transcript=hyp, cer=round(score, 4))
            if mora_count(row["text"]) <= args.short_mora:
                rep.update(qc_status="SHORT", reason="kisa madde: tek-klip ASR guvenilmez, concat_cer'e bak")
            elif score >= args.threshold:
                rep.update(qc_status="FAIL", reason=f"CER {score:.3f} >= {args.threshold}")

        report.append(rep)
        cer_s = f"{rep['cer']:6.3f}" if rep["cer"] != "" else "   -  "
        print(f"{rep['character']:18} {rep['id']:6} {cer_s}  {rep['qc_status']:6}  "
              f"{rep['text'][:34]}  {rep['reason']}")

    # --- Rapor dosyaları ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run}_qc.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(report)

    per_char: dict[str, list[dict]] = defaultdict(list)
    for r in report:
        per_char[r["character"]].append(r)
    summary_chars = {}
    for c, rs in per_char.items():
        longs = [r for r in rs if r["qc_status"] != "SHORT"]
        shorts = [r for r in rs if r["qc_status"] == "SHORT"]
        cers = [r["cer"] for r in longs if r["cer"] != ""]
        ccer = concat_cer(asr, shorts) if (asr is not None and shorts) else None
        summary_chars[c] = {
            "n": len(rs),
            "n_ok": sum(r["qc_status"] == "OK" for r in rs),
            "n_short": len(shorts),
            "mean_cer": round(statistics.mean(cers), 4) if cers else None,
            "max_cer": round(max(cers), 4) if cers else None,
            "concat_cer": ccer,
            "failed_ids": [r["id"] for r in rs if r["qc_status"] == "FAIL"],
            "short_ids": [r["id"] for r in shorts],
        }
    all_cers = [r["cer"] for r in report if r["cer"] != "" and r["qc_status"] != "SHORT"]
    n_ok = sum(r["qc_status"] in ("OK", "SHORT") for r in report)
    summary = {
        "run": run, "manifest": str(manifest), "threshold": args.threshold,
        "asr_model": None if args.no_asr else args.model,
        "n": len(report), "n_ok": n_ok, "n_fail": len(report) - n_ok,
        "n_short": sum(r["qc_status"] == "SHORT" for r in report),
        "mean_cer": round(statistics.mean(all_cers), 4) if all_cers else None,
        "characters": summary_chars,
    }
    json_path = out_dir / f"{run}_qc.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 80)
    for c, s in summary_chars.items():
        mc = f"{s['mean_cer']:.3f}" if s["mean_cer"] is not None else "-"
        extra = ""
        if s["n_short"]:
            cc = s["concat_cer"]
            warn = "  (!) kisa kelimeler zayif" if cc is not None and cc >= SHORT_CONCAT_THRESHOLD else ""
            extra = f"  kisa={s['n_short']} concat_cer={cc if cc is not None else '-'}{warn}"
        print(f"{c:18} {s['n_ok']}/{s['n'] - s['n_short']} gecti  ort.CER={mc}  fail={s['failed_ids']}{extra}")
    mc = f"{summary['mean_cer']:.3f}" if summary["mean_cer"] is not None else "-"
    n_short = sum(r["qc_status"] == "SHORT" for r in report)
    print(f"\nSonuc: {n_ok - n_short}/{len(report) - n_short} gecti (CER<{args.threshold}), "
          f"ort.CER={mc}, kisa madde={n_short}")
    print(f"[rapor] {csv_path}\n[ozet ] {json_path}")
    sys.exit(0 if n_ok == len(report) else 1)


if __name__ == "__main__":
    main()
