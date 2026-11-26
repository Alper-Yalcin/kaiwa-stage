"""
JVS klasörünü tarar ve metadata/jvs_manifest.csv üretir.

Beklenen JVS yapısı:
    data/raw/jvs/.../jvs001/parallel100/transcripts_utf8.txt
    data/raw/jvs/.../jvs001/parallel100/wav24kHz16bit/*.wav
    (nonpara30, whisper10, falsetto10 aynı düzende)
    data/raw/jvs/.../gender_f0range.txt

transcripts_utf8.txt satır formatı:
    FILEID:transkript metni

Manifest sütunları:
    audio_path, text, speaker_id, style_id, gender, duration_sec

Kullanım:
    python scripts/build_manifest.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "jvs"
MANIFEST = ROOT / "metadata" / "jvs_manifest.csv"

# JVS gerçek klasör adları: falsetto klasörü "falset10" olarak gelir.
# style_id eşlemesi: normal okuma stilleri "normal"e indirgenir.
STYLE_DIRS = {
    "parallel100": "normal",
    "nonpara30": "normal",
    "whisper10": "whisper",
    "falset10": "falsetto",
}


def load_gender_map() -> dict[str, str]:
    """gender_f0range.txt -> {speaker_id: gender}. Bulunamazsa boş döner."""
    gmap: dict[str, str] = {}
    candidates = list(RAW_DIR.rglob("gender_f0range.txt"))
    if not candidates:
        print("[uyarı] gender_f0range.txt bulunamadı; gender boş bırakılacak.")
        return gmap
    with open(candidates[0], encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower().startswith("jvs"):
                gmap[parts[0]] = parts[1]  # M / F
    print(f"[ok] gender bilgisi: {len(gmap)} konuşmacı")
    return gmap


def get_duration(wav_path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return 0.0


def build() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"JVS bulunamadı: {RAW_DIR}. Önce download_jvs.py çalıştır.")

    gender_map = load_gender_map()
    rows: list[dict] = []

    # Her konuşmacı klasörü: 'jvs001' .. 'jvs100'
    speaker_dirs = sorted(p for p in RAW_DIR.rglob("jvs[0-9][0-9][0-9]") if p.is_dir())
    print(f"[ok] {len(speaker_dirs)} konuşmacı klasörü bulundu.")

    for spk_dir in speaker_dirs:
        speaker_id = spk_dir.name
        gender = gender_map.get(speaker_id, "")
        for style, style_id in STYLE_DIRS.items():
            style_dir = spk_dir / style
            transcript = style_dir / "transcripts_utf8.txt"
            wav_dir = style_dir / "wav24kHz16bit"
            if not transcript.exists() or not wav_dir.exists():
                continue
            with open(transcript, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    file_id, text = line.split(":", 1)
                    wav_path = wav_dir / f"{file_id}.wav"
                    if not wav_path.exists():
                        continue
                    rows.append({
                        "audio_path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                        "text": text.strip(),
                        "speaker_id": speaker_id,
                        "style_id": style_id,
                        "gender": gender,
                        "duration_sec": get_duration(wav_path),
                    })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "text", "speaker_id", "style_id", "gender", "duration_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] Manifest yazıldı: {MANIFEST} ({len(rows)} kayıt)")


if __name__ == "__main__":
    build()
