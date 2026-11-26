"""
JVNV korpusunu (4 konuşmacı × 6 duygu, CC BY-SA 4.0) SBV2 eğitim formatına hazırlar.

JVNV'nin her cümlesi bir sözsüz ses (NV: gülme, ağlama, "ああ？" vb.) içerir ve
nv_label/ altında zaman aralığı verilir. TTS için NV aralığı SESTEN KESİLİR,
NV öbeği METİNDEN çıkarılır; geriye duygulu, soru/ünlem dolu temiz konuşma
cümlesi kalır (soru tonlaması ve vurgu için eğitim verisi — MVP 6).

Çıktı: third_party/Style-Bert-VITS2/Data/<model>/raw/<emotion>/jvnv_<spk>_<emotion>_<session>_<id>.wav
       + <model>/esd_jvnv.list  (satır: <emotion>/<ad>.wav|jvnv_<spk>|JP|metin)

Kullanım:
    python scripts/prepare_jvnv_data.py -m jvs_jvnv16                 # hepsi
    python scripts/prepare_jvnv_data.py -m jvs_jvnv16 --max-sec 12 --speakers F1 M1
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

JVNV = ROOT / "data" / "raw" / "jvnv" / "jvnv_v1"
DATA_ROOT = ROOT / "third_party" / "Style-Bert-VITS2" / "Data"
EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
TARGET_SR = 44100
XFADE = 0.02   # NV kesildikten sonra parçaları birleştirirken çapraz geçiş (s)


def load_transcripts() -> dict[str, tuple[str, str]]:
    out = {}
    for line in io.open(JVNV / "transcription.csv", encoding="utf-8"):
        if not line.strip():
            continue
        key, nv, text = line.rstrip("\n").split("|")[:3]
        out[key] = (nv, text)
    return out


def clean_text(nv: str, text: str) -> str:
    """NV öbeğini metinden çıkarır ('NA' = metinde NV yok)."""
    if nv and nv != "NA" and nv in text:
        text = text.replace(nv, "", 1)
    text = text.strip(" 　、，,")
    if text and text[-1] not in "。！？!?":
        text += "。"
    return text


def cut_spans(y: np.ndarray, sr: int, spans: list[tuple[float, float]]) -> np.ndarray:
    """Verilen aralıkları sesten çıkarır; ekleri kısa çapraz geçişle birleştirir."""
    keep, pos = [], 0
    for s, e in sorted(spans):
        a, b = int(s * sr), int(e * sr)
        if a > pos:
            keep.append(y[pos:a])
        pos = max(pos, b)
    if pos < len(y):
        keep.append(y[pos:])
    keep = [k for k in keep if len(k) > 0]
    if not keep:
        return y[:0]
    n = int(XFADE * sr)
    out = keep[0]
    for k in keep[1:]:
        if len(out) > n and len(k) > n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            out = np.concatenate([out[:-n], out[-n:] * (1 - ramp) + k[:n] * ramp, k[n:]])
        else:
            out = np.concatenate([out, k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-name", required=True)
    ap.add_argument("--speakers", nargs="*", default=["F1", "F2", "M1", "M2"])
    ap.add_argument("--emotions", nargs="*", default=EMOTIONS)
    ap.add_argument("--max-sec", type=float, default=12.0, help="NV kesildikten sonra bu süreden uzunlar atlanır")
    ap.add_argument("--min-sec", type=float, default=0.8)
    args = ap.parse_args()

    tr = load_transcripts()
    raw_dir = DATA_ROOT / args.model_name / "raw"
    lines, skipped, total_sec = [], {"long": 0, "short": 0, "notext": 0}, 0.0
    for spk in args.speakers:
        for emo in args.emotions:
            for wav in sorted((JVNV / spk / emo).rglob("*.wav")):
                base = wav.stem                              # F1_anger_regular_01
                key = "_".join(base.split("_")[1:])          # anger_regular_01
                nv, text = tr.get(key, ("", ""))
                text = clean_text(nv, text)
                if len(text) < 2:
                    skipped["notext"] += 1
                    continue
                label = JVNV / "nv_label" / spk / f"{base}.txt"
                spans = [tuple(map(float, l.split()[:2]))
                         for l in io.open(label, encoding="utf-8") if l.strip()] if label.exists() else []
                y, sr = sf.read(str(wav), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                y = cut_spans(y, sr, spans)
                dur = len(y) / sr
                if dur > args.max_sec:
                    skipped["long"] += 1
                    continue
                if dur < args.min_sec:
                    skipped["short"] += 1
                    continue
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                name = f"jvnv_{base}.wav"
                (raw_dir / emo).mkdir(parents=True, exist_ok=True)
                sf.write(str(raw_dir / emo / name), y, TARGET_SR, subtype="PCM_16")
                lines.append(f"{emo}/{name}|jvnv_{spk}|JP|{text}\n")
                total_sec += dur
        print(f"[{spk}] toplam {len(lines)} satır")

    out = DATA_ROOT / args.model_name / "esd_jvnv.list"
    out.write_text("".join(lines), encoding="utf-8")
    q = sum("？" in l.split("|")[3] for l in lines)
    e = sum("！" in l.split("|")[3] for l in lines)
    print(f"[ok] {out}: {len(lines)} cümle, {total_sec / 60:.1f} dk, soru {q}, ünlem {e}; atlanan {skipped}")


if __name__ == "__main__":
    main()
