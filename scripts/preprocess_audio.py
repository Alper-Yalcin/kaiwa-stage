"""
JVS manifest'ini eğitim için FİLTRELER ve doğrular (Aşama 1 / MVP 2).

Tasarım kararı: Style-Bert-VITS2'nin kendi eğitim ön-işlemesi (repo'daki
resample.py) ham wav'ları config örnekleme oranına (44.1 kHz) zaten getirir.
Bu yüzden burada 15k dosyayı YENİDEN resample edip ~8.5 GB ara dosya yazmıyoruz.
Bunun yerine:
  - süre filtresi (çok kısa / çok uzun ele)
  - dosya varlığı / okunabilirlik kontrolü
  - filtrelenmiş manifest (ham 24k yollarını korur) + istatistik raporu

Resample MVP 3'te SBV2 tarafından yapılır.

Kullanım:
    python scripts/preprocess_audio.py
    python scripts/preprocess_audio.py --speakers jvs001 jvs002   # alt-küme
    python scripts/preprocess_audio.py --verify-audio             # her dosyayı aç (yavaş)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))

MANIFEST = ROOT / CFG["metadata"]["manifest"]
OUT_MANIFEST = ROOT / "metadata" / "jvs_manifest_processed.csv"

MIN_D = CFG["audio"]["min_duration_sec"]
MAX_D = CFG["audio"]["max_duration_sec"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", nargs="*", help="Sadece bu konuşmacılar (ör. jvs001)")
    ap.add_argument("--verify-audio", action="store_true",
                    help="Her dosyayı açıp okunabilirliği doğrula (yavaş)")
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST)
    n0 = len(df)
    reasons = {}

    if args.speakers:
        df = df[df["speaker_id"].isin(args.speakers)]

    # 1) dosya varlığı
    exists = df["audio_path"].apply(lambda p: (ROOT / p).exists())
    reasons["dosya_yok"] = int((~exists).sum())
    df = df[exists]

    # 2) süre filtresi
    dur_ok = (df["duration_sec"] >= MIN_D) & (df["duration_sec"] <= MAX_D)
    reasons["sure_disi"] = int((~dur_ok).sum())
    df = df[dur_ok]

    # 3) (opsiyonel) ses okunabilirlik doğrulaması
    if args.verify_audio:
        import soundfile as sf
        from tqdm import tqdm
        good = []
        for p in tqdm(df["audio_path"], desc="verify"):
            try:
                sf.info(str(ROOT / p)); good.append(True)
            except Exception:
                good.append(False)
        good = pd.Series(good, index=df.index)
        reasons["bozuk_ses"] = int((~good).sum())
        df = df[good]

    df.to_csv(OUT_MANIFEST, index=False, encoding="utf-8")

    # --- rapor ---
    print(f"Girdi kayit       : {n0}")
    for k, v in reasons.items():
        print(f"  elendi ({k:10}): {v}")
    print(f"Kalan kayit       : {len(df)}")
    print(f"Konusmaci sayisi  : {df['speaker_id'].nunique()}")
    print(f"Toplam sure (saat): {round(df['duration_sec'].sum()/3600, 2)}")
    print("Stil dagilimi     :")
    for s, c in df["style_id"].value_counts().items():
        print(f"  {s:9}: {c}")
    print(f"\n[ok] -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
