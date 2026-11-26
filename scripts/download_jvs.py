"""
JVS Corpus'u Google Drive'dan indirir ve data/raw/jvs altına açar.

Kullanım:
    python scripts/download_jvs.py

Notlar:
- JVS ~3 GB'lık bir zip. Google Drive büyük dosyalarda "virus scan"
  uyarısı verir; gdown bunu otomatik aşmaya çalışır.
- Otomatik indirme başarısız olursa, README'deki manuel indirme
  adımlarını izle ve zip'i data/raw/jvs.zip olarak koy, sonra bu
  script'i tekrar çalıştır (var olan zip'i açar).
- JVS ticari OLMAYAN araştırma içindir; lisansı README'de.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "jvs"
ZIP_PATH = ROOT / "data" / "raw" / "jvs.zip"

FILE_ID = os.environ.get("JVS_GDRIVE_FILE_ID", "19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt")


def download() -> None:
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        print(f"[skip] Zip zaten var: {ZIP_PATH}")
        return
    try:
        import gdown
    except ImportError:
        sys.exit("gdown kurulu değil. Önce: pip install -r requirements.txt")

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"[download] {url}\n[download] -> {ZIP_PATH}")
    gdown.download(url, str(ZIP_PATH), quiet=False)

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        sys.exit(
            "İndirme başarısız. README'deki MANUEL indirme adımlarını izle "
            f"ve zip'i şuraya koy: {ZIP_PATH}"
        )


def extract() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {ZIP_PATH} -> {RAW_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    print("[extract] tamam.")

    # JVS klasör yapısı kontrolü (jvs_ver1/jvs001/... beklenir)
    speakers = list(RAW_DIR.rglob("jvs001"))
    if speakers:
        print(f"[ok] JVS bulundu: {speakers[0]}")
    else:
        print("[uyarı] 'jvs001' klasörü bulunamadı. Zip içeriğini kontrol et.")


if __name__ == "__main__":
    download()
    extract()
