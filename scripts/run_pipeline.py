"""
Tek komutla üretim + QC (MVP 5).

    batch_generate.py  (torch/CUDA, bu süreç)  ->  qc_batch.py (Whisper, AYRI süreç)

İki adım ayrı süreçlerde koşar: ctranslate2 + torch-CUDA aynı process'te cuDNN
çakışıyor (MVP 1 notu). Çıkış kodu: 0 = üretim ve QC'nin tamamı geçti.

Kullanım:
    python scripts/run_pipeline.py --texts tests/test_sentences.json --characters all --run mvp4
    python scripts/run_pipeline.py --texts cumleler.txt --characters chill_male --run demo --skip-qc
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Windows konsolu (cp1254) Japonca basamıyor; UTF-8'e zorla.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SCRIPTS = ROOT / "scripts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--characters", nargs="+", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--skip-qc", action="store_true")
    args = ap.parse_args()

    gen = [sys.executable, str(SCRIPTS / "batch_generate.py"), "--texts", args.texts,
           "--characters", *args.characters, "--run", args.run, "--device", args.device]
    if args.model_dir:
        gen += ["--model-dir", args.model_dir]
    print("[1/2] uretim:", " ".join(gen[1:]))
    gen_rc = subprocess.run(gen, cwd=ROOT).returncode

    manifest = ROOT / "outputs" / "logs" / f"{args.run}_manifest.csv"
    if args.skip_qc:
        sys.exit(gen_rc)
    if not manifest.exists():
        print("[hata] manifest yok, QC atlandi:", manifest)
        sys.exit(1)

    qc = [sys.executable, str(SCRIPTS / "qc_batch.py"), str(manifest),
          "--threshold", str(args.threshold), "--model", args.asr_model]
    print("\n[2/2] QC (ayri surec):", " ".join(qc[1:]))
    qc_rc = subprocess.run(qc, cwd=ROOT).returncode
    sys.exit(0 if gen_rc == 0 and qc_rc == 0 else 1)


if __name__ == "__main__":
    main()
