"""
Eğitilmiş SBV2 modelini inference klasörüne kurar (MVP 3b -> 4 köprüsü).

third_party/Style-Bert-VITS2/model_assets/<model>/
    config.json + style_vectors.npy + <model>_e*_s*.safetensors (çok sayıda)
        -> models/finetuned/<model>/  (config + style_vectors + SEÇİLEN tek checkpoint)

synth.py, models/finetuned altında model bulursa pretrained yerine onu kullanır.

Kullanım:
    python scripts/install_finetuned.py -m jvs_multi12              # en yüksek adım
    python scripts/install_finetuned.py -m jvs_multi12 --step 6000  # belirli checkpoint
    python scripts/install_finetuned.py -m jvs_multi12 --list
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Windows konsolu (cp1254) Japonca basamıyor; UTF-8'e zorla.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.config import SBV2_REPO, PATHS       # noqa: E402
from japonca_tts.synth import pick_checkpoint, _step_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-name", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    src = SBV2_REPO / "model_assets" / args.model_name
    if not src.exists():
        raise SystemExit(f"Model assets yok: {src}")
    ckpts = sorted(src.glob("*.safetensors"), key=_step_of)
    if args.list:
        for c in ckpts:
            print(f"  step={_step_of(c):6}  {c.name}  {c.stat().st_size/1e6:.0f} MB")
        return

    if args.step is not None:
        match = [c for c in ckpts if _step_of(c) == args.step]
        if not match:
            raise SystemExit(f"step {args.step} bulunamadı. --list ile bak.")
        ckpt = match[0]
    else:
        ckpt = pick_checkpoint(src)

    dst = ROOT / PATHS["models"]["finetuned"] / args.model_name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("config.json", "style_vectors.npy"):
        shutil.copy2(src / name, dst / name)
    shutil.copy2(ckpt, dst / ckpt.name)

    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    print(f"[ok] {ckpt.name} -> {dst}")
    print(f"[ok] konusmacilar: {list(cfg['data']['spk2id'])}")
    print(f"[ok] stiller     : {list(cfg['data']['style2id'])}")


if __name__ == "__main__":
    main()
