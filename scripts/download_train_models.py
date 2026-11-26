"""
SBV2 eğitimi için SADECE gerekli model ağırlıklarını indirir (JP-only).
initialize.py'nin tamamını çalıştırmak EN/ZH BERT + inference modelleri dahil
~5 GB indirir; biz yalnızca şunları indiriyoruz (~2.5 GB):

  - JP DeBERTa ağırlığı  -> bert/deberta-v2-large-japanese-char-wwm/pytorch_model.bin
  - WavLM (SLM, eğitim)  -> slm/wavlm-base-plus/pytorch_model.bin
  - JP-Extra base        -> pretrained_jp_extra/{G_0,D_0,WD_0}.safetensors

(config/tokenizer dosyaları repoda zaten mevcut.)
Ayrıca configs/paths.yml yoksa default'tan kopyalanır.

Kullanım (.venv-train ile):
    python scripts/download_train_models.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1] / "third_party" / "Style-Bert-VITS2"

JOBS = [
    ("ku-nlp/deberta-v2-large-japanese-char-wwm", "pytorch_model.bin",
     REPO / "bert" / "deberta-v2-large-japanese-char-wwm"),
    ("microsoft/wavlm-base-plus", "pytorch_model.bin",
     REPO / "slm" / "wavlm-base-plus"),
    ("litagin/Style-Bert-VITS2-2.0-base-JP-Extra", "G_0.safetensors",
     REPO / "pretrained_jp_extra"),
    ("litagin/Style-Bert-VITS2-2.0-base-JP-Extra", "D_0.safetensors",
     REPO / "pretrained_jp_extra"),
    ("litagin/Style-Bert-VITS2-2.0-base-JP-Extra", "WD_0.safetensors",
     REPO / "pretrained_jp_extra"),
]


def main() -> None:
    for repo_id, file, dest in JOBS:
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / file).exists():
            print(f"[skip] {dest.name}/{file} zaten var")
            continue
        print(f"[get ] {repo_id} :: {file} -> {dest}")
        hf_hub_download(repo_id, file, local_dir=str(dest))
        print(f"[ok  ] {dest / file}")

    # configs/paths.yml
    paths = REPO / "configs" / "paths.yml"
    default = REPO / "configs" / "default_paths.yml"
    if not paths.exists() and default.exists():
        shutil.copy(default, paths)
        print(f"[ok  ] {paths} olusturuldu (default'tan)")

    print("\nTamam. Sonraki: preprocess_all.py -m jvs_smoke ...")


if __name__ == "__main__":
    main()
