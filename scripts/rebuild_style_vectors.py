"""
Karma (JVS + JVNV) model için stil vektörlerini konuşmacı-ofsetsiz yeniden kurar.

SBV2'nin varsayılanı (save_styles_by_dirs) her stil klasörünün düz ortalamasını
alır. Duygu klasörleri yalnızca JVNV kliplerinden oluştuğu için "happy" vektörü
JVNV konuşmacılarının perde/tını ofsetini de taşır; JVS karakterine uygulanınca
duygu yerine perde +3…+7 st kayar (ölçüldü). Burada her stil, kendi alt-korpusunun
genel ortalamasına göre DELTA olarak kurulur:

    Neutral       = tüm kliplerin ortalaması
    <duygu>       = Neutral + (mean(JVNV, duygu) − mean(JVNV))
    normal/whisper/falsetto = Neutral + (mean(JVS, stil) − mean(JVS))

Inference'ta SBV2: mean + (style − mean) × weight → yalnızca duygu/stil yönü uygulanır.

Kullanım:
    python scripts/rebuild_style_vectors.py -m jvs_jvnv16
    (Data/<m>/wavs/**/*.npy'den okur; models/finetuned/<m>/style_vectors.npy + config.json günceller)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

JVS_STYLES = ["normal", "whisper", "falsetto"]
EMO_STYLES = ["anger", "disgust", "fear", "happy", "sad", "surprise"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-name", required=True)
    args = ap.parse_args()
    wavs = ROOT / "third_party" / "Style-Bert-VITS2" / "Data" / args.model_name / "wavs"
    dst = ROOT / "models" / "finetuned" / args.model_name

    groups: dict[str, list[np.ndarray]] = {}
    for f in wavs.rglob("*.npy"):
        groups.setdefault(f.parent.name, []).append(np.load(f))
    means = {k: np.mean(v, axis=0) for k, v in groups.items()}
    counts = {k: len(v) for k, v in groups.items()}
    all_vecs = [x for v in groups.values() for x in v]
    neutral = np.mean(all_vecs, axis=0)
    jvs_mean = np.mean([x for k in JVS_STYLES for x in groups.get(k, [])], axis=0)
    jvnv_mean = np.mean([x for k in EMO_STYLES for x in groups.get(k, [])], axis=0)

    style2id = {"Neutral": 0}
    rows = [neutral]
    for k in JVS_STYLES:
        if k in means:
            style2id[k] = len(rows)
            rows.append(neutral + (means[k] - jvs_mean))
    for k in EMO_STYLES:
        if k in means:
            style2id[k] = len(rows)
            rows.append(neutral + (means[k] - jvnv_mean))

    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["data"]["style2id"] = style2id
    cfg["data"]["num_styles"] = len(rows)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(dst / "style_vectors.npy", np.stack(rows).astype(np.float32))
    print(f"[ok] {dst / 'style_vectors.npy'} {np.stack(rows).shape}; klip sayıları {counts}")
    print(f"[ok] style2id = {style2id}")
    print(f"[info] |JVS−JVNV ortalama farkı| = {np.linalg.norm(jvs_mean - jvnv_mean):.2f} "
          f"(bu ofset artık stil vektörlerinde yok)")


if __name__ == "__main__":
    main()
