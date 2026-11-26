"""
SBV2 eğitim veri klasörünü hazırlar: Data/<model>/raw/*.wav + esd.list.
Hem SMOKE (alt-küme) hem TAM veri için aynı script kullanılır.

Kaynak: metadata/jvs_manifest_processed.csv (MVP 2 çıktısı).
Hedef : <dataset-root>/<model>/raw/  + <dataset-root>/<model>/esd.list
        (dataset-root varsayılan: SBV2 reposu içindeki Data/)

Benzersiz wav adı: jvsXXX_<style>_<orijinal>.wav (stiller çakışmasın diye).

Örnekler:
    # SMOKE: 2 konuşmacı, normal stil, her birinden en çok 60 cümle
    python scripts/prepare_sbv2_data.py -m jvs_smoke --speakers jvs001 jvs002 \
        --styles normal --max-per-speaker 60

    # TAM veri: tüm konuşmacılar, tüm stiller
    python scripts/prepare_sbv2_data.py -m jvs_full --all

    # MVP 3b: 12 konuşmacı, stil alt-klasörlü (SBV2 normal/whisper/falsetto
    # stil vektörlerini otomatik üretsin), konuşmacı başına 60 normal + tüm
    # whisper/falsetto
    python scripts/prepare_sbv2_data.py -m jvs_multi12 --speakers jvs001 ...         --style-dirs --normal-per-speaker 60

--style-dirs: wav'lar raw/<style>/ altına konur ve esd.list'te "<style>/<ad>.wav"
yazılır. SBV2 train_ms_jp_extra.py, wavs/ altındaki alt klasörlerden stil
vektörü çıkarır (default_style.save_styles_by_dirs) -> style2id:
{Neutral, falsetto, normal, whisper}.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "metadata" / "jvs_manifest_processed.csv"
DEFAULT_DATASET_ROOT = ROOT / "third_party" / "Style-Bert-VITS2" / "Data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model-name", required=True)
    ap.add_argument("--speakers", nargs="*", help="Seçili konuşmacılar (ör. jvs001)")
    ap.add_argument("--all", action="store_true", help="Tüm konuşmacılar")
    ap.add_argument("--styles", nargs="*", default=None,
                    help="Stil filtresi: normal whisper falsetto (varsayılan: hepsi)")
    ap.add_argument("--max-per-speaker", type=int, default=0,
                    help="Konuşmacı başına en çok N cümle (0 = sınırsız)")
    ap.add_argument("--normal-per-speaker", type=int, default=0,
                    help="Konuşmacı başına en çok N 'normal' cümle (rastgele, seed 42); "
                         "whisper/falsetto'nun tamamı kalır (0 = sınırsız)")
    ap.add_argument("--style-dirs", action="store_true",
                    help="wav'ları raw/<style>/ alt klasörlerine koy (SBV2 çoklu stil)")
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    args = ap.parse_args()

    if not PROC.exists():
        raise SystemExit("Önce: python scripts/preprocess_audio.py")
    if not args.all and not args.speakers:
        raise SystemExit("--all veya --speakers vermelisin.")

    df = pd.read_csv(PROC)
    if args.speakers:
        df = df[df["speaker_id"].isin(args.speakers)]
    if args.styles:
        df = df[df["style_id"].isin(args.styles)]
    if args.max_per_speaker > 0:
        df = df.groupby("speaker_id", group_keys=False).head(args.max_per_speaker)
    if args.normal_per_speaker > 0:
        normal = df[df["style_id"] == "normal"]
        normal = (normal.groupby("speaker_id", group_keys=False)
                  .apply(lambda g: g.sample(n=min(len(g), args.normal_per_speaker),
                                            random_state=42)))
        df = pd.concat([normal, df[df["style_id"] != "normal"]])
        df = df.sort_values(["speaker_id", "style_id", "audio_path"])

    if df.empty:
        raise SystemExit("Seçime uyan kayıt yok.")

    data_dir = Path(args.dataset_root) / args.model_name
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    esd_lines = []
    for _, row in df.iterrows():
        src = ROOT / row["audio_path"]
        name = f"{row['speaker_id']}_{row['style_id']}_{src.stem}.wav"
        if args.style_dirs:
            (raw_dir / row["style_id"]).mkdir(exist_ok=True)
            rel = f"{row['style_id']}/{name}"
        else:
            rel = name
        shutil.copy2(src, raw_dir / rel)
        text = str(row["text"]).replace("|", " ").strip()
        esd_lines.append(f"{rel}|{row['speaker_id']}|JP|{text}")

    (data_dir / "esd.list").write_text("\n".join(esd_lines) + "\n", encoding="utf-8")

    print(f"[ok] model        : {args.model_name}")
    print(f"[ok] hedef        : {data_dir}")
    print(f"[ok] konusmaci    : {df['speaker_id'].nunique()}  -> {sorted(df['speaker_id'].unique())}")
    print(f"[ok] cumle/wav    : {len(esd_lines)}")
    print(f"[ok] stiller      : {dict(df['style_id'].value_counts())}")
    print(f"[ok] toplam sure  : {round(df['duration_sec'].sum()/60,1)} dk")
    print(f"\nSonraki: preprocess_all.py -m {args.model_name} ... (.venv-train ile)")


if __name__ == "__main__":
    main()
