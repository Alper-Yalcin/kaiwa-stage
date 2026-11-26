"""
Eğitilmiş bir SBV2 checkpoint'ini daha fazla konuşmacıyla devam ettirmek için
speaker embedding'ini (emb_g) genişletir.

Neden: SBV2 load_checkpoint, şekli uyuşmayan tensörü (12x256 -> 16x256) sessizce
rastgele başlatır; 12 JVS sesinin kimliği kaybolur. Burada mevcut satırlar aynen
korunur, yeni satırlar mevcutların ortalaması + küçük gürültü ile başlar
(yeni konuşmacılar hızlı yakınsar, eskiler bozulmaz). Adam durumu da (exp_avg,
exp_avg_sq) aynı biçimde sıfırla genişletilir; aksi halde optimizer yüklenemez
ve eğitim "sıfırdan" başlar.

Kullanım:
    python scripts/expand_checkpoint.py --src third_party/Style-Bert-VITS2/Data/jvs_multi12/models
                                        --dst third_party/Style-Bert-VITS2/Data/jvs_jvnv16/models
                                        --n-speakers 16 [--step 13839]
D_*.pth ve WD_*.pth konuşmacıdan bağımsızdır; olduğu gibi kopyalanır.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def latest(d: Path, prefix: str, step: int | None) -> Path:
    if step is not None:
        return d / f"{prefix}_{step}.pth"
    cands = sorted(d.glob(f"{prefix}_*.pth"),
                   key=lambda p: int(re.search(r"_(\d+)\.pth$", p.name).group(1)))
    if not cands:
        raise FileNotFoundError(f"{prefix}_*.pth yok: {d}")
    return cands[-1]


def expand_rows(t: torch.Tensor, n: int, noise: float) -> torch.Tensor:
    if t.shape[0] >= n:
        return t
    extra = n - t.shape[0]
    if noise > 0:
        base = t.mean(dim=0, keepdim=True).repeat(extra, 1)
        base = base + noise * t.std(dim=0, keepdim=True) * torch.randn_like(base)
    else:
        base = torch.zeros(extra, *t.shape[1:], dtype=t.dtype)
    return torch.cat([t, base], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n-speakers", type=int, required=True)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    g_path = latest(src, "G", args.step)
    ck = torch.load(g_path, map_location="cpu")
    w = ck["model"]["emb_g.weight"]
    old_n = w.shape[0]
    ck["model"]["emb_g.weight"] = expand_rows(w, args.n_speakers, noise=0.1)
    print(f"[G] emb_g {tuple(w.shape)} -> {tuple(ck['model']['emb_g.weight'].shape)}")

    # Adam durumu: emb_g'ye ait exp_avg / exp_avg_sq (aynı eski şekil) sıfırla genişletilir.
    n_fixed = 0
    for st in ck.get("optimizer", {}).get("state", {}).values():
        for key in ("exp_avg", "exp_avg_sq"):
            v = st.get(key)
            if isinstance(v, torch.Tensor) and v.shape == w.shape:
                st[key] = expand_rows(v, args.n_speakers, noise=0.0)
                n_fixed += 1
    print(f"[G] optimizer: {n_fixed} tensör genişletildi (beklenen 2)")
    if n_fixed != 2:
        print("  (!) beklenmeyen sayı; optimizer yüklenemezse eğitim sıfırdan başlar — kontrol et")

    torch.save(ck, dst / g_path.name)
    for prefix in ("D", "WD"):
        p = latest(src, prefix, args.step)
        shutil.copy(p, dst / p.name)
    print(f"[ok] {dst}: {g_path.name} (+D/WD), epoch={ck.get('iteration')}, "
          f"{old_n} -> {args.n_speakers} konuşmacı")


if __name__ == "__main__":
    main()
