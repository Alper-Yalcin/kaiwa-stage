"""
Okunuş denetimi (üretimden ÖNCE): Open JTalk (SBV2'nin G2P'si) ile SudachiPy
sözlüğünün okunuşunu karşılaştırır, anlaşmazlıkları listeler ve lexicon.yaml
için düzeltme satırı önerir.

Neden: Open JTalk sözlüğü bazı bileşik kelimeleri yanlış okuyor (ölçüldü,
2026-09-17): 長押し→ナゲシシ (doğru ナガオシ), 叩き→ハタキ (タタキ), 一段落→
イチダンオチ (イチダンラク), 明王→アキラオー (ミョーオー), 海の幸→コー (サチ),
乾ドック→イヌイ (カン). JVS metinlerinde ~20 cümlede 1. Sudachi de hatasız değil
(中央駅→チュンアンエキ, 見つける→ミッケル), bu yüzden OTOMATİK DEĞİŞTİRME YOK:
insan bakar, doğru olanı lexicon.yaml'a ekler (normalize() metni değiştirir).

Kullanım:
    python scripts/check_reading.py "まず電源ボタンを三秒間長押しし、画面が点灯します。"
    python scripts/check_reading.py --file cumleler.txt          # satır başına bir cümle
    python scripts/check_reading.py --file tests/stress_sentences.json
Çıktı: her anlaşmazlık için  OJT「…」 / Sudachi「…」  + öneri satırı.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.text_frontend import normalize  # noqa: E402

_ROW: dict[str, str] = {}
for _v, _chars in {"a": "アカサタナハマヤラワガザダバパャァ", "i": "イキシチニヒミリギジヂビピィ",
                   "u": "ウクスツヌフムユルグズヅブプュゥッ", "e": "エケセテネヘメレゲゼデベペェ",
                   "o": "オコソトノホモヨロヲゴゾドボポョォ"}.items():
    for _c in _chars:
        _ROW[_c] = _v

_PARTICLES = {"は": "ワ", "へ": "エ", "を": "オ"}
_KNOWN_ALT = {"明日", "今日", "昨日", "一日", "二日", "十日", "何時", "他人事", "漁場", "両脚", "星"}  # iki okunuş da geçerli


def canon(k: str) -> str:
    """Uzun ünlü / ヅヂヲ yazım farklarını kanonikleştirir (キョウ=キョー)."""
    k = k.replace("ヅ", "ズ").replace("ヂ", "ジ").replace("ヲ", "オ")
    out: list[str] = []
    for ch in k:
        if out:
            pv = _ROW.get(out[-1])
            if ((ch == "ウ" and pv in ("o", "u")) or (ch == "イ" and pv in ("e", "i"))
                    or (ch == "ア" and pv == "a") or (ch == "エ" and pv == "e") or (ch == "オ" and pv == "o")):
                out.append("ー")
                continue
        out.append(ch)
    return re.sub(r"[^ァ-ヶー]", "", "".join(out))


def analyze(text: str) -> list[dict]:
    import pyopenjtalk
    from sudachipy import dictionary, tokenizer
    tok = dictionary.Dictionary(dict="core").create()
    mode = tokenizer.Tokenizer.SplitMode.C

    from style_bert_vits2.nlp.japanese.normalizer import normalize_text
    norm = normalize_text(normalize(text))   # rakam->kanji: OJT kelime dizgileri metinle eşleşsin
    # OJT kelimelerinin karakter aralıkları (bağlam içi okunuş için hizalama)
    ojt_words, pos_i = [], 0
    for d in pyopenjtalk.run_frontend(norm):
        w = d["string"]
        start = norm.find(w, pos_i)
        if start < 0:
            start = pos_i
        ojt_words.append((start, start + len(w), d["pron"]))
        pos_i = start + len(w)

    findings = []
    for m in tok.tokenize(norm, mode):
        surf = m.surface()
        pos = m.part_of_speech()[0]
        if pos in ("記号", "補助記号", "空白") or surf in _KNOWN_ALT:
            continue
        if surf in _PARTICLES and pos == "助詞":
            continue
        if re.search(r"[0-9A-Za-z]", surf) or not re.search(r"[一-鿿]", surf):
            continue  # rakam/Latin: OJT'ye bırak; kana-only: fark okunuş değil
        if re.fullmatch(r"[〇零一二三四五六七八九十百千万億兆点]+[日分秒時月年個]?", surf):
            continue  # sayı kanjileri: Sudachi düz okur, OJT bağlamlı (ジュッパーセント) -> OJT doğru
        su = m.reading_form()
        if not su or su == "*":
            continue
        b, e = m.begin(), m.end()
        # Sudachi kelimesini tam örten OJT kelimeleri (kısmi örtüşmede karşılaştırma güvenilmez)
        cover = [(ws, we, pr) for ws, we, pr in ojt_words if ws < e and we > b]
        if not cover or cover[0][0] != b or cover[-1][1] != e:
            continue
        ojt_ctx = "".join(pr for _, _, pr in cover)
        o, s = canon(ojt_ctx), canon(su)
        if o and s and o != s and len(o) + len(s) >= 3:
            findings.append({"word": surf, "pos": pos, "ojt": ojt_ctx, "sudachi": su})
    return findings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--file")
    args = ap.parse_args()

    if args.file:
        p = Path(args.file)
        if p.suffix == ".json":
            texts = [s["text"] for s in json.loads(p.read_text(encoding="utf-8"))["sentences"]]
        else:
            texts = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        texts = [args.text]
    else:
        ap.error("metin ya da --file ver")

    import pyopenjtalk
    total = 0
    for t in texts:
        f = analyze(t)
        print(f"\n{t}")
        print(f"  okunuş: {pyopenjtalk.g2p(normalize(t), kana=True)}")
        for x in f:
            total += 1
            print(f"  (?) {x['word']} [{x['pos']}]  OJT「{x['ojt']}」 / Sudachi「{x['sudachi']}」"
                  f"   -> Sudachi doğruysa lexicon.yaml: {x['word']}: {x['sudachi']}")
    print(f"\n{len(texts)} cümle, {total} anlaşmazlık. Kural: ikisi de yanılabilir; kulakla/sözlükle doğrula.")


if __name__ == "__main__":
    main()
