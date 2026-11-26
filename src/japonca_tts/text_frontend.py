"""
Japonca dil ön-yüzü (Aşama 2 / MVP 5+):

  normalize(text)  -> SBV2'ye girmeden ÖNCE metni okunabilir hale getirir.
  split_sentences  -> uzun metni cümle/parça listesine böler (synth.py kullanır).
  to_phonemes      -> pyopenjtalk G2P (QC için kana okunuş).
  apply_compat_patches -> pyopenjtalk-plus ↔ SBV2 g2p uyumsuzluklarını kapatır.

SBV2 kendi normalizasyonunu da yapar (rakam→kanji, tam/yarım genişlik, noktalama).
Burada SBV2'nin YANLIŞ / EKSİK bıraktığı yerler ele alınır (ölçüldü, 2026-09-17):
  * "10%"            -> SBV2: "十"  (yüzde yutuluyor)        -> "十パーセント"
  * "−3℃"            -> SBV2: "-三C"                         -> "マイナス三度"
  * "03-1234-5678"   -> SBV2: 三-千二百三十四-... (sayı gibi)   -> rakam rakam + "の"
  * "1/2"            -> SBV2: 一二 -> "カズジ"                 -> "二分の一"
  * "2026/9/17"      -> ...                                  -> "二千二十六年九月十七日"
  * "15:30"          -> ...                                  -> "十五時三十分"
  * "iPhone"         -> pyopenjtalk harf harf heceliyor        -> lexicon.yaml: "アイフォン"
  * "!"              -> pyopenjtalk-plus pron "！" döndürür, SBV2 g2p ValueError
                        ("Input must be katakana only") -> apply_compat_patches()
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEXICON_PATH = ROOT / "configs" / "lexicon.yaml"

# --- Rakam okunuşları (telefon numarası vb. tek tek okuma) ---
_DIGIT_KANA = {"0": "ゼロ", "1": "イチ", "2": "ニー", "3": "サン", "4": "ヨン",
               "5": "ゴー", "6": "ロク", "7": "ナナ", "8": "ハチ", "9": "キュー"}

# --- Birimler: sayıdan sonra gelirse ---
_UNITS = {
    "%": "パーセント", "％": "パーセント", "℃": "度", "°C": "度", "°": "度",
    "kg": "キログラム", "g": "グラム", "mg": "ミリグラム", "t": "トン",
    "km": "キロメートル", "cm": "センチメートル", "mm": "ミリメートル", "m": "メートル",
    "ml": "ミリリットル", "L": "リットル", "l": "リットル",
    "GB": "ギガバイト", "MB": "メガバイト", "KB": "キロバイト", "TB": "テラバイト",
    "GHz": "ギガヘルツ", "MHz": "メガヘルツ", "Hz": "ヘルツ",
    "kWh": "キロワットアワー", "kW": "キロワット", "W": "ワット", "V": "ボルト",
    "¥": "円", "￥": "円", "$": "ドル", "€": "ユーロ",
}
_UNIT_RE = re.compile(
    r"(?<=\d)\s*(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")(?![A-Za-z])"
)

# --- Sembol/işaret eşlemeleri (sayı bağlamından bağımsız) ---
_SYMBOLS = {
    "＆": "アンド", "&": "アンド", "＋": "プラス", "+": "プラス", "＝": "イコール", "=": "イコール",
    "×": "かける", "÷": "わる", "→": "、", "⇒": "、", "←": "、", "↑": "", "↓": "",
    "★": "", "☆": "", "♪": "", "♡": "", "♥": "", "※": "、", "＃": "", "#": "",
    "＠": "アット", "@": "アット", "／": "、", "｜": "、", "|": "、", "〜": "、", "~": "、",
    "＿": "", "_": "", "＊": "", "*": "", "「": "、", "」": "、", "『": "、", "』": "、",
    "【": "、", "】": "、", "（": "、", "）": "、", "(": "、", ")": "、", "［": "、", "］": "、",
    "[": "、", "]": "、", "《": "、", "》": "、", "〈": "、", "〉": "、",
}

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF⭐⭕‼⁉️]"
)


@lru_cache(maxsize=1)
def _lexicon() -> list[tuple[re.Pattern, str]]:
    """configs/lexicon.yaml -> [(regex, katakana)], uzun anahtar önce."""
    if not LEXICON_PATH.exists():
        return []
    data = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8")) or {}
    words = {str(k): str(v) for k, v in data.items() if not str(k).startswith("_")}
    out = []
    for k in sorted(words, key=len, reverse=True):
        # Latin sözcükler: kelime sınırı + büyük/küçük harf duyarsız
        if re.fullmatch(r"[A-Za-z0-9 .+\-']+", k):
            pat = re.compile(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", re.IGNORECASE)
        else:
            pat = re.compile(re.escape(k))
        out.append((pat, words[k]))
    return out


def _digits_kana(s: str) -> str:
    return "".join(_DIGIT_KANA.get(ch, ch) for ch in s)


def _phone(m: re.Match) -> str:
    return "の".join(_digits_kana(p) for p in re.split(r"[-‐－ー()（）\s]+", m.group(0)) if p)


def normalize(text: str) -> str:
    """Metni SBV2 öncesi normalize eder (okunuş garantisi). Çıktı yine Japonca metindir."""
    t = unicodedata.normalize("NFKC", text.strip())
    t = _EMOJI_RE.sub("", t)
    t = t.replace("　", " ")

    # 1) Tarih / saat / telefon / kesir  (sayı rakamlarıyken)
    t = re.sub(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)", r"\1年\2月\3日", t)
    t = re.sub(r"(?<!\d)(\d{1,2})[/／](\d{1,2})日", r"\1月\2日", t)
    t = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)",
               lambda m: f"{int(m.group(1))}時{int(m.group(2))}分" + (f"{int(m.group(3))}秒" if m.group(3) else ""), t)
    t = re.sub(r"(?<![\d\-])(0\d{1,4})[-‐－](\d{1,4})[-‐－](\d{3,4})(?![\d\-])", _phone, t)   # 03-1234-5678
    t = re.sub(r"(?<!\d)(\d{1,2})/(\d{1,3})(?![\d/])", r"\2分の\1", t)                       # 1/2 -> 二分の一

    # 2) Eksi işareti, önek para birimi ($20 -> 20ドル), yüzde/birim
    t = re.sub(r"(?<!\d)[−\-‐](?=\d)", "マイナス", t)
    t = re.sub(r"([$¥￥€])\s*(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(2) + _UNITS[m.group(1)], t)
    t = _UNIT_RE.sub(lambda m: _UNITS[m.group(1)], t)

    # 3) Sözlük (İngilizce/marka -> katakana)
    for pat, kana in _lexicon():
        t = pat.sub(kana, t)

    # 4) Semboller
    for k, v in _SYMBOLS.items():
        t = t.replace(k, v)

    # 5) Noktalama sadeleştirme: SBV2 çoklu/ardışık işaretlerde kırılgan
    t = re.sub(r"[!！]+", "！", t)
    t = re.sub(r"[?？]+", "？", t)
    t = re.sub(r"[？！]{2,}", lambda m: "？" if "？" in m.group(0) else "！", t)
    t = re.sub(r"[.。]{2,}|…+|‥+", "…", t)
    t = re.sub(r"[、,]{2,}", "、", t)
    t = re.sub(r"^[、，,]+|(?<=[。！？])[、，,]+", "", t)
    t = re.sub(r"[ \t]+", " ", t).strip(" 、")
    return t


_SENT_END = re.compile(r"(?<=[。！？!?\n])")


def split_sentences(text: str, max_chars: int = 80) -> list[str]:
    """Metni cümle sınırlarından böler; kısa cümleleri max_chars'a kadar gruplar,
    çok uzun cümleleri 、 üzerinden ikiye ayırır. Boş parça döndürmez."""
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    # Çok uzun tek cümleleri 、'dan böl
    pieces: list[str] = []
    for p in parts:
        if len(p) <= max_chars * 1.5:
            pieces.append(p)
            continue
        buf = ""
        for sub in re.split(r"(?<=、)", p):
            if buf and len(buf) + len(sub) > max_chars:
                pieces.append(buf.strip("、"))
                buf = ""
            buf += sub
        if buf.strip("、"):
            pieces.append(buf.strip("、"))
    # Kısa cümleleri grupla
    chunks: list[str] = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = ""
        buf += p
    if buf:
        chunks.append(buf)
    return chunks or [text]


def to_phonemes(text: str, kana: bool = False) -> str:
    """pyopenjtalk ile G2P. kana=True ise katakana okunuş döner."""
    import pyopenjtalk
    return pyopenjtalk.g2p(text, kana=kana)


_patched = False


def apply_compat_patches() -> None:
    """pyopenjtalk-plus, '!' için pron '！' döndürür; SBV2 g2p bunu tanımaz ve
    'Input must be katakana only: ！' ile çöker. pron'u '、' yaparak SBV2'nin
    kendi noktalama dalına (yomi=='、' ve kelime PUNCTUATIONS içinde -> '!'
    sembolü korunur) düşürürüz. Bir kez uygulanır; idempotent."""
    global _patched
    if _patched:
        return
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pw
    _orig = pw.run_frontend

    def run_frontend(text: str):
        out = _orig(text)
        for d in out:
            if d.get("pron") == "！":
                d["pron"] = "、"
        return out

    pw.run_frontend = run_frontend
    _patched = True
