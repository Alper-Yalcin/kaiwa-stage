"""
Style-Bert-VITS2 inference sarmalayıcısı (Aşama 5 / MVP 1, 3, 4).

- MVP 1: models/pretrained altındaki JP-Extra modelini kullanır.
- MVP 3+: models/finetuned altında bir model varsa onu tercih eder
  (birden çok safetensors varsa en yüksek adımlı olan seçilir).
- MVP 4: karakterin speaker_id'si (jvsXXX) ve style_id'si (normal/whisper/
  falsetto) modelin spk2id/style2id tablosuna eşlenir; eşleşmeyen stil
  Neutral'a, eşleşmeyen konuşmacı 0'a düşer ve uyarı loglanır.

Karakter parametreleri (length_scale, sdp_ratio, noise, style) SBV2 infer
çağrısına model SEVİYESİNDE geçirilir. DSP pitch shift kullanılmaz.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .characters import Character
from .config import ROOT, PATHS, PROSODY
from .prosody import apply_question_rise
from .text_frontend import normalize, split_sentences, apply_compat_patches, to_phonemes

BERT_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
_bert_loaded = False
log = logging.getLogger("japonca_tts.synth")


def _step_of(p: Path) -> int:
    """'jvs_multi12_e12_s6000.safetensors' -> 6000; adım yoksa -1."""
    m = re.search(r"_s(\d+)$", p.stem)
    return int(m.group(1)) if m else -1


def pick_checkpoint(mdir: Path) -> Path:
    """Klasördeki safetensors'lardan en yüksek adımlı olanı seçer."""
    ckpts = sorted(mdir.glob("*.safetensors"), key=lambda p: (_step_of(p), p.name))
    if not ckpts:
        raise FileNotFoundError(f"safetensors yok: {mdir}")
    return ckpts[-1]


def _ensure_bert() -> None:
    """Japonca DeBERTa'yı bir kez yükler."""
    global _bert_loaded
    if _bert_loaded:
        return
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    bert_models.load_model(Languages.JP, BERT_REPO)
    bert_models.load_tokenizer(Languages.JP, BERT_REPO)
    apply_compat_patches()   # '!' -> SBV2 g2p çökmesin (pyopenjtalk-plus uyumu)
    _bert_loaded = True


def _discover_model_dir(model_dir: Optional[str]) -> Path:
    """Kullanılacak model klasörünü bulur.

    Öncelik: açıkça verilen > paths.yaml models.default (varsa) > models/finetuned
    (varsa) > models/pretrained. Bir model klasörü; config.json + *.safetensors +
    style_vectors.npy içerir.
    """
    if model_dir:
        return ROOT / model_dir
    default = PATHS["models"].get("default")
    if default and (ROOT / default / "config.json").exists():
        return ROOT / default

    def _has_model(d: Path) -> Optional[Path]:
        for sub in [d, *[p for p in d.glob("*") if p.is_dir()]]:
            if (sub / "config.json").exists() and any(sub.glob("*.safetensors")):
                return sub
        return None

    for key in ("finetuned", "pretrained"):
        found = _has_model(ROOT / PATHS["models"][key])
        if found:
            return found
    raise FileNotFoundError(
        "Model bulunamadı: models/finetuned/<model>/ (config.json + *.safetensors + style_vectors.npy)"
    )


@lru_cache(maxsize=4)
def get_model(model_dir: Optional[str] = None, device: str = "cuda"):
    """TTSModel'i yükler ve cache'ler."""
    from style_bert_vits2.tts_model import TTSModel
    _ensure_bert()

    mdir = _discover_model_dir(model_dir)
    config = mdir / "config.json"
    safet = pick_checkpoint(mdir)
    style_vec = mdir / "style_vectors.npy"
    if not style_vec.exists():
        raise FileNotFoundError(f"style_vectors.npy yok: {mdir}")

    model = TTSModel(
        model_path=safet,
        config_path=config,
        style_vec_path=style_vec,
        device=device,
    )
    # Üretim metadata'sı için (MVP 5 log)
    model._jtts_model_dir = mdir            # type: ignore[attr-defined]
    model._jtts_checkpoint = safet.name     # type: ignore[attr-defined]
    return model


def model_info(model_dir: Optional[str] = None, device: str = "cuda") -> dict:
    """Yüklü modelin kimliği: klasör, checkpoint, konuşmacılar, stiller."""
    m = get_model(model_dir, device)
    return {
        "model_dir": str(getattr(m, "_jtts_model_dir", "")),
        "checkpoint": getattr(m, "_jtts_checkpoint", ""),
        "speakers": list((getattr(m, "spk2id", None) or {}).keys()),
        "styles": list((getattr(m, "style2id", None) or {}).keys()),
    }


def _resolve_speaker_id(model, speaker_id: str) -> int:
    """Karakterin speaker_id'sini (ör. 'jvs042') modelin int id'sine çevirir.

    Eşleşme yoksa (ör. tek-konuşmacılı pretrained model) 0 döner.
    """
    spk2id = getattr(model, "spk2id", None) or {}
    if speaker_id in spk2id:
        return int(spk2id[speaker_id])
    if str(speaker_id).isdigit():
        return int(speaker_id)
    if len(spk2id) > 1:
        log.warning("speaker_id %r modelde yok (%s); 0 kullanılıyor",
                    speaker_id, list(spk2id))
    return 0


def _resolve_style(model, style_id: str) -> str:
    """Karakter stilini modelin tanıdığı stile eşler; yoksa 'Neutral'."""
    style2id = getattr(model, "style2id", None) or {}
    if style_id in style2id:
        return style_id
    if len(style2id) > 1:
        log.warning("style_id %r modelde yok (%s); Neutral kullanılıyor",
                    style_id, list(style2id))
    return "Neutral"


def _to_float(audio) -> np.ndarray:
    """SBV2 int16 ya da float dönebilir; float32 [-1, 1]'e getirir."""
    audio = np.asarray(audio)
    if audio.dtype.kind in ("i", "u"):
        audio = audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


# Kısa girdi modu: 1-4 moralık tek kelimeler (猫, 手, はい) normal ayarlarla
# 0.2-0.4 s'ye sıkışıp bozuluyor (ölçüldü: 目->メイ, 火->ヒン). Bu eşiğin altında
# tempo yavaşlatılır, örnekleme deterministikleştirilir, başa/sona sessizlik konur.
SHORT_MORA = 4
SHORT_LENGTH_MUL = 1.25
SHORT_SDP_MAX = 0.1
SHORT_NOISE_MAX = 0.4
SHORT_PAD_SEC = 0.25
_TERMINALS = "。！？…"


def _mora_count(text: str) -> int:
    kana = to_phonemes(text, kana=True)
    return sum(1 for ch in kana if ch not in "ァィゥェォャュョ、。！？，．・ 　'")


def _pause_sec(chunk: str) -> float:
    """Parça sonundaki işarete göre eklenecek sessizlik."""
    end = chunk.rstrip()[-1:] if chunk.strip() else ""
    if end in "。！？!?":
        return 0.35
    return 0.2


def synthesize_wav(
    text: str,
    character: Character,
    out_wav: Path,
    model_dir: Optional[str] = None,
    device: str = "cuda",
    max_chunk_chars: int = 80,
    style: Optional[str] = None,
    style_weight: Optional[float] = None,
    question_rise: Optional[bool] = None,
) -> Path:
    """Metni karakterle seslendirir, 44.1 kHz WAV yazar.

    style/style_weight: cümle bazlı duygu geçersiz kılma (dialog.json). Model o
    stili tanımıyorsa karakterin kendi stili kullanılır (Neutral'a düşmez).

    Uzun metin cümle sınırlarından parçalanır (text_frontend.split_sentences),
    her parça ayrı sentezlenir ve cümle sonu işaretine göre 0.2–0.35 s sessizlikle
    birleştirilir. VITS tipi modeller ~100+ karakterde aceleye getirip hece
    düşürebildiği için bu, uzun paragraflarda anlaşılırlığı korur.
    """
    from style_bert_vits2.constants import Languages

    if question_rise is None:
        question_rise = bool(PROSODY.get("question_rise", True))
    model = get_model(model_dir or character.model_dir, device)
    sid = _resolve_speaker_id(model, character.speaker_id)
    style2id = getattr(model, "style2id", None) or {}
    if style and style in style2id:
        weight = character.style_weight if style_weight is None else style_weight
    else:
        style, weight = _resolve_style(model, character.style_id), character.style_weight

    norm = normalize(text)
    if not norm:
        raise ValueError("Boş metin (normalize sonrası)")
    if norm[-1] not in _TERMINALS:
        norm += "。"   # cümle sonu düşüşü; sondaki hece yutulmasın
    chunks = split_sentences(norm, max_chars=max_chunk_chars)

    length, sdp, noise = character.length_scale, character.sdp_ratio, character.noise
    pad = 0.0
    if len(chunks) == 1 and _mora_count(norm) <= SHORT_MORA:
        length *= SHORT_LENGTH_MUL
        sdp, noise = min(sdp, SHORT_SDP_MAX), min(noise, SHORT_NOISE_MAX)
        pad = SHORT_PAD_SEC

    pieces: list[np.ndarray] = []
    sr = 44100
    for i, chunk in enumerate(chunks):
        sr, audio = model.infer(
            text=chunk,
            language=Languages.JP,
            speaker_id=sid,
            sdp_ratio=sdp,
            noise=noise,
            length=length,
            style=style,
            style_weight=weight,
        )
        piece = _to_float(audio)
        if question_rise and chunk.rstrip()[-1:] in "？?":
            piece = apply_question_rise(piece, sr,
                                        target_st=float(PROSODY.get("question_rise_st", 5.0)),
                                        stretch=float(PROSODY.get("question_stretch", 1.3)))
        pieces.append(piece)
        if i < len(chunks) - 1:
            pieces.append(np.zeros(int(sr * _pause_sec(chunk)), dtype=np.float32))

    audio = np.concatenate(pieces)
    if pad:
        z = np.zeros(int(sr * pad), dtype=np.float32)
        audio = np.concatenate([z, audio, z])

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sr)
    return out_wav
