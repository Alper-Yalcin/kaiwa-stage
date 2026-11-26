"""Karakter preset yükleme (configs/characters.yaml)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import CHARACTERS


@dataclass
class Character:
    name: str
    speaker_id: str
    style_id: str = "normal"
    style_weight: float = 0.7
    length_scale: float = 1.0
    sdp_ratio: float = 0.2
    noise: float = 0.6
    description: str = ""
    model_dir: str | None = None     # karakter başka bir modelden geliyorsa (ör. JVNV)


def list_characters() -> list[str]:
    """Tanımlı karakter adları (yaml'daki sıra ile)."""
    return [k for k in CHARACTERS if not str(k).startswith("_")]


def load_character(name: str) -> Character:
    if name not in CHARACTERS:
        raise KeyError(f"Karakter bulunamadı: {name}. Mevcut: {list_characters()}")
    return Character(name=name, **CHARACTERS[name])
