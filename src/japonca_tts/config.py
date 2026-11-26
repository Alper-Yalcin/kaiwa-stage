"""Merkezi config: paths.yaml ve characters.yaml'ı okur, mutlak yollara çevirir."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


PATHS = _load_yaml("configs/paths.yaml")
CHARACTERS = _load_yaml("configs/characters.yaml")

OUTPUTS = {
    "wav": ROOT / PATHS["outputs"]["wav"],
    "mp3": ROOT / PATHS["outputs"]["mp3"],
    "logs": ROOT / PATHS["outputs"]["logs"],
}
EXPORT = PATHS["export"]
AUDIO = PATHS["audio"]
SBV2_REPO = ROOT / PATHS["third_party"]["sbv2_repo"]
PROSODY = PATHS.get("prosody", {})
