"""WAV -> MP3 dönüştürme + loudness normalize (Aşama 5, FFmpeg)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import EXPORT


def _resolve_ffmpeg() -> str:
    """ffmpeg çalıştırılabilirini bulur.

    Sıra: FFMPEG_BIN env > PATH > winget kurulum yolu.
    """
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    found = shutil.which("ffmpeg")
    if found:
        return found
    # winget (Gyan.FFmpeg) varsayılan kurulum yolu
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.exists():
        for exe in base.rglob("ffmpeg.exe"):
            return str(exe)
    raise FileNotFoundError(
        "ffmpeg bulunamadı. PATH'e ekleyin veya FFMPEG_BIN ortam değişkenini ayarlayın."
    )


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    ffmpeg = _resolve_ffmpeg()
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path)]
    if EXPORT.get("loudness_normalize"):
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += [
        "-ar", str(EXPORT["sample_rate"]),
        "-b:a", EXPORT["bitrate"],
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True)
    return mp3_path
