"""Конвертация аудио через ffmpeg: OGG/Opus (формат голосовых Telegram) -> WAV 16 кГц моно."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .errors import AudioConversionError

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT_SEC = 60


def _build_command(src: Path, dst: Path, normalize: bool, ffmpeg: str = "ffmpeg") -> list[str]:
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn",            # без видеодорожки
        "-ar", "16000",   # частота дискретизации 16 кГц
        "-ac", "1",       # моно
    ]
    if normalize:
        cmd += ["-af", "dynaudnorm"]
    cmd += ["-f", "wav", str(dst)]
    return cmd


async def _run_ffmpeg(src: Path, dst: Path, normalize: bool, ffmpeg: str = "ffmpeg") -> None:
    cmd = _build_command(src, dst, normalize, ffmpeg)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise AudioConversionError(details=f"ffmpeg не найден: {ffmpeg}") from exc

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise AudioConversionError(details="ffmpeg превысил лимит времени") from exc

    if proc.returncode != 0:
        tail = stderr.decode("utf-8", errors="replace")[-500:]
        raise AudioConversionError(details=f"ffmpeg завершился с кодом {proc.returncode}: {tail}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioConversionError(details="ffmpeg создал пустой файл")


async def convert_to_wav(
    src: Path, dst: Path, *, normalize: bool = False, retries: int = 1, ffmpeg: str = "ffmpeg"
) -> Path:
    """Конвертирует src в WAV 16 кГц моно. При сбое повторяет попытку retries раз.

    ffmpeg - имя команды или полный путь к ffmpeg.exe (настройка FFMPEG_PATH).
    """
    last_error: AudioConversionError | None = None
    for attempt in range(retries + 1):
        try:
            await _run_ffmpeg(src, dst, normalize, ffmpeg)
            return dst
        except AudioConversionError as exc:
            last_error = exc
            log.warning("Конвертация не удалась (попытка %d из %d): %s", attempt + 1, retries + 1, exc.details)
            if exc.details and "не найден" in exc.details:
                break  # повтор не поможет
    assert last_error is not None
    raise last_error
