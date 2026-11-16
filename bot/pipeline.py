"""Конвейер обработки голосового сообщения (без зависимости от aiogram, поэтому легко тестируется).

Шаги: проверка лимитов -> скачивание -> конвертация ffmpeg -> распознавание -> проверка результата.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .audio import convert_to_wav
from .config import Settings
from .errors import (
    BotError,
    DownloadError,
    NoSpeechError,
    ProcessingTimeoutError,
    TooLargeError,
    TooLongError,
    TranscriptionError,
)
from .stt import TranscriptionResult, WhisperTranscriber
from .textutils import is_meaningful

log = logging.getLogger(__name__)


async def process_voice(
    *,
    bot,
    voice,
    settings: Settings,
    transcriber: WhisperTranscriber,
    semaphore: asyncio.Semaphore,
    workdir: Path,
) -> TranscriptionResult:
    """Обрабатывает одно голосовое сообщение и возвращает результат распознавания.

    bot должен иметь метод download(file_id, destination=path) (как в aiogram 3).
    voice должен иметь file_id, duration и file_size (как aiogram.types.Voice).
    Бросает BotError с понятным пользователю текстом.
    """
    duration = getattr(voice, "duration", None)
    if duration is not None and duration > settings.max_duration_sec:
        minutes = settings.max_duration_sec / 60
        raise TooLongError(
            f"Сообщение слишком длинное ({duration} с). Максимум {minutes:g} мин. "
            "Запишите короче или разбейте на несколько сообщений.",
            details=f"duration={duration}",
        )

    size = getattr(voice, "file_size", None)
    if size is not None and size > settings.max_file_bytes:
        raise TooLargeError(
            f"Файл слишком большой (лимит {settings.max_file_mb} МБ). Запишите сообщение короче.",
            details=f"file_size={size}",
        )

    ogg = workdir / "voice.ogg"
    wav = workdir / "voice.wav"

    try:
        await bot.download(voice.file_id, destination=ogg)
    except Exception as exc:  # TelegramBadRequest, сетевые ошибки и т. д.
        raise DownloadError(details=f"{type(exc).__name__}: {exc}") from exc

    await convert_to_wav(
        ogg, wav, normalize=settings.normalize_audio, retries=1, ffmpeg=settings.ffmpeg_path
    )

    async with semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(transcriber.transcribe, wav),
                timeout=settings.process_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise ProcessingTimeoutError(details=f"timeout {settings.process_timeout_sec} s") from exc
        except BotError:
            raise
        except Exception as exc:
            raise TranscriptionError(details=f"{type(exc).__name__}: {exc}") from exc

    if not is_meaningful(result.text, max_ratio=settings.compression_ratio_threshold):
        raise NoSpeechError(details=f"пустой или повторяющийся результат, отброшено сегментов: {result.dropped}")

    return result
