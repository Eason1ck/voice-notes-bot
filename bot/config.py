"""Настройки бота. Все значения можно переопределить в файле .env или в переменных окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    """Минимальный загрузчик .env (без внешних зависимостей).

    Уже заданные переменные окружения не перезаписываются.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    # Telegram
    bot_token: str = ""

    # Whisper (faster-whisper)
    whisper_model: str = "small"        # tiny, base, small, medium, large-v3
    whisper_device: str = "cpu"         # cpu или cuda
    whisper_compute_type: str = "int8"  # для CPU: int8; для GPU: float16
    language: str = "ru"
    beam_size: int = 5
    vad_filter: bool = True

    # Лимиты (предварительные значения, подбираются по результатам тестов)
    max_duration_sec: int = 180         # максимальная длительность голосового
    max_file_mb: int = 20               # лимит скачивания файлов Bot API
    process_timeout_sec: int = 180      # лимит времени на распознавание
    max_concurrent_jobs: int = 1        # сколько записей распознаётся одновременно

    # Пороги отбраковки сегментов (значения по умолчанию из Whisper)
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

    # Прочее
    ffmpeg_path: str = "ffmpeg"         # имя команды или полный путь к ffmpeg.exe
    normalize_audio: bool = False       # нормализация громкости ffmpeg (dynaudnorm)
    message_chunk: int = 4000           # лимит сообщения Telegram 4096 символов
    temp_dir: str = "tmp"
    log_dir: str = "logs"
    log_level: str = "INFO"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env") -> "Settings":
        _load_dotenv(dotenv_path)
        d = cls()
        return cls(
            bot_token=os.environ.get("BOT_TOKEN", d.bot_token).strip(),
            whisper_model=os.environ.get("WHISPER_MODEL", d.whisper_model),
            whisper_device=os.environ.get("WHISPER_DEVICE", d.whisper_device),
            whisper_compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", d.whisper_compute_type),
            language=os.environ.get("LANGUAGE", d.language),
            beam_size=_get_int("BEAM_SIZE", d.beam_size),
            vad_filter=_get_bool("VAD_FILTER", d.vad_filter),
            max_duration_sec=_get_int("MAX_DURATION_SEC", d.max_duration_sec),
            max_file_mb=_get_int("MAX_FILE_MB", d.max_file_mb),
            process_timeout_sec=_get_int("PROCESS_TIMEOUT_SEC", d.process_timeout_sec),
            max_concurrent_jobs=_get_int("MAX_CONCURRENT_JOBS", d.max_concurrent_jobs),
            no_speech_threshold=_get_float("NO_SPEECH_THRESHOLD", d.no_speech_threshold),
            logprob_threshold=_get_float("LOGPROB_THRESHOLD", d.logprob_threshold),
            compression_ratio_threshold=_get_float(
                "COMPRESSION_RATIO_THRESHOLD", d.compression_ratio_threshold
            ),
            ffmpeg_path=os.environ.get("FFMPEG_PATH", "").strip() or d.ffmpeg_path,
            normalize_audio=_get_bool("NORMALIZE_AUDIO", d.normalize_audio),
            message_chunk=_get_int("MESSAGE_CHUNK", d.message_chunk),
            temp_dir=os.environ.get("TEMP_DIR", d.temp_dir),
            log_dir=os.environ.get("LOG_DIR", d.log_dir),
            log_level=os.environ.get("LOG_LEVEL", d.log_level).upper(),
        )
