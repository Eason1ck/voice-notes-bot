"""Распознавание речи: обёртка над faster-whisper.

Главная функция для следующих модулей: transcribe(path) -> str.
Если нужны сегменты, язык и оценки, используйте WhisperTranscriber.transcribe(path).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0


@dataclass
class TranscriptionResult:
    text: str
    language: str = ""
    language_probability: float = 0.0
    segments: list[Segment] = field(default_factory=list)  # только принятые сегменты
    dropped: int = 0                                        # сколько сегментов отброшено как ненадёжные
    duration_sec: float = 0.0
    processing_ms: int = 0
    model: str = ""


class WhisperTranscriber:
    """Загружает модель один раз и распознаёт файлы.

    Параметр model нужен для тестов: можно подставить заглушку вместо настоящей модели.
    """

    def __init__(self, settings: Settings, model=None):
        self.settings = settings
        self._model = model

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # импорт здесь: тяжёлая библиотека

        s = self.settings
        log.info("Загрузка модели Whisper: %s (%s, %s)", s.whisper_model, s.whisper_device, s.whisper_compute_type)
        started = time.perf_counter()
        self._model = WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type)
        log.info("Модель загружена за %.1f с", time.perf_counter() - started)

    def _is_unreliable(self, seg: Segment) -> bool:
        s = self.settings
        silent = seg.no_speech_prob > s.no_speech_threshold and seg.avg_logprob < s.logprob_threshold
        repetitive = seg.compression_ratio > s.compression_ratio_threshold
        return silent or repetitive

    def transcribe(self, path: str | Path) -> TranscriptionResult:
        self.load()
        s = self.settings
        started = time.perf_counter()
        raw_segments, info = self._model.transcribe(
            str(path),
            language=s.language,
            vad_filter=s.vad_filter,
            beam_size=s.beam_size,
        )

        accepted: list[Segment] = []
        dropped = 0
        for raw in raw_segments:  # генератор: распознавание идёт здесь
            seg = Segment(
                start=float(raw.start),
                end=float(raw.end),
                text=(raw.text or "").strip(),
                no_speech_prob=float(getattr(raw, "no_speech_prob", 0.0) or 0.0),
                avg_logprob=float(getattr(raw, "avg_logprob", 0.0) or 0.0),
                compression_ratio=float(getattr(raw, "compression_ratio", 0.0) or 0.0),
            )
            if not seg.text:
                continue
            if self._is_unreliable(seg):
                dropped += 1
                log.info("Сегмент %.1f-%.1f с отброшен как ненадёжный", seg.start, seg.end)
                continue
            accepted.append(seg)

        text = " ".join(seg.text for seg in accepted).strip()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = TranscriptionResult(
            text=text,
            language=getattr(info, "language", s.language),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            segments=accepted,
            dropped=dropped,
            duration_sec=float(getattr(info, "duration", 0.0) or 0.0),
            processing_ms=elapsed_ms,
            model=s.whisper_model,
        )
        log.info(
            "Распознано: аудио %.1f с, сегментов %d (отброшено %d), %d мс",
            result.duration_sec, len(accepted), dropped, elapsed_ms,
        )
        return result


_default: WhisperTranscriber | None = None


def transcribe(path: str | Path) -> str:
    """Интерфейс для следующего модуля: путь к аудиофайлу -> строка с текстом."""
    global _default
    if _default is None:
        _default = WhisperTranscriber(Settings.from_env())
    return _default.transcribe(path).text
