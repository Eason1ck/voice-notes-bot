"""Ошибки обработки голосовых сообщений.

У каждой ошибки есть user_message - дружелюбный текст, который уходит пользователю,
и details - техническая причина, которая записывается только в журнал.
"""
from __future__ import annotations


class BotError(Exception):
    default_message = "Что-то пошло не так. Попробуйте ещё раз."

    def __init__(self, user_message: str | None = None, *, details: str | None = None):
        self.user_message = user_message or self.default_message
        self.details = details
        super().__init__(details or self.user_message)


class TooLongError(BotError):
    default_message = "Сообщение слишком длинное. Запишите его короче или разбейте на несколько."


class TooLargeError(BotError):
    default_message = "Файл слишком большой. Запишите сообщение короче."


class DownloadError(BotError):
    default_message = "Не удалось получить файл от Telegram. Отправьте сообщение ещё раз."


class AudioConversionError(BotError):
    default_message = "Не удалось обработать аудио. Отправьте сообщение ещё раз."


class TranscriptionError(BotError):
    default_message = "Сейчас не получается распознать речь. Попробуйте чуть позже."


class ProcessingTimeoutError(BotError):
    default_message = "Обработка заняла слишком много времени. Попробуйте записать сообщение короче."


class NoSpeechError(BotError):
    default_message = (
        "Речь не распознана. Запишите сообщение ещё раз: в тихом месте, "
        "ближе к микрофону, медленнее и чётче."
    )
