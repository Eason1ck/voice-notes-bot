"""Обработчики сообщений Telegram (aiogram 3)."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message

from .config import Settings
from .errors import BotError
from .pipeline import process_voice
from .stt import WhisperTranscriber
from .textutils import split_message

log = logging.getLogger(__name__)
router = Router()

WELCOME = (
    "Привет! Я превращаю голосовые сообщения в текст.\n\n"
    "Просто отправьте мне голосовое сообщение на русском языке, и я пришлю расшифровку."
)
HELP = (
    "Как пользоваться:\n"
    "1. Запишите голосовое сообщение и отправьте его сюда.\n"
    "2. Подождите несколько секунд, я пришлю текст.\n\n"
    "Советы: говорите чётко, ближе к микрофону и по возможности в тихом месте."
)
NOT_VOICE = "Я понимаю только голосовые сообщения. Запишите голосовое и отправьте его сюда."
GENERIC_ERROR = "Произошла непредвиденная ошибка. Попробуйте ещё раз чуть позже."


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(WELCOME)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP)


async def _finish(status: Message, message: Message, parts: list[str]) -> None:
    """Заменяет сообщение «распознаю…» первой частью ответа, остальные части отправляет отдельно."""
    first, rest = parts[0], parts[1:]
    try:
        await status.edit_text(first)
    except TelegramAPIError:
        await message.answer(first)
    for part in rest:
        await message.answer(part)


@router.message(F.voice)
async def on_voice(
    message: Message,
    bot: Bot,
    settings: Settings,
    transcriber: WhisperTranscriber,
    semaphore: asyncio.Semaphore,
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    voice = message.voice
    log.info("Голосовое: user=%s msg=%s duration=%s size=%s", user_id, message.message_id, voice.duration, voice.file_size)

    status = await message.reply("Принял, распознаю...")
    started = time.perf_counter()

    try:
        Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
        # Временная папка удаляется автоматически, даже если произошла ошибка
        with tempfile.TemporaryDirectory(prefix="vnb_", dir=settings.temp_dir) as tmp:
            result = await process_voice(
                bot=bot,
                voice=voice,
                settings=settings,
                transcriber=transcriber,
                semaphore=semaphore,
                workdir=Path(tmp),
            )
    except BotError as err:
        log.warning("Ошибка обработки: user=%s msg=%s %s: %s", user_id, message.message_id, type(err).__name__, err.details or err.user_message)
        await _finish(status, message, [err.user_message])
        return
    except Exception:
        log.exception("Непредвиденная ошибка: user=%s msg=%s", user_id, message.message_id)
        await _finish(status, message, [GENERIC_ERROR])
        return

    # Здесь модуль 2 (суммаризация, заголовок, темы) получит result.text:
    #     note = await nlp.process(result.text)

    parts = split_message(result.text, settings.message_chunk)
    if result.dropped:
        parts[-1] += "\n\n(Часть записи была неразборчива и пропущена.)"
    await _finish(status, message, parts)
    log.info(
        "Готово: user=%s msg=%s символов=%d всего=%d мс",
        user_id, message.message_id, len(result.text), int((time.perf_counter() - started) * 1000),
    )


@router.message()
async def on_other(message: Message) -> None:
    await message.answer(NOT_VOICE)
