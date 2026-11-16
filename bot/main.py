"""Запуск бота: python -m bot.main"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiogram import Bot, Dispatcher

from .config import Settings
from .handlers import router
from .stt import WhisperTranscriber

log = logging.getLogger("bot")


def setup_logging(settings: Settings) -> None:
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        Path(settings.log_dir) / "bot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


async def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings)

    if not settings.bot_token:
        log.error("Не задан BOT_TOKEN. Создайте файл .env по образцу .env.example.")
        sys.exit(1)
    if shutil.which(settings.ffmpeg_path) is None:
        log.error(
            "ffmpeg не найден (%s). Установите ffmpeg или впишите полный путь к ffmpeg.exe "
            "в переменную FFMPEG_PATH в файле .env (см. README).",
            settings.ffmpeg_path,
        )
        sys.exit(1)
    log.info("ffmpeg: %s", shutil.which(settings.ffmpeg_path))

    transcriber = WhisperTranscriber(settings)
    await asyncio.to_thread(transcriber.load)  # модель загружается один раз при старте

    bot = Bot(settings.bot_token)
    dp = Dispatcher(
        settings=settings,
        transcriber=transcriber,
        semaphore=asyncio.Semaphore(settings.max_concurrent_jobs),
    )
    dp.include_router(router)

    me = await bot.get_me()
    log.info("Бот запущен: @%s", me.username)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Бот остановлен")
