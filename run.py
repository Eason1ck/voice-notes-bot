"""Запуск бота: python run.py (или кнопка Run в PyCharm для этого файла)."""
import asyncio
import logging

from bot.main import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger("bot").info("Бот остановлен")
