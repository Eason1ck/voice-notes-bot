"""Нормализация текста перед подсчётом WER и CER.

Эталон и результат распознавания приводятся к одному виду, иначе метрика будет считать
ошибкой разницу в регистре или пунктуации, которая нас не интересует.

Правила: нижний регистр, «ё» равно «е», пунктуация и знаки заменяются пробелом,
повторные пробелы схлопываются. Цифры не преобразуются в слова: если эталон написан словами,
а Whisper вывел цифры (или наоборот), это будет засчитано как ошибка. Поэтому в наборе лучше
не использовать числа.
"""
from __future__ import annotations

import re
import unicodedata

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    text = text.replace("_", " ")
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
