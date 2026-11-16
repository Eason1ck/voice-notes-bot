"""Небольшие функции для работы с текстом (без внешних зависимостей)."""
from __future__ import annotations

import re
import zlib


def split_message(text: str, limit: int = 4000) -> list[str]:
    """Разбивает длинный текст на части не длиннее limit символов.

    Режет по переводу строки, затем по концу предложения, затем по пробелу.
    """
    text = text.strip()
    if not text:
        return []
    if limit < 1:
        raise ValueError("limit должен быть положительным")

    chunks: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if cut != -1:
                cut += 1  # оставляем знак препинания в первой части
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit
        chunk = text[:cut].strip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def compression_ratio(text: str) -> float:
    """Отношение размера текста к размеру после сжатия.

    Высокое значение означает много повторов (типичный признак «галлюцинации» Whisper).
    """
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def is_repetitive(text: str, *, window: int = 300, max_ratio: float = 2.4) -> bool:
    """Ищет зацикливание (повторы) по окнам текста.

    Проверять нужно по окнам, а не по всему тексту: длинный естественный текст сам по себе
    хорошо сжимается (русский текст на 3500+ символов даёт коэффициент выше 2,4), а в окнах по
    300 символов естественная речь остаётся заметно ниже порога.
    """
    for start in range(0, len(text), window):
        chunk = text[start:start + window]
        if len(chunk.encode("utf-8")) < 100:
            continue
        if compression_ratio(chunk) > max_ratio:
            return True
    return False


def is_meaningful(text: str, *, min_words: int = 2, max_ratio: float = 2.4) -> bool:
    """Проверяет, что распознанный текст не пуст, не слишком короткий и не состоит из повторов."""
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) < min_words:
        return False
    return not is_repetitive(text, max_ratio=max_ratio)
