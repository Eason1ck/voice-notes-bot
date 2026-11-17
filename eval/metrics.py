"""Метрики качества и скорости (без внешних зависимостей).

WER = (S + D + I) / N, где S - замены, D - удаления, I - вставки, N - число слов в эталоне.
CER считается так же, но по символам (пробелы между словами тоже считаются символами).
При суммировании по набору складываются числа ошибок и длины эталонов, а не средние проценты:
длинные записи тогда влияют на итог сильнее коротких.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from eval.textnorm import normalize_text


@dataclass(frozen=True)
class ErrorStats:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Доля ошибок. Для пустого эталона: 0 при отсутствии ошибок, иначе 1 (вырожденный случай)."""
        if self.ref_len == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.ref_len

    def __add__(self, other: "ErrorStats") -> "ErrorStats":
        return ErrorStats(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.ref_len + other.ref_len,
        )


def edit_ops(ref: Sequence, hyp: Sequence) -> ErrorStats:
    """Расстояние Левенштейна с разбивкой на замены, удаления и вставки.

    Для каждой ячейки хранится кортеж (стоимость, замены, удаления, вставки), память O(len(hyp)).
    Если оптимальных выравниваний несколько, общее число ошибок одинаково,
    а разбивка по типам может зависеть от выбранного выравнивания.
    """
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        r = ref[i - 1]
        for j in range(1, m + 1):
            if r == hyp[j - 1]:
                best = prev[j - 1]
            else:
                c, s, d, ins = prev[j - 1]
                best = (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]            # слово эталона пропущено
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, ins)
            c, s, d, ins = cur[j - 1]         # лишнее слово в результате
            if c + 1 < best[0]:
                best = (c + 1, s, d, ins + 1)
            cur.append(best)
        prev = cur
    _, s, d, ins = prev[m]
    return ErrorStats(s, d, ins, n)


def word_stats(reference: str, hypothesis: str) -> ErrorStats:
    return edit_ops(normalize_text(reference).split(), normalize_text(hypothesis).split())


def char_stats(reference: str, hypothesis: str) -> ErrorStats:
    return edit_ops(list(normalize_text(reference)), list(normalize_text(hypothesis)))


def percentile(values: Sequence[float], p: float) -> float:
    """Процентиль с линейной интерполяцией (как numpy.percentile по умолчанию)."""
    if not values:
        return math.nan
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan
