import math

from eval.metrics import ErrorStats, char_stats, edit_ops, mean, median, percentile, word_stats
from eval.textnorm import normalize_text


def test_normalize_text():
    assert normalize_text("  Привет,   Мир!\nЁлка_и  ёж. ") == "привет мир елка и еж"


def test_edit_ops_identical():
    s = edit_ops(["а", "б"], ["а", "б"])
    assert (s.substitutions, s.deletions, s.insertions, s.ref_len) == (0, 0, 0, 2)


def test_edit_ops_kinds():
    assert edit_ops(["а", "б", "в"], ["а", "х", "в"]).substitutions == 1
    assert edit_ops(["а", "б", "в"], ["а", "в"]).deletions == 1
    assert edit_ops(["а", "в"], ["а", "б", "в"]).insertions == 1
    assert edit_ops(["а", "б"], []).deletions == 2
    assert edit_ops([], ["а"]).insertions == 1


def test_word_wer():
    s = word_stats("Один два три четыре", "один, два, три")
    assert s.errors == 1 and s.rate == 0.25


def test_wer_ignores_case_and_punctuation():
    assert word_stats("Привет, мир!", "привет мир").rate == 0


def test_cer():
    s = char_stats("кот", "кит")
    assert math.isclose(s.rate, 1 / 3)


def test_rate_empty_reference():
    assert ErrorStats().rate == 0.0
    assert ErrorStats(insertions=2).rate == 1.0


def test_add():
    a = ErrorStats(1, 2, 3, 10)
    b = ErrorStats(0, 1, 0, 5)
    c = a + b
    assert (c.substitutions, c.deletions, c.insertions, c.ref_len) == (1, 3, 3, 15)
    assert c.errors == 7


def test_percentile():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 50) == 3
    assert math.isclose(percentile(xs, 95), 4.8)
    assert percentile([7], 95) == 7
    assert math.isnan(percentile([], 50))
    assert median([3, 1, 2]) == 2
    assert mean([1, 2, 3]) == 2
    assert math.isnan(mean([]))
