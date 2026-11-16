from pathlib import Path

from bot.textutils import compression_ratio, is_meaningful, is_repetitive, split_message

LONG_RU = (Path(__file__).parent / "data" / "long_ru_sample.txt").read_text(encoding="utf-8")


def test_long_natural_text_is_not_rejected():
    # Целиком такой текст сжимается лучше порога 2,4 (поэтому проверка идёт по окнам),
    # но настоящая длинная речь не должна считаться «повторами».
    assert compression_ratio(LONG_RU) > 2.4
    assert not is_repetitive(LONG_RU)
    assert is_meaningful(LONG_RU)


def test_repetition_inside_long_text_is_found():
    assert is_repetitive(LONG_RU + " " + "Спасибо за просмотр. " * 30)


def test_split_short_text_unchanged():
    assert split_message("Привет, мир", 4000) == ["Привет, мир"]


def test_split_empty_text():
    assert split_message("   ", 100) == []


def test_split_respects_limit_and_keeps_all_words():
    text = " ".join(f"слово{i}" for i in range(500))
    parts = split_message(text, 200)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_split_prefers_sentence_end():
    text = ("Это первое предложение. " * 5) + ("Это второе предложение. " * 5)
    parts = split_message(text, 100)
    assert all(len(p) <= 100 for p in parts)
    assert all(p.endswith(".") for p in parts)


def test_split_long_word_without_spaces():
    parts = split_message("а" * 250, 100)
    assert [len(p) for p in parts] == [100, 100, 50]


def test_compression_ratio_repetition_is_high():
    assert compression_ratio("Спасибо за просмотр. " * 30) > 2.4
    assert compression_ratio("") == 0.0


def test_is_meaningful_normal_text():
    assert is_meaningful("Нужно подготовить отчёт по курсовой до пятницы")


def test_is_meaningful_rejects_empty_and_too_short():
    assert not is_meaningful("")
    assert not is_meaningful("   ")
    assert not is_meaningful("Да")


def test_is_meaningful_rejects_repetition():
    assert not is_meaningful("Спасибо за просмотр. " * 30)
