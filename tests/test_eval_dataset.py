import pytest

from eval.dataset import load_samples


def _touch(path, text=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        path.write_bytes(b"x")
    else:
        path.write_text(text, encoding="utf-8")


def test_load_pairs_and_conditions(tmp_path):
    _touch(tmp_path / "quiet" / "01.ogg")
    _touch(tmp_path / "quiet" / "01.txt", "Привет мир.")
    _touch(tmp_path / "noisy" / "01.m4a")
    _touch(tmp_path / "noisy" / "01.txt", "Купить хлеб.")
    _touch(tmp_path / "root.ogg")
    _touch(tmp_path / "root.txt", "Один.")
    samples, problems = load_samples(tmp_path)
    assert problems == []
    assert {s.name: s.condition for s in samples} == {
        "quiet/01": "quiet", "noisy/01": "noisy", "root": "default",
    }


def test_bom_is_stripped(tmp_path):
    _touch(tmp_path / "a.ogg")
    (tmp_path / "a.txt").write_bytes("﻿Текст".encode("utf-8"))
    samples, _ = load_samples(tmp_path)
    assert samples[0].reference == "Текст"


def test_problems(tmp_path):
    _touch(tmp_path / "no_ref.ogg")
    _touch(tmp_path / "empty.ogg")
    _touch(tmp_path / "empty.txt", "   ")
    _touch(tmp_path / "orphan.txt", "Текст")
    _touch(tmp_path / "phrases.txt", "список фраз")
    samples, problems = load_samples(tmp_path)
    assert samples == []
    text = "\n".join(problems)
    assert "no_ref" in text and "empty" in text and "orphan" in text
    assert "phrases" not in text


def test_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path / "nope")
