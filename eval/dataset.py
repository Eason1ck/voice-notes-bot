"""Чтение набора записей: аудиофайл + текстовый файл с тем же именем (эталон).

Структура папки:
    eval/dataset/quiet/01.ogg   eval/dataset/quiet/01.txt
    eval/dataset/noisy/01.ogg   eval/dataset/noisy/01.txt
Имя первой вложенной папки считается условием записи (quiet, noisy и т. д.).
Файлы прямо в корне набора получают условие «default».
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".ogg", ".oga", ".opus", ".wav", ".mp3", ".m4a", ".flac", ".aac", ".webm"}
DEFAULT_CONDITION = "default"


@dataclass(frozen=True)
class Sample:
    name: str          # относительный путь без расширения, например quiet/01
    audio_path: Path
    reference: str     # эталонный текст
    condition: str


def load_samples(root: str | Path) -> tuple[list[Sample], list[str]]:
    """Возвращает (записи, предупреждения). Записи без эталона пропускаются с предупреждением."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Папка с набором не найдена: {root}")

    samples: list[Sample] = []
    problems: list[str] = []
    used_refs: set[Path] = set()

    for audio in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS):
        rel = audio.relative_to(root)
        name = rel.with_suffix("").as_posix()
        ref_path = audio.with_suffix(".txt")
        if not ref_path.is_file():
            problems.append(f"{rel.as_posix()}: нет файла с эталонным текстом ({ref_path.name}), запись пропущена")
            continue
        reference = ref_path.read_text(encoding="utf-8-sig").strip()
        if not reference:
            problems.append(f"{ref_path.relative_to(root).as_posix()}: эталон пустой, запись пропущена")
            continue
        used_refs.add(ref_path)
        condition = rel.parts[0] if len(rel.parts) > 1 else DEFAULT_CONDITION
        samples.append(Sample(name=name, audio_path=audio, reference=reference, condition=condition))

    for txt in sorted(root.rglob("*.txt")):
        if txt not in used_refs and txt.name.lower() not in {"readme.txt", "phrases.txt"}:
            has_audio = any(txt.with_suffix(ext).is_file() for ext in AUDIO_EXTENSIONS)
            if not has_audio:
                problems.append(f"{txt.relative_to(root).as_posix()}: нет аудиофайла с таким именем")
    return samples, problems
