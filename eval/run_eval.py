"""Замеры качества и скорости распознавания на собственных записях.

Запуск (из корня проекта):
    python evaluate.py --dataset eval/dataset
    python evaluate.py --model base --repeats 3 --label base_cpu

Скрипт использует тот же конвейер, что и бот: ffmpeg-конвертация и WhisperTranscriber.
Время загрузки модели считается отдельно, перед замерами выполняется прогревочный запуск.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Callable

from bot.audio import convert_to_wav
from bot.config import Settings
from bot.stt import WhisperTranscriber
from bot.textutils import is_meaningful
from eval.dataset import Sample, load_samples
from eval.metrics import char_stats, mean, median, percentile, word_stats

log = logging.getLogger("eval")


@dataclass
class SampleResult:
    name: str
    condition: str
    audio_sec: float = 0.0
    convert_ms: float = 0.0
    stt_ms: float = 0.0          # время распознавания (медиана по повторам)
    total_ms: float = 0.0
    rtf: float = math.nan        # время распознавания / длительность аудио
    ref_words: int = 0
    wer_s: int = 0
    wer_d: int = 0
    wer_i: int = 0
    wer: float = 0.0
    ref_chars: int = 0
    cer_errors: int = 0
    cer: float = 0.0
    dropped: int = 0             # сегментов отброшено как ненадёжные
    ok: bool = False             # обработано без ошибки и результат осмысленный
    error: str = ""
    reference: str = ""
    hypothesis: str = ""


SAMPLE_COLUMNS = [
    "name", "condition", "audio_sec", "convert_ms", "stt_ms", "total_ms", "rtf",
    "ref_words", "wer_s", "wer_d", "wer_i", "wer", "ref_chars", "cer_errors", "cer",
    "dropped", "ok", "error", "reference", "hypothesis",
]
SUMMARY_COLUMNS = [
    "group", "n", "ok", "ok_rate", "errors", "wer", "wer_substitutions", "wer_deletions",
    "wer_insertions", "cer", "audio_total_s", "convert_mean_ms", "stt_mean_ms",
    "stt_median_ms", "stt_p95_ms", "rtf_mean", "rtf_corpus", "dropped_segments",
]
_DIGITS = {
    "audio_sec": 2, "convert_ms": 0, "stt_ms": 0, "total_ms": 0, "rtf": 3, "wer": 4, "cer": 4,
    "ok_rate": 4, "audio_total_s": 2, "convert_mean_ms": 0, "stt_mean_ms": 0,
    "stt_median_ms": 0, "stt_p95_ms": 0, "rtf_mean": 3, "rtf_corpus": 3,
}


def wav_duration_sec(path: Path) -> float:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError, OSError):
        return 0.0


async def _evaluate_one(
    sample: Sample, transcriber, settings: Settings, workdir: Path, repeats: int, convert: Callable
) -> SampleResult:
    res = SampleResult(name=sample.name, condition=sample.condition, reference=sample.reference)
    wav = workdir / "sample.wav"
    wav.unlink(missing_ok=True)
    hypothesis = ""
    try:
        started = time.perf_counter()
        await convert(
            sample.audio_path, wav,
            normalize=settings.normalize_audio, retries=0, ffmpeg=settings.ffmpeg_path,
        )
        res.convert_ms = (time.perf_counter() - started) * 1000
        res.audio_sec = wav_duration_sec(wav)

        times: list[float] = []
        result = None
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            result = transcriber.transcribe(wav)
            times.append((time.perf_counter() - started) * 1000)
        res.stt_ms = median(times)
        hypothesis = result.text
        res.dropped = result.dropped
        res.ok = is_meaningful(hypothesis, max_ratio=settings.compression_ratio_threshold)
    except Exception as exc:  # запись не обработана: в метрики попадает как «ничего не распознано»
        detail = getattr(exc, "details", None) or str(exc)
        res.error = f"{type(exc).__name__}: {detail}"
        res.ok = False
        hypothesis = ""

    res.hypothesis = hypothesis
    res.total_ms = res.convert_ms + res.stt_ms
    if res.audio_sec > 0 and not res.error:
        res.rtf = res.stt_ms / (res.audio_sec * 1000)

    w = word_stats(sample.reference, hypothesis)
    c = char_stats(sample.reference, hypothesis)
    res.ref_words, res.wer_s, res.wer_d, res.wer_i, res.wer = (
        w.ref_len, w.substitutions, w.deletions, w.insertions, w.rate,
    )
    res.ref_chars, res.cer_errors, res.cer = c.ref_len, c.errors, c.rate
    return res


async def _warmup(sample: Sample, transcriber, settings: Settings, workdir: Path, convert: Callable) -> None:
    wav = workdir / "warmup.wav"
    try:
        await convert(
            sample.audio_path, wav,
            normalize=settings.normalize_audio, retries=0, ffmpeg=settings.ffmpeg_path,
        )
        transcriber.transcribe(wav)
    except Exception as exc:
        log.warning("Прогревочный запуск не удался: %s", exc)


def run_evaluation(
    samples: list[Sample],
    transcriber,
    settings: Settings,
    *,
    repeats: int = 1,
    warmup: bool = True,
    convert: Callable = convert_to_wav,
    on_result: Callable[[int, int, SampleResult], None] | None = None,
) -> list[SampleResult]:
    async def runner() -> list[SampleResult]:
        results: list[SampleResult] = []
        with tempfile.TemporaryDirectory(prefix="vnb_eval_") as tmp:
            workdir = Path(tmp)
            if warmup and samples:
                await _warmup(samples[0], transcriber, settings, workdir, convert)
            for index, sample in enumerate(samples, 1):
                result = await _evaluate_one(sample, transcriber, settings, workdir, repeats, convert)
                results.append(result)
                if on_result:
                    on_result(index, len(samples), result)
        return results

    return asyncio.run(runner())


def _summarize_group(group: str, rs: list[SampleResult]) -> dict:
    timed = [r for r in rs if not r.error]
    ref_words = sum(r.ref_words for r in rs)
    ref_chars = sum(r.ref_chars for r in rs)
    audio_total = sum(r.audio_sec for r in timed)
    stt_total = sum(r.stt_ms for r in timed)
    stt_times = [r.stt_ms for r in timed]
    ok = sum(1 for r in rs if r.ok)
    return {
        "group": group,
        "n": len(rs),
        "ok": ok,
        "ok_rate": ok / len(rs) if rs else math.nan,
        "errors": sum(1 for r in rs if r.error),
        "wer": sum(r.wer_s + r.wer_d + r.wer_i for r in rs) / ref_words if ref_words else math.nan,
        "wer_substitutions": sum(r.wer_s for r in rs),
        "wer_deletions": sum(r.wer_d for r in rs),
        "wer_insertions": sum(r.wer_i for r in rs),
        "cer": sum(r.cer_errors for r in rs) / ref_chars if ref_chars else math.nan,
        "audio_total_s": audio_total,
        "convert_mean_ms": mean([r.convert_ms for r in timed]),
        "stt_mean_ms": mean(stt_times),
        "stt_median_ms": median(stt_times),
        "stt_p95_ms": percentile(stt_times, 95),
        "rtf_mean": mean([r.rtf for r in timed if not math.isnan(r.rtf)]),
        "rtf_corpus": stt_total / (audio_total * 1000) if audio_total > 0 else math.nan,
        "dropped_segments": sum(r.dropped for r in rs),
    }


def summarize(results: list[SampleResult]) -> list[dict]:
    """Сводка по каждому условию записи и итоговая строка ALL."""
    groups: dict[str, list[SampleResult]] = {}
    for r in results:
        groups.setdefault(r.condition, []).append(r)
    rows = [_summarize_group(name, rs) for name, rs in sorted(groups.items())]
    rows.append(_summarize_group("ALL", results))
    return rows


def _fmt(key: str, value, excel: bool):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        digits = _DIGITS.get(key, 4)
        value = round(value, digits)
        if digits == 0:
            value = int(value)
        if excel:
            return str(value).replace(".", ",")
    return value


def _write_csv(path: Path, columns: list[str], rows: list[dict], style: str) -> None:
    excel = style == "excel"
    # Для русского Excel: разделитель «;», десятичная запятая и BOM, чтобы открывалась кириллица
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";" if excel else ",")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(c, row.get(c, ""), excel) for c in columns])


def write_samples_csv(path: Path, results: list[SampleResult], style: str = "excel") -> None:
    _write_csv(path, SAMPLE_COLUMNS, [dataclasses.asdict(r) for r in results], style)


def write_summary_csv(path: Path, rows: list[dict], style: str = "excel") -> None:
    _write_csv(path, SUMMARY_COLUMNS, rows, style)


def format_summary(rows: list[dict]) -> str:
    def pct(x):
        return "   -  " if math.isnan(x) else f"{x * 100:5.1f}%"

    def sec(x):
        return "    -" if math.isnan(x) else f"{x / 1000:5.2f}"

    head = f"{'группа':<12}{'n':>4}{'ok':>5}{'WER':>8}{'CER':>8}{'STT ср,с':>10}{'p95,с':>8}{'RTF':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        rtf = "    -" if math.isnan(r["rtf_corpus"]) else f"{r['rtf_corpus']:5.2f}"
        lines.append(
            f"{r['group']:<12}{r['n']:>4}{r['ok']:>5}{pct(r['wer']):>8}{pct(r['cer']):>8}"
            f"{sec(r['stt_mean_ms']):>10}{sec(r['stt_p95_ms']):>8}{rtf:>8}"
        )
    return "\n".join(lines)


def _versions() -> dict:
    out = {}
    for pkg in ("faster-whisper", "ctranslate2", "av"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def build_meta(settings: Settings, args, n_samples: int, model_load_s: float) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": settings.whisper_model,
        "device": settings.whisper_device,
        "compute_type": settings.whisper_compute_type,
        "language": settings.language,
        "beam_size": settings.beam_size,
        "vad_filter": settings.vad_filter,
        "normalize_audio": settings.normalize_audio,
        "thresholds": {
            "no_speech": settings.no_speech_threshold,
            "logprob": settings.logprob_threshold,
            "compression_ratio": settings.compression_ratio_threshold,
        },
        "repeats": args.repeats,
        "warmup": not args.no_warmup,
        "samples": n_samples,
        "model_load_s": round(model_load_s, 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "versions": _versions(),
        "note": "RAM и модель процессора впишите вручную, если нужны в отчёте",
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Замеры качества (WER, CER) и скорости распознавания речи")
    p.add_argument("--dataset", default="eval/dataset", help="папка с записями и эталонами")
    p.add_argument("--output", default="eval/results", help="папка для отчётов")
    p.add_argument("--model", help="tiny, base, small, medium, large-v3 (по умолчанию из .env)")
    p.add_argument("--device", help="cpu или cuda")
    p.add_argument("--compute-type", dest="compute_type", help="int8, float16 и т. д.")
    p.add_argument("--beam-size", dest="beam_size", type=int)
    p.add_argument("--repeats", type=int, default=1, help="сколько раз распознавать каждую запись (берётся медиана времени)")
    p.add_argument("--no-warmup", action="store_true", help="не делать прогревочный запуск")
    p.add_argument("--no-vad", action="store_true", help="выключить VAD-фильтр")
    p.add_argument("--normalize-audio", action="store_true", help="нормализация громкости ffmpeg")
    p.add_argument("--csv-style", choices=["excel", "plain"], default="excel",
                   help="excel: «;» и десятичная запятая; plain: «,» и точка")
    p.add_argument("--label", default="", help="метка, добавляется в имена файлов")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    settings = Settings.from_env()
    overrides = {}
    if args.model:
        overrides["whisper_model"] = args.model
    if args.device:
        overrides["whisper_device"] = args.device
    if args.compute_type:
        overrides["whisper_compute_type"] = args.compute_type
    if args.beam_size:
        overrides["beam_size"] = args.beam_size
    if args.no_vad:
        overrides["vad_filter"] = False
    if args.normalize_audio:
        overrides["normalize_audio"] = True
    settings = dataclasses.replace(settings, **overrides)

    try:
        samples, problems = load_samples(args.dataset)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    for problem in problems:
        print("Предупреждение:", problem)
    if not samples:
        print("В наборе нет пригодных записей. Нужны пары: имя.ogg и имя.txt с эталонным текстом.")
        return 2
    if shutil.which(settings.ffmpeg_path) is None:
        print(f"ffmpeg не найден ({settings.ffmpeg_path}). Укажите FFMPEG_PATH в .env.")
        return 2

    conditions = sorted({s.condition for s in samples})
    print(f"Записей: {len(samples)}, условия: {', '.join(conditions)}")
    print(f"Модель: {settings.whisper_model} ({settings.whisper_device}, {settings.whisper_compute_type})")

    transcriber = WhisperTranscriber(settings)
    print("Загрузка модели...")
    started = time.perf_counter()
    transcriber.load()
    model_load_s = time.perf_counter() - started
    print(f"Модель загружена за {model_load_s:.1f} с")

    def progress(i: int, total: int, r: SampleResult) -> None:
        if r.error:
            print(f"[{i}/{total}] {r.name}: ОШИБКА {r.error}")
        else:
            print(f"[{i}/{total}] {r.name}: WER {r.wer * 100:5.1f}%  STT {r.stt_ms / 1000:5.2f} с  RTF {r.rtf:5.2f}")

    results = run_evaluation(
        samples, transcriber, settings,
        repeats=args.repeats, warmup=not args.no_warmup, on_result=progress,
    )
    rows = summarize(results)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{settings.whisper_model}"
    if args.label:
        base += f"_{args.label}"
    samples_path = out / f"{base}_samples.csv"
    summary_path = out / f"{base}_summary.csv"
    meta_path = out / f"{base}_meta.json"
    write_samples_csv(samples_path, results, args.csv_style)
    write_summary_csv(summary_path, rows, args.csv_style)
    meta_path.write_text(
        json.dumps(build_meta(settings, args, len(samples), model_load_s), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(format_summary(rows))
    print()
    print("Файлы отчёта:")
    for path in (samples_path, summary_path, meta_path):
        print("  ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
