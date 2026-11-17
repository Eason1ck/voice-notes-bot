import csv
import math
import wave
from types import SimpleNamespace

from bot.config import Settings
from eval.dataset import Sample
from eval.run_eval import (
    run_evaluation, summarize, wav_duration_sec, write_samples_csv, write_summary_csv,
)


async def fake_convert(src, dst, *, normalize=False, retries=1, ffmpeg="ffmpeg"):
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 2)  # 2 секунды


class FakeTranscriber:
    def __init__(self, text="привет мир сегодня хорошо", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def transcribe(self, path):
        self.calls += 1
        if self.fail:
            raise RuntimeError("модель сломалась")
        return SimpleNamespace(text=self.text, dropped=1)


def _samples(tmp_path):
    audio = tmp_path / "a.ogg"
    audio.write_bytes(b"x")
    return [
        Sample("quiet/01", audio, "Привет, мир сегодня хорошо", "quiet"),
        Sample("noisy/01", audio, "Привет мир сегодня очень хорошо", "noisy"),
    ]


def test_wav_duration(tmp_path):
    import asyncio
    p = tmp_path / "x.wav"
    asyncio.run(fake_convert(None, p))
    assert math.isclose(wav_duration_sec(p), 2.0)
    assert wav_duration_sec(tmp_path / "missing.wav") == 0.0


def test_run_and_summarize(tmp_path):
    tr = FakeTranscriber()
    results = run_evaluation(_samples(tmp_path), tr, Settings(bot_token="t"), repeats=2, convert=fake_convert)
    assert tr.calls == 1 + 2 * 2  # прогрев + повторы
    assert results[0].wer == 0.0 and results[0].ok
    assert results[1].wer_d == 1 and math.isclose(results[1].wer, 0.2)
    assert math.isclose(results[0].audio_sec, 2.0)
    assert results[0].rtf >= 0

    rows = {r["group"]: r for r in summarize(results)}
    assert set(rows) == {"quiet", "noisy", "ALL"}
    assert rows["ALL"]["n"] == 2
    assert math.isclose(rows["ALL"]["wer"], 1 / 9)  # 1 ошибка на 4 + 5 слов
    assert rows["ALL"]["dropped_segments"] == 2


def test_no_warmup(tmp_path):
    tr = FakeTranscriber()
    run_evaluation(_samples(tmp_path), tr, Settings(bot_token="t"), warmup=False, convert=fake_convert)
    assert tr.calls == 2


def test_failing_transcriber(tmp_path):
    results = run_evaluation(
        _samples(tmp_path), FakeTranscriber(fail=True), Settings(bot_token="t"),
        warmup=False, convert=fake_convert,
    )
    r = results[0]
    assert not r.ok and r.wer == 1.0 and "модель сломалась" in r.error
    assert math.isnan(r.rtf)
    row = summarize(results)[-1]
    assert row["errors"] == 2 and row["ok"] == 0
    assert math.isnan(row["stt_mean_ms"])


def test_csv_output(tmp_path):
    results = run_evaluation(_samples(tmp_path), FakeTranscriber(), Settings(bot_token="t"),
                             warmup=False, convert=fake_convert)
    p1, p2 = tmp_path / "s.csv", tmp_path / "sum.csv"
    write_samples_csv(p1, results)
    write_summary_csv(p2, summarize(results), style="plain")
    assert p1.read_bytes().startswith(b"\xef\xbb\xbf")
    with open(p1, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert rows[0]["reference"] == "Привет, мир сегодня хорошо"
    assert rows[1]["wer"] == "0,2"
    with open(p2, encoding="utf-8-sig", newline="") as f:
        summary = list(csv.DictReader(f))
    assert summary[-1]["group"] == "ALL" and summary[-1]["n"] == "2"
