import asyncio
import dataclasses
import time
from types import SimpleNamespace

import pytest

from bot import pipeline
from bot.config import Settings
from bot.errors import (
    AudioConversionError,
    DownloadError,
    NoSpeechError,
    ProcessingTimeoutError,
    TooLargeError,
    TooLongError,
    TranscriptionError,
)
from bot.stt import TranscriptionResult


class FakeBot:
    def __init__(self, fail=False):
        self.fail = fail
        self.downloads = 0

    async def download(self, file_id, destination):
        self.downloads += 1
        if self.fail:
            raise RuntimeError("Bad Request: file is too big")
        destination.write_bytes(b"OggS")


class FakeTranscriber:
    def __init__(self, text="Нужно подготовить отчёт по курсовой", error=None, delay=0.0):
        self.text, self.error, self.delay = text, error, delay

    def transcribe(self, path):
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return TranscriptionResult(text=self.text, language="ru")


@pytest.fixture(autouse=True)
def fake_ffmpeg(monkeypatch):
    async def fake_convert(src, dst, *, normalize=False, retries=1, ffmpeg="ffmpeg"):
        dst.write_bytes(b"RIFF")
        return dst

    monkeypatch.setattr(pipeline, "convert_to_wav", fake_convert)


def run(tmp_path, *, voice=None, bot=None, transcriber=None, settings=None):
    voice = voice or SimpleNamespace(file_id="abc", duration=10, file_size=50_000)
    return asyncio.run(
        pipeline.process_voice(
            bot=bot or FakeBot(),
            voice=voice,
            settings=settings or Settings(),
            transcriber=transcriber or FakeTranscriber(),
            semaphore=asyncio.Semaphore(1),
            workdir=tmp_path,
        )
    )


def test_success(tmp_path):
    result = run(tmp_path)
    assert result.text == "Нужно подготовить отчёт по курсовой"


def test_too_long_is_rejected_before_download(tmp_path):
    bot = FakeBot()
    voice = SimpleNamespace(file_id="abc", duration=400, file_size=50_000)
    with pytest.raises(TooLongError):
        run(tmp_path, voice=voice, bot=bot)
    assert bot.downloads == 0


def test_too_large_is_rejected(tmp_path):
    voice = SimpleNamespace(file_id="abc", duration=10, file_size=30 * 1024 * 1024)
    with pytest.raises(TooLargeError):
        run(tmp_path, voice=voice)


def test_download_failure(tmp_path):
    with pytest.raises(DownloadError) as info:
        run(tmp_path, bot=FakeBot(fail=True))
    assert "file is too big" in info.value.details


def test_conversion_error_is_propagated(tmp_path, monkeypatch):
    async def broken(*args, **kwargs):
        raise AudioConversionError(details="boom")

    monkeypatch.setattr(pipeline, "convert_to_wav", broken)
    with pytest.raises(AudioConversionError):
        run(tmp_path)


def test_model_failure_becomes_transcription_error(tmp_path):
    with pytest.raises(TranscriptionError):
        run(tmp_path, transcriber=FakeTranscriber(error=RuntimeError("CUDA out of memory")))


def test_timeout(tmp_path):
    settings = dataclasses.replace(Settings(), process_timeout_sec=0)
    with pytest.raises(ProcessingTimeoutError):
        run(tmp_path, transcriber=FakeTranscriber(delay=0.3), settings=settings)


@pytest.mark.parametrize("text", ["", "   ", "Да", "Спасибо за просмотр. " * 30])
def test_empty_short_or_repetitive_text_is_no_speech(tmp_path, text):
    with pytest.raises(NoSpeechError):
        run(tmp_path, transcriber=FakeTranscriber(text=text))


def test_errors_have_friendly_messages(tmp_path):
    with pytest.raises(NoSpeechError) as info:
        run(tmp_path, transcriber=FakeTranscriber(text=""))
    assert "Речь не распознана" in info.value.user_message
