import asyncio
import shutil
import subprocess
import wave

import pytest

from bot import audio
from bot.errors import AudioConversionError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")


def _make_sine(path, codec_args):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        *codec_args, str(path),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


@needs_ffmpeg
def test_convert_ogg_to_wav_16k_mono(tmp_path):
    src = tmp_path / "in.ogg"
    if not _make_sine(src, ["-c:a", "libopus"]):
        pytest.skip("в этой сборке ffmpeg нет кодека libopus")
    dst = tmp_path / "out.wav"

    asyncio.run(audio.convert_to_wav(src, dst))

    with wave.open(str(dst)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() > 0


@needs_ffmpeg
def test_convert_with_normalize(tmp_path):
    src = tmp_path / "in.wav"
    assert _make_sine(src, [])
    dst = tmp_path / "out.wav"
    asyncio.run(audio.convert_to_wav(src, dst, normalize=True))
    assert dst.stat().st_size > 0


@needs_ffmpeg
def test_corrupt_file_raises(tmp_path):
    src = tmp_path / "bad.ogg"
    src.write_bytes("это не аудио".encode("utf-8"))
    with pytest.raises(AudioConversionError):
        asyncio.run(audio.convert_to_wav(src, tmp_path / "out.wav", retries=0))


def test_ffmpeg_missing_raises_without_retry(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def fake_exec(*args, **kwargs):
        calls["n"] += 1
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(AudioConversionError) as info:
        asyncio.run(audio.convert_to_wav(tmp_path / "a.ogg", tmp_path / "a.wav", retries=3))
    assert "не найден" in info.value.details
    assert calls["n"] == 1  # повторы не имеют смысла


def test_custom_ffmpeg_path_is_used(tmp_path, monkeypatch):
    used = {}

    async def fake_exec(*args, **kwargs):
        used["program"] = args[0]
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(AudioConversionError) as info:
        asyncio.run(
            audio.convert_to_wav(
                tmp_path / "a.ogg", tmp_path / "a.wav", retries=0, ffmpeg=r"C:\ffmpeg\bin\ffmpeg.exe"
            )
        )
    assert used["program"] == r"C:\ffmpeg\bin\ffmpeg.exe"
    assert r"C:\ffmpeg\bin\ffmpeg.exe" in info.value.details


def test_retry_after_failure(tmp_path, monkeypatch):
    attempts = {"n": 0}

    async def fake_run(src, dst, normalize, ffmpeg="ffmpeg"):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise AudioConversionError(details="временный сбой")
        dst.write_bytes(b"RIFF")

    monkeypatch.setattr(audio, "_run_ffmpeg", fake_run)
    out = asyncio.run(audio.convert_to_wav(tmp_path / "a.ogg", tmp_path / "a.wav", retries=1))
    assert out.exists()
    assert attempts["n"] == 2
