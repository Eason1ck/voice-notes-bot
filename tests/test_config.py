import os

from bot.config import Settings


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {})
    s = Settings.from_env(tmp_path / "нет_такого.env")
    assert s.whisper_model == "small"
    assert s.whisper_device == "cpu"
    assert s.max_duration_sec == 180
    assert s.max_file_bytes == 20 * 1024 * 1024
    assert s.vad_filter is True


def test_values_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {})
    env = tmp_path / ".env"
    env.write_text(
        "# комментарий\n"
        "BOT_TOKEN=123:ABC\n"
        'WHISPER_MODEL="medium"\n'
        "MAX_DURATION_SEC=60\n"
        "VAD_FILTER=false\n"
        "NO_SPEECH_THRESHOLD=0.5\n",
        encoding="utf-8",
    )
    s = Settings.from_env(env)
    assert s.bot_token == "123:ABC"
    assert s.whisper_model == "medium"
    assert s.max_duration_sec == 60
    assert s.vad_filter is False
    assert s.no_speech_threshold == 0.5


def test_ffmpeg_path_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {})
    env = tmp_path / ".env"
    env.write_text("FFMPEG_PATH=C:\\ffmpeg\\bin\\ffmpeg.exe\n", encoding="utf-8")
    assert Settings.from_env(env).ffmpeg_path == "C:\\ffmpeg\\bin\\ffmpeg.exe"
    monkeypatch.setattr(os, "environ", {})
    assert Settings.from_env(tmp_path / "нет.env").ffmpeg_path == "ffmpeg"


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {"WHISPER_MODEL": "tiny"})
    env = tmp_path / ".env"
    env.write_text("WHISPER_MODEL=large-v3\n", encoding="utf-8")
    assert Settings.from_env(env).whisper_model == "tiny"
