import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import eval.run_eval as run_eval

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")


class StubTranscriber:
    def __init__(self, settings, model=None):
        pass

    def load(self):
        pass

    def transcribe(self, path):
        return SimpleNamespace(text="привет мир", dropped=0)


def _make_ogg(path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "libopus", str(path)],
        check=True,
    )


def test_cli_end_to_end(tmp_path, monkeypatch):
    ds = tmp_path / "ds" / "quiet"
    ds.mkdir(parents=True)
    _make_ogg(ds / "01.ogg")
    (ds / "01.txt").write_text("Привет, мир.", encoding="utf-8")
    monkeypatch.setattr(run_eval, "WhisperTranscriber", StubTranscriber)
    monkeypatch.setenv("BOT_TOKEN", "t")
    out = tmp_path / "out"
    code = run_eval.main(["--dataset", str(tmp_path / "ds"), "--output", str(out),
                          "--no-warmup", "--model", "tiny", "--label", "t"])
    assert code == 0
    names = sorted(p.name for p in out.iterdir())
    assert any(n.endswith("_tiny_t_samples.csv") for n in names)
    assert any(n.endswith("_tiny_t_summary.csv") for n in names)
    meta = json.loads(next(out.glob("*_meta.json")).read_text(encoding="utf-8"))
    assert meta["model"] == "tiny" and meta["samples"] == 1


def test_cli_empty_dataset(tmp_path, capsys):
    (tmp_path / "ds").mkdir()
    assert run_eval.main(["--dataset", str(tmp_path / "ds")]) == 2
    assert run_eval.main(["--dataset", str(tmp_path / "missing")]) == 2
