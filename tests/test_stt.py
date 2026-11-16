from types import SimpleNamespace

from bot.config import Settings
from bot.stt import WhisperTranscriber


def seg(text, start=0.0, end=1.0, no_speech=0.0, logprob=-0.3, ratio=1.2):
    return SimpleNamespace(
        start=start, end=end, text=text,
        no_speech_prob=no_speech, avg_logprob=logprob, compression_ratio=ratio,
    )


class FakeModel:
    def __init__(self, segments, language="ru", duration=5.0):
        self._segments = segments
        self._info = SimpleNamespace(language=language, language_probability=0.99, duration=duration)
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return iter(self._segments), self._info


def make(segments, **settings_kwargs):
    model = FakeModel(segments)
    return WhisperTranscriber(Settings(**settings_kwargs), model=model), model


def test_joins_segments_in_order():
    tr, _ = make([seg(" Привет. ", 0, 1), seg("Это тест.", 1, 2)])
    result = tr.transcribe("any.wav")
    assert result.text == "Привет. Это тест."
    assert [s.text for s in result.segments] == ["Привет.", "Это тест."]
    assert result.language == "ru"
    assert result.duration_sec == 5.0
    assert result.dropped == 0


def test_passes_language_vad_and_beam_to_model():
    tr, model = make([seg("текст")], beam_size=3, vad_filter=False)
    tr.transcribe("any.wav")
    assert model.kwargs == {"language": "ru", "vad_filter": False, "beam_size": 3}


def test_drops_segment_without_speech():
    tr, _ = make([seg("Нормальная фраза."), seg("Субтитры сделал DimaTorzok", no_speech=0.9, logprob=-2.0)])
    result = tr.transcribe("any.wav")
    assert result.text == "Нормальная фраза."
    assert result.dropped == 1


def test_keeps_low_logprob_if_speech_is_likely():
    tr, _ = make([seg("Тихая, но настоящая речь.", no_speech=0.1, logprob=-1.5)])
    assert tr.transcribe("any.wav").text == "Тихая, но настоящая речь."


def test_drops_repetitive_segment():
    tr, _ = make([seg("Первая фраза."), seg("да да да да да да", ratio=3.5)])
    result = tr.transcribe("any.wav")
    assert result.text == "Первая фраза."
    assert result.dropped == 1


def test_empty_segments_are_skipped_and_result_can_be_empty():
    tr, _ = make([seg("   "), seg("")])
    result = tr.transcribe("any.wav")
    assert result.text == ""
    assert result.segments == []
    assert result.dropped == 0


def test_thresholds_come_from_settings():
    tr, _ = make([seg("Фраза.", no_speech=0.5, logprob=-1.2)], no_speech_threshold=0.4)
    assert tr.transcribe("any.wav").dropped == 1
