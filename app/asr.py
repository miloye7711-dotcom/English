"""ASR layer. Whisper integration is optional; the speaking endpoint accepts a
transcript directly when Whisper is unavailable (used for testing / demo)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Transcription:
    text: str
    avg_confidence: float = 1.0


class MockASR:
    """Used in tests / when no model is installed. Echoes nothing."""

    def available(self) -> bool:
        return False


class WhisperASR:
    """faster-whisper wrapper. Lazy import so the app runs without the heavy dep."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size

    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def _load(self):
        if self.model is None:
            from faster_whisper import WhisperModel  # noqa: WPS433
            try:
                self.model = WhisperModel(self.model_size, device="cuda",
                                          compute_type="float16")
            except Exception:
                self.model = WhisperModel(self.model_size, device="cpu")
        return self.model

    def transcribe(self, audio_path: str) -> Transcription:
        model = self._load()
        segments, _info = model.transcribe(audio_path, word_timestamps=True)
        words_text: list[str] = []
        confs: list[float] = []
        for seg in segments:
            for w in (seg.words or []):
                words_text.append(w.word)
                confs.append(getattr(w, "probability", 1.0))
        text = "".join(words_text).strip()
        avg_conf = sum(confs) / len(confs) if confs else 1.0
        return Transcription(text=text, avg_confidence=avg_conf)


_asr: object = None


def get_asr() -> object:
    global _asr
    if _asr is None:
        try:
            _asr = WhisperASR()
            if not _asr.available():
                _asr = MockASR()
        except Exception:
            _asr = MockASR()
    return _asr
