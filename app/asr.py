"""ASR layer backed by faster-whisper (CTranslate2).

The model is lazy-loaded so the app still runs when the heavy dependency is
missing; the speaking endpoint then accepts a transcript typed directly.
Set environment variable WHISPER_MODEL to choose size (tiny/base/small/...).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

# On Windows, pip-installed CUDA runtime DLLs (cuBLAS/NVRTC) live under the
# nvidia/ */bin directories; add them to the DLL search path before loading
# ctranslate2 so the CUDA device can be initialized.
# Default HF mirror (China); override by setting HF_ENDPOINT yourself.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _register_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    try:
        import site
        dirs = []
        for sp in site.getsitepackages():
            for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
                if os.path.isdir(d):
                    dirs.append(d)
        for d in dirs:
            os.add_dll_directory(d)
        # also prepend to PATH so the loader finds dependent CUDA libs
        existing = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(dirs + [existing])
    except Exception:
        pass


_register_cuda_dll_dirs()


@dataclass
class Transcription:
    text: str
    avg_confidence: float = 1.0


class MockASR:
    """Used in tests / when no model is installed."""

    def available(self) -> bool:
        return False


class WhisperASR:
    """faster-whisper wrapper. Lazy import / load so the app starts without deps."""

    def __init__(self, model_size: str | None = None):
        self.model = None
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "base")
        self._device = "cpu"

    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def _load(self):
        if self.model is not None:
            return self.model
        from faster_whisper import WhisperModel  # noqa: WPS433
        try:
            self.model = WhisperModel(self.model_size, device="cuda",
                                      compute_type="float16")
            self._device = "cuda"
        except Exception:
            # fall back to CPU int8 if CUDA unavailable / broken
            self.model = WhisperModel(self.model_size, device="cpu",
                                      compute_type="int8")
            self._device = "cpu"
        return self.model

    def transcribe(self, audio_path: str) -> Transcription:
        model = self._load()
        segments, _info = model.transcribe(
            audio_path, language="en", word_timestamps=True,
            vad_filter=True,
        )
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
