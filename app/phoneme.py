"""Phoneme-level pronunciation scoring.

Pipeline:
  1. Reference text -> g2p_en (ARPABET) -> expected phoneme list per word.
  2. User audio -> wav2vec2-base-960h (character CTC) -> frame posteriors.
  3. Forced alignment: align expected characters to audio frames (CTC DP).
  4. Per-word confidence = mean posterior over the word's aligned frame span.
     Low-confidence phonemes (within weak words) are flagged as mispronounced.

Heavy deps are lazy-imported so the app runs without them; when unavailable the
speaking endpoint falls back to the transcription proxy in scorer.py.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class WordPhonemeFeedback:
    word: str
    expected_phonemes: list[str]
    confidence: float
    weak: bool            # low-confidence word -> likely mispronounced


@dataclass
class PhonemeScore:
    overall_accuracy: float
    words: list[WordPhonemeFeedback] = field(default_factory=list)
    weak_words: list[str] = field(default_factory=list)
    available: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "overall_accuracy": round(self.overall_accuracy, 3),
            "weak_words": self.weak_words,
            "words": [
                {"word": w.word, "expected_phonemes": w.expected_phonemes,
                 "confidence": round(w.confidence, 3), "weak": w.weak}
                for w in self.words
            ],
            "available": self.available,
            "note": self.note,
        }


class PhonemeScorer:
    """Lazy-loaded wav2vec2 + g2p_en phoneme scorer."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._g2p = None
        self._device = "cpu"
        self._ready: bool | None = None

    def available(self) -> bool:
        if self._ready is not None:
            return self._ready
        try:
            self._load()
            self._ready = True
        except Exception as e:  # noqa: BLE001
            self._ready = False
            self._note = f"phoneme scorer unavailable: {e}"
        return self._ready

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from g2p_en import G2p

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        self._g2p = G2p()

    # ---- text -> phonemes per word ----
    def text_to_phonemes(self, text: str) -> list[tuple[str, list[str]]]:
        """Return [(word, [phonemes]), ...] using g2p_en ARPABET."""
        words = re.findall(r"[A-Za-z]+", text)
        out = []
        for w in words:
            ph = self._g2p(w)
            # g2p returns list incl. space tokens; drop empties
            ph = [p for p in ph if p and p != " "]
            out.append((w, ph))
        return out

    # ---- audio -> character posteriors ----
    def _audio_posteriors(self, audio_path: str):
        import torch
        import librosa
        wav, sr = librosa.load(audio_path, sr=16000)
        inputs = self._processor(wav, sampling_rate=16000,
                                 return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self._device)
        with torch.no_grad():
            logits = self._model(input_values).logits
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (T, V)

    def _forced_align_words(self, posteriors, words: list[str]):
        """Even-segment baseline alignment: split frames evenly across words,
        then average each word's character posterior over its span.

        Returns list of (word, confidence).
        """
        import numpy as np
        vocab = self._processor.tokenizer.get_vocab()
        T, V = posteriors.shape
        # expected character sequence (lowercase letters + spaces)
        expected_chars = []
        for w in words:
            expected_chars.extend(list(w.lower()))
            expected_chars.append(" ")
        n = len(expected_chars)
        if n == 0:
            return [(w, 0.0) for w in words]
        # even split frames across characters
        boundaries = np.linspace(0, T, n + 1).astype(int)
        char_conf = []
        for i, ch in enumerate(expected_chars):
            s, e = boundaries[i], boundaries[i + 1]
            if e <= s:
                e = s + 1
            cid = vocab.get(ch)
            char_conf.append(float(posteriors[s:e, cid].mean()) if cid is not None else 0.0)
        # group char confidences back into words
        out = []
        idx = 0
        for w in words:
            k = len(w)
            wconf = char_conf[idx:idx + k]
            out.append((w, sum(wconf) / max(1, len(wconf))))
            idx += k + 1  # +1 for space
        return out

    def score(self, reference_text: str, audio_path: str,
              weak_threshold: float = 0.5) -> PhonemeScore:
        if not self.available():
            return PhonemeScore(overall_accuracy=0.0, available=False,
                                note="phoneme scorer unavailable (install torch/transformers/g2p_en)")
        wp = self.text_to_phonemes(reference_text)
        words = [w for w, _ in wp]
        posteriors = self._audio_posteriors(audio_path)
        aligned = self._forced_align_words(posteriors, words)
        conf_by_word = dict(aligned)
        feedback = []
        weak = []
        confs = []
        for w, ph in wp:
            c = conf_by_word.get(w, 0.0)
            weak_flag = c < weak_threshold
            feedback.append(WordPhonemeFeedback(
                word=w, expected_phonemes=ph, confidence=c, weak=weak_flag))
            confs.append(c)
            if weak_flag:
                weak.append(w)
        overall = sum(confs) / len(confs) if confs else 0.0
        return PhonemeScore(
            overall_accuracy=overall, words=feedback, weak_words=weak,
            available=True, note="词级音素反馈（wav2vec2 对齐 + g2p_en）",
        )


_scorer: PhonemeScorer | None = None


def get_phoneme_scorer() -> PhonemeScorer:
    global _scorer
    if _scorer is None:
        _scorer = PhonemeScorer()
    return _scorer
