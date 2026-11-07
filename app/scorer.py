"""Word-level scoring: listening dictation (WER) and speaking (transcription proxy)."""
from __future__ import annotations

import re
from dataclasses import dataclass


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]


def _levenshtein_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via word-level edit distance."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = cost to align ref[:i] -> hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # backtrack to count ops
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: (S+D+I)/N. 0 = perfect."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    s, d, i = _levenshtein_ops(ref, hyp)
    return (s + d + i) / len(ref)


@dataclass
class ScoreResult:
    accuracy: float          # 0..1, higher is better
    wer: float              # 0..1, lower is better
    wrong_words: list[str]   # reference words not produced correctly
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 3),
            "wer": round(self.wer, 3),
            "wrong_words": self.wrong_words,
            "note": self.note,
        }


def score_listening(reference: str, hypothesis: str) -> ScoreResult:
    """Dictation: how many words the user typed correctly."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    w = wer(reference, hypothesis)
    # identify wrong words: words in ref missing/wrong in hyp
    from collections import Counter
    ref_c, hyp_c = Counter(ref), Counter(hyp)
    missing = list((ref_c - hyp_c).elements())
    return ScoreResult(
        accuracy=max(0.0, 1.0 - w),
        wer=w,
        wrong_words=missing,
        note="听写：对照原文逐词核对",
    )


def score_speaking(reference: str, transcript: str, avg_confidence: float = 1.0) -> ScoreResult:
    """Speaking (MVP proxy): compare ASR transcript to reference.

    accuracy = blend(word_transcription_accuracy, avg_word_confidence).
    This is a proxy; the production upgrade path is wav2vec2 forced alignment +
    phoneme-level posterior (see design doc).
    """
    ref = normalize(reference)
    hyp = normalize(transcript)
    w = wer(reference, transcript)
    word_acc = max(0.0, 1.0 - w)
    conf = max(0.0, min(1.0, avg_confidence))
    accuracy = 0.7 * word_acc + 0.3 * conf
    from collections import Counter
    ref_c, hyp_c = Counter(ref), Counter(hyp)
    missing = list((ref_c - hyp_c).elements())
    return ScoreResult(
        accuracy=accuracy,
        wer=w,
        wrong_words=missing,
        note="跟读：ASR转写与原文对比（近似发音分）",
    )
