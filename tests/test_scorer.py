"""Unit tests for scorer and leveling (no audio / no external deps needed)."""
from app.scorer import wer, score_listening, score_speaking
from app.leveling import decide, weakness, next_level, prev_level


def test_wer_perfect():
    assert wer("I drink coffee every morning", "i drink coffee every morning") == 0.0


def test_wer_missing_word():
    w = wer("I drink coffee every morning", "I drink every morning")
    assert abs(w - 1 / 5) < 1e-6


def test_listening_accuracy():
    r = score_listening("She goes to the park", "she goes to park")
    assert r.accuracy < 1.0 and "the" in r.wrong_words


def test_speaking_blend():
    r = score_speaking("What time does the train arrive", "what time does train arrive",
                       avg_confidence=0.9)
    assert 0.0 < r.accuracy < 1.0


def test_decide_up():
    d = decide(0.9, 0.85)
    assert d.action == "UP"


def test_decide_down():
    d = decide(0.5, 0.7)
    assert d.action == "DOWN"


def test_decide_stay():
    d = decide(0.75, 0.7)
    assert d.action == "STAY"


def test_level_order():
    assert next_level("A2") == "B1"
    assert prev_level("B1") == "A2"


def test_weakness():
    assert weakness(0.8, 0.6) == "speaking"
    assert weakness(0.6, 0.8) == "listening"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("ALL TESTS PASSED")
