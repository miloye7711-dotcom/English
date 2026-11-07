"""CEFR leveling and upgrade/downgrade decisions."""
from __future__ import annotations

from dataclasses import dataclass

# Ordered CEFR levels, low -> high
LEVELS = ["A1", "A2", "B1", "B2", "C1"]

# Thresholds (tuneable; calibrate with speechocean762 in production)
LISTENING_UP = 0.85     # >= this => listening strong
LISTENING_DOWN = 0.65    # < this => listening weak
SPEAKING_UP = 0.80
SPEAKING_DOWN = 0.60

PRACTICE_UNITS_PER_LEVEL = 15   # train units before an EVALUATE triggers


@dataclass
class LevelDecision:
    action: str          # "UP" | "DOWN" | "STAY"
    listening: float
    speaking: float
    reason: str


def next_level(level: str) -> str:
    i = LEVELS.index(level)
    return LEVELS[min(i + 1, len(LEVELS) - 1)]


def prev_level(level: str) -> str:
    i = LEVELS.index(level)
    return LEVELS[max(i - 1, 0)]


def decide(listening: float, speaking: float) -> LevelDecision:
    """Given averaged scores, decide whether to level up / down / stay."""
    if listening >= LISTENING_UP and speaking >= SPEAKING_UP:
        return LevelDecision(
            "UP", listening, speaking,
            f"听力{listening:.0%}≥{LISTENING_UP:.0%} 且 口语{speaking:.0%}≥{SPEAKING_UP:.0%}，升级")
    if listening < LISTENING_DOWN or speaking < SPEAKING_DOWN:
        return LevelDecision(
            "DOWN", listening, speaking,
            f"听力{listening:.0%} 或 口语{speaking:.0%} 低于降级线，回炉巩固")
    return LevelDecision(
        "STAY", listening, speaking,
        f"听力{listening:.0%} 口语{speaking:.0%}，继续当前等级训练")


def diagnose_place(level_listening: dict[str, float]) -> str:
    """Adaptive placement: given {level: avg_listening_score}, pick the highest
    level where listening >= 0.80 (comprehension reached)."""
    placed = "A1"
    for lv in LEVELS:
        if level_listening.get(lv, 0.0) >= 0.80:
            placed = lv
    return placed


def weakness(listening: float, speaking: float) -> str:
    if listening - speaking > 0.05:
        return "speaking"
    if speaking - listening > 0.05:
        return "listening"
    return "balanced"
