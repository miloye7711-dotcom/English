"""Closed-loop state machine: DIAGNOSE -> TRAIN -> EVALUATE -> (UP/DOWN/STAY)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import leveling
from .scorer import score_listening, score_speaking

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "sentences.json")


@dataclass
class Item:
    item_id: str
    level: str
    task: str          # "listening" | "speaking"
    text: str          # reference sentence
    audio: str = ""    # optional audio path/url


@dataclass
class Session:
    level: str = "A2"
    phase: str = "DIAGNOSE"     # DIAGNOSE / TRAIN / EVALUATE
    weakness: str = "balanced"
    practice_count: int = 0
    # transient buffers
    diag_scores: dict = field(default_factory=dict)   # {level: {"listen": [], "speak": []}}
    eval_listen: list = field(default_factory=list)
    eval_speak: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class Trainer:
    def __init__(self) -> None:
        with open(DATA_FILE, encoding="utf-8") as f:
            self.bank = json.load(f)            # {"A2": [...], "B1": [...], ...}
        self.s = Session()
        self._item_cursor = 0

    # ---------- item selection ----------
    def _items(self, level: str, task: str, n: int, offset: int = 0) -> list[Item]:
        rows = self.bank.get(level, [])
        out: list[Item] = []
        for k, r in enumerate(rows):
            if r.get("task") != task:
                continue
            out.append(Item(item_id=f"{level}-{task}-{k}", level=level,
                            task=task, text=r["text"], audio=r.get("audio", "")))
        return out[offset:offset + n]

    def next_item(self) -> Optional[Item]:
        """Return the next task item depending on phase."""
        s = self.s
        if s.phase == "DIAGNOSE":
            # adaptive: test listening at current level, then speaking
            listen = self._items(s.level, "listening", 3)
            speak = self._items(s.level, "speaking", 3)
            batch = listen + speak
            # pick by round-robin cursor
            if self._item_cursor < len(batch):
                it = batch[self._item_cursor]
                self._item_cursor += 1
                return it
            return None  # batch done -> call finalize_diagnose
        if s.phase == "TRAIN":
            # alternate listening/speaking, favoring the weak side
            task = "speaking" if s.weakness == "speaking" else "listening"
            items = self._items(s.level, task, 1)
            if not items:
                items = self._items(s.level, "speaking", 1)
            return items[0] if items else None
        if s.phase == "EVALUATE":
            items = self._items(s.level, "listening", 3) + self._items(s.level, "speaking", 3)
            if self._item_cursor < len(items):
                it = items[self._item_cursor]
                self._item_cursor += 1
                return it
            return None
        return None

    # ---------- answer handling ----------
    def submit(self, item: Item, listening_text: str = "",
               speaking_transcript: str = "", speaking_confidence: float = 1.0) -> dict:
        s = self.s
        feedback: dict = {"item": item.item_id, "task": item.task}

        if item.task == "listening":
            res = score_listening(item.text, listening_text)
            feedback["score"] = res.as_dict()
            self._record(s, "listen", res.accuracy)
        else:
            res = score_speaking(item.text, speaking_transcript, speaking_confidence)
            feedback["score"] = res.as_dict()
            self._record(s, "speak", res.accuracy)

        # phase bookkeeping
        if s.phase == "TRAIN":
            s.practice_count += 1
            if s.practice_count >= leveling.PRACTICE_UNITS_PER_LEVEL:
                s.phase = "EVALUATE"
                self._item_cursor = 0
                feedback["next_phase"] = "EVALUATE"

        if s.phase == "DIAGNOSE" and self._item_cursor == len(
                self._diag_batch()):
            self.finalize_diagnose()
        if s.phase == "EVALUATE" and self._item_cursor == len(self._eval_batch()):
            self.finalize_evaluate()

        feedback["state"] = s.as_dict()
        return feedback

    def _diag_batch(self) -> list:
        s = self.s
        return self._items(s.level, "listening", 3) + self._items(s.level, "speaking", 3)

    def _eval_batch(self) -> list:
        s = self.s
        return self._items(s.level, "listening", 3) + self._items(s.level, "speaking", 3)

    def _record(self, s: Session, kind: str, acc: float) -> None:
        if s.phase == "DIAGNOSE":
            bucket = s.diag_scores.setdefault(s.level, {"listen": [], "speak": []})
            bucket[kind].append(acc)
        elif s.phase == "EVALUATE":
            if kind == "listen":
                s.eval_listen.append(acc)
            else:
                s.eval_speak.append(acc)

    def finalize_diagnose(self) -> dict:
        s = self.s
        # average listening/speaking at the tested level
        bucket = s.diag_scores.get(s.level, {"listen": [], "speak": []})
        lv = sum(bucket["listen"]) / max(1, len(bucket["listen"]))
        sp = sum(bucket["speak"]) / max(1, len(bucket["speak"]))
        s.weakness = leveling.weakness(lv, sp)
        s.phase = "TRAIN"
        s.practice_count = 0
        self._item_cursor = 0
        s.history.append({"event": "diagnose", "level": s.level,
                          "listening": round(lv, 3), "speaking": round(sp, 3)})
        return {"placed_level": s.level, "weakness": s.weakness,
                "listening": round(lv, 3), "speaking": round(sp, 3)}

    def finalize_evaluate(self) -> dict:
        s = self.s
        lv = sum(s.eval_listen) / max(1, len(s.eval_listen))
        sp = sum(s.eval_speak) / max(1, len(s.eval_speak))
        decision = leveling.decide(lv, sp)
        result = {"listening": round(lv, 3), "speaking": round(sp, 3),
                  "action": decision.action, "reason": decision.reason}
        if decision.action == "UP":
            s.level = leveling.next_level(s.level)
            s.phase = "DIAGNOSE"
            self._item_cursor = 0
        elif decision.action == "DOWN":
            s.level = leveling.prev_level(s.level)
            s.phase = "TRAIN"
        else:
            s.phase = "TRAIN"
        s.practice_count = 0
        s.eval_listen.clear()
        s.eval_speak.clear()
        s.diag_scores.clear()
        s.history.append({"event": "evaluate", "level": s.level, **result})
        return result

    def reset(self) -> None:
        self.s = Session()
        self._item_cursor = 0
