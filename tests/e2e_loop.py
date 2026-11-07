"""End-to-end driver: DIAGNOSE -> TRAIN(15) -> EVALUATE -> UP, then a DOWN scenario."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.state_machine import Trainer

def run_phase(t, perfect=True):
    """Answer items until the current phase finalizes. Returns the phase's finalize result."""
    start_phase = t.s.phase
    while True:
        it = t.next_item()
        if it is None:
            if start_phase == "DIAGNOSE":
                return t.finalize_diagnose()
            if start_phase == "EVALUATE":
                return t.finalize_evaluate()
        if perfect:
            kw = {"listening_text": it.text, "speaking_transcript": it.text}
        else:
            kw = {"listening_text": "garbage words here", "speaking_transcript": "blah"}
        t.submit(it, **kw)
        if t.s.phase != start_phase:
            # phase changed (finalize happened inside submit) -> done
            return None

# Scenario A: perfect answers
t = Trainer()
print("== A: PERFECT answers ==")
print("start:", t.s.phase, t.s.level)
r = run_phase(t); print("DIAGNOSE ->", r)
# TRAIN until EVALUATE
n = 0
while t.s.phase == "TRAIN" and n < 40:
    it = t.next_item()
    if it is None: break
    t.submit(it, listening_text=it.text, speaking_transcript=it.text)
    n += 1
print(f"TRAIN done after {n} items -> phase:", t.s.phase, "practice_count=", t.s.practice_count)
r = run_phase(t); print("EVALUATE ->", r)
print("final:", t.s.phase, "level=", t.s.level)

# Scenario B: reset to B1, EVALUATE with bad answers -> DOWN
print("\n== B: BAD answers at B1 ==")
t.reset()
t.s.level = "B1"; t.s.phase = "EVALUATE"; t._item_cursor = 0
r = run_phase(t, perfect=False); print("EVALUATE(bad) ->", r)
print("final:", t.s.phase, "level=", t.s.level)

print("\nHISTORY:")
for h in t.s.history:
    print(" ", h)
print("\nE2E DONE")
