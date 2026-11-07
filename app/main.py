"""FastAPI app: serves the web UI and the closed-loop training API."""
from __future__ import annotations

import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .state_machine import Trainer, Item

app = FastAPI(title="English Adaptive Trainer", version="0.1.0")
trainer = Trainer()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/state")
def state():
    return trainer.s.as_dict()


@app.get("/api/next")
def next_item():
    it = trainer.next_item()
    if it is None:
        # batch finished: run finalizer depending on phase
        if trainer.s.phase == "DIAGNOSE":
            res = trainer.finalize_diagnose()
            return JSONResponse({"batch_done": True, "diagnose": res,
                                 "state": trainer.s.as_dict()})
        if trainer.s.phase == "EVALUATE":
            res = trainer.finalize_evaluate()
            return JSONResponse({"batch_done": True, "evaluate": res,
                                 "state": trainer.s.as_dict()})
    return {"item": it.__dict__ if it else None, "state": trainer.s.as_dict()}


@app.post("/api/answer")
async def answer(
    item_id: str = Form(...),
    task: str = Form(...),
    listening_text: str = Form(""),
    speaking_transcript: str = Form(""),
    audio: UploadFile | None = File(None),
):
    # reconstruct the item
    it = _find_item(item_id, task)
    conf = 1.0
    audio_path = None
    if task == "speaking" and audio is not None:
        audio_path = f"_tmp_{audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        # ASR transcription if no transcript provided
        if not speaking_transcript:
            from .asr import get_asr
            asr = get_asr()
            if getattr(asr, "available", lambda: False)():
                tr = asr.transcribe(audio_path)
                speaking_transcript = tr.text
                conf = tr.avg_confidence
    result = trainer.submit(it, listening_text=listening_text,
                            speaking_transcript=speaking_transcript,
                            speaking_confidence=conf)
    # phoneme-level feedback when audio was provided and scorer is available
    if audio_path and task == "speaking":
        try:
            from .phoneme import get_phoneme_scorer
            ps = get_phoneme_scorer()
            if ps.available():
                result["phoneme"] = ps.score(it.text, audio_path).as_dict()
        except Exception as e:  # noqa: BLE001
            result["phoneme"] = {"available": False, "note": str(e)}
        os.remove(audio_path)
    return result


@app.post("/api/reset")
def reset():
    trainer.reset()
    return {"state": trainer.s.as_dict()}


def _find_item(item_id: str, task: str) -> Item:
    # item_id like "B1-listening-2"
    level, task_part, idx = item_id.split("-")
    rows = trainer.bank[level]
    k = 0
    for r in rows:
        if r["task"] == task:
            if k == int(idx):
                return Item(item_id=item_id, level=level, task=task,
                            text=r["text"])
            k += 1
    raise HTTPException(404, "item not found")


from fastapi import HTTPException  # noqa: E402

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
