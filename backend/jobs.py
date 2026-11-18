"""Tiny in-process job runner for long experiments (training, screening, ingestion)."""
from __future__ import annotations
import threading, traceback, uuid, time


class Jobs:
    def __init__(self): self._j: dict[str, dict] = {}; self._lock = threading.Lock()

    def submit(self, name: str, fn, *a, **kw) -> str:
        jid = uuid.uuid4().hex[:12]
        rec = {"id": jid, "name": name, "status": "running", "progress": 0.0, "result": None, "error": None,
               "started": time.time()}
        with self._lock: self._j[jid] = rec
        def run():
            try:
                rec["result"] = fn(*a, progress=lambda p: rec.__setitem__("progress", float(p)), **kw) \
                    if "progress" in fn.__code__.co_varnames else fn(*a, **kw)
                rec["status"] = "done"; rec["progress"] = 1.0
            except Exception as exc:
                rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"; rec["trace"] = traceback.format_exc()[-1500:]
            rec["seconds"] = round(time.time() - rec["started"], 2)
        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid: str):
        return self._j.get(jid)
