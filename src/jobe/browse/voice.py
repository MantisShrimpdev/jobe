# -*- coding: utf-8 -*-
"""Speech to text for the chat window's microphone - on this PC, through Juno.

Juno already runs a local Whisper sidecar,
`tools/stt_server.py`: POST raw WAV bytes to /transcribe, get {"text", "ms"}
back, nothing leaves the machine. The window records, encodes 16 kHz mono WAV
itself, and this module hands it to that sidecar - Juno's if it is running,
otherwise one started here from Juno's own script, which Juno's supervisor
adopts when it starts later. Chrome's built-in recognition was the other
option; it sends the audio to Google.

    JOBE_STT_URL      where the sidecar listens      (default http://127.0.0.1:8793)
    JUNO_STT_SCRIPT   Juno's stt_server.py           (no default: set it, for example in desktop/local.cmd)
    JOBE_STT_PYTHON   a Python with faster-whisper   (default: the first `python` on PATH that has it)

Nothing is installed anywhere: a Python that already has faster-whisper is
used as it is. The sidecar runs on the CPU, so it never competes with the model
on the card, and it does not write what was said into its log.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

STT_URL = os.environ.get("JOBE_STT_URL", "http://127.0.0.1:8793").rstrip("/")
_JUNO_STT = os.environ.get("JUNO_STT_SCRIPT", "").strip()
JUNO_STT = Path(_JUNO_STT) if _JUNO_STT else None
JUNO_BRIDGE = "http://127.0.0.1:8791/"
#: When Jobe starts the sidecar itself: small.en, not Juno's base.en default,
#: which misheard the first spoken requests; small.en is the next size up, already
#: in the machine's model cache, and still well under two seconds for a short command.
MODEL = os.environ.get("JOBE_STT_MODEL", "small.en")
LOG = Path(__file__).resolve().parents[3] / "runs" / "stt.log"

#: 16 kHz mono 16-bit is 32 kB a second; this is about five minutes.
MAX_WAV = 10 * 1024 * 1024

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def healthy(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(STT_URL + "/health", timeout=timeout) as r:
            return bool(json.loads(r.read() or b"{}").get("ok"))
    except (OSError, ValueError):
        return False


def _python() -> str | None:
    """A Python that can already import faster_whisper - used, never installed into."""
    for exe in (os.environ.get("JOBE_STT_PYTHON"), shutil.which("python"), shutil.which("py")):
        if not exe:
            continue
        try:
            if subprocess.run([exe, "-c", "import faster_whisper"], capture_output=True,
                              timeout=60).returncode == 0:
                return exe
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def ensure(wait: float = 90.0) -> str | None:
    """Make sure a sidecar is answering. None when it is; otherwise what is wrong."""
    global _proc
    with _lock:
        if healthy():
            return None
        if _proc is None or _proc.poll() is not None:
            if JUNO_STT is None:
                return "speech-to-text needs Juno's sidecar: start Juno, or set JUNO_STT_SCRIPT"
            if not JUNO_STT.exists():
                return "speech-to-text needs Juno's sidecar, and %s is not there" % JUNO_STT
            py = _python()
            if py is None:
                return "no Python with faster-whisper was found for speech-to-text"
            LOG.parent.mkdir(parents=True, exist_ok=True)
            log = open(LOG, "a", encoding="utf-8")
            port = str(urlparse(STT_URL).port or 8793)
            flags = 0x08000000 | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            _proc = subprocess.Popen([py, str(JUNO_STT), "--port", port, "--model", MODEL],
                                     stdout=log, stderr=log,
                                     cwd=str(JUNO_STT.parent.parent), creationflags=flags)
        deadline = time.time() + wait
        while time.time() < deadline:
            if healthy():
                return None
            if _proc.poll() is not None:
                return "the speech sidecar stopped while starting - see runs/stt.log"
            time.sleep(0.5)
        return "the speech sidecar did not become ready in time"


def warm() -> None:
    """Start loading the model as soon as the person starts talking, not when they stop."""
    threading.Thread(target=ensure, daemon=True).start()


def transcribe(wav: bytes) -> dict:
    """{"text": ..., "ms": ...}, or {"error": ...} saying what to fix."""
    if not wav.startswith(b"RIFF") or len(wav) > MAX_WAV:
        return {"error": "that was not a WAV recording Jobe can use"}
    problem = ensure()
    if problem:
        return {"error": problem}
    req = urllib.request.Request(STT_URL + "/transcribe", data=wav, method="POST",
                                 headers={"Content-Type": "audio/wav"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read()).get("error") or "transcription failed"}
        except ValueError:
            return {"error": "transcription failed (HTTP %d)" % e.code}
    except OSError as e:
        return {"error": "could not reach speech-to-text: %s" % e}


def _juno_running() -> bool:
    try:
        urllib.request.urlopen(JUNO_BRIDGE, timeout=1)
        return True
    except urllib.error.HTTPError:
        return True                      # it answered, even if not with 200
    except OSError:
        return False


def stop() -> None:
    """Stop the sidecar only if Jobe started it and Juno is not using it."""
    if _proc is not None and _proc.poll() is None and not _juno_running():
        _proc.terminate()
