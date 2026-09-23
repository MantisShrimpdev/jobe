# -*- coding: utf-8 -*-
"""Open the Jobe chat window: always on top, driving a browser through Jobe.

    Double-click desktop/Jobe.cmd, or:  .venv/Scripts/python.exe desktop/jobe_chat.py

Starts the one-process server (model + browser + chat, `jobe.browse.app`) if it
is not already running, opens it as a chromeless app window with a profile of
its own - separate from your everyday browser, so it neither sees your sessions
nor fights your window layout - and pins it above every other window. Run it
again while it is open and it just brings the window back.

Chrome, not Edge. The first version opened an Edge app window, and on a PC
signed in with a Microsoft account Edge signs every NEW profile into that
account and turns on sync: the chat window opened on "We are now syncing your
browsing data across all your devices" instead of the chat. Chrome does not
sign a fresh profile in by itself. Without Chrome, the Chromium that
Playwright installed for the agent is used - it is always there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("JOBE_CHAT_PORT", "7900"))
URL = "http://127.0.0.1:%d/" % PORT
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
          os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")]
PROFILE = Path(os.environ.get("LOCALAPPDATA", str(ROOT / "runs"))) / "JobeChatWindow"


def health():
    try:
        with urllib.request.urlopen(URL + "health", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def start_server() -> None:
    (ROOT / "runs").mkdir(exist_ok=True)
    log = open(ROOT / "runs" / "chat-server.log", "a", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1")
    python = str(PYTHON if PYTHON.exists() else sys.executable)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000   # CREATE_NO_WINDOW
    subprocess.Popen([python, "-m", "jobe.browse.app", "--port", str(PORT)], cwd=str(ROOT),
                     env=env, stdout=log, stderr=log, creationflags=flags)


def playwright_chromium() -> str | None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            path = p.chromium.executable_path
        return path if path and os.path.exists(path) else None
    except Exception:
        return None


def open_window() -> None:
    exe = next((p for p in CHROME if os.path.exists(p)), None) or playwright_chromium()
    if exe is None:
        import webbrowser
        webbrowser.open(URL)
        return
    subprocess.Popen([exe, "--app=" + URL, "--user-data-dir=" + str(PROFILE),
                      "--window-size=460,780", "--window-position=24,40", "--no-first-run",
                      "--no-default-browser-check", "--disable-sync", "--disable-features=Translate"])


def main() -> int:
    if health() is None:
        print("starting Jobe (the model takes ~15 s to load)…", flush=True)
        start_server()
        for _ in range(100):
            if health() is not None:
                break
            time.sleep(0.3)
    sys.path.insert(0, str(ROOT / "src"))
    from jobe.browse.app import is_topmost, reveal, set_topmost, window_open
    if not window_open():
        open_window()                     # already open: just pin it and bring it back
    # Pin, then keep checking: while a new window finishes setting itself up the
    # browser resets its z-order, which silently undid the first pin. Done once
    # the pin has held for a second and a half.
    held = 0
    for _ in range(60):
        time.sleep(0.25)
        if is_topmost():
            held += 1
            if held >= 6:
                reveal()                  # a window that opened behind another never painted
                print("Jobe chat is open and pinned on top.", flush=True)
                return 0
        else:
            held = 0
            set_topmost(True)
    # Windows only lets a window be raised by a process allowed into the
    # foreground; a double-click launch is, a background one may not be.
    reveal()
    print("Jobe chat is open (it would not stay on top - use the pin button).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
