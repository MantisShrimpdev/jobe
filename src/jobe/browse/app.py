# -*- coding: utf-8 -*-
"""The chat window's server: one process holding the model, the browser and the chat.

    PYTHONPATH=src python -m jobe.browse.app            # serve on 127.0.0.1:7900
    python desktop/jobe_chat.py                         # ...and open the always-on-top window

One process on purpose. The first version of this was three - a decision
server, a Node console and a Tk widget - and most of an evening's failures
were the seams between them: a stale session behind one port, a wedged lock
behind another, two copies of an 8.9 GB model fighting over one card. Here a
single worker thread owns the model and the browser (Playwright's sync API
must stay on the thread that created it), HTTP threads only enqueue commands
and read events, and there is nothing to keep in step.

Security, because a local server that drives a browser is a target: any page
you visit can send a request to 127.0.0.1. So every command must carry a token
that is minted per launch and only readable by this app's own page (another
origin can send a request but cannot read the response that holds it), the
Origin header must be this app's, and the socket binds loopback only.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).with_name("static")
MODEL = os.environ.get("JOBE_MODEL", r"D:\Coding\models\qwen35-4b")
TOKEN = secrets.token_urlsafe(24)
TITLE = "Jobe"

#: Events the window has already missed are replayed on connect, so reopening
#: it shows the conversation. Screenshots are kept only on the newest events -
#: a long session would otherwise hold hundreds of megabytes of JPEGs.
REPLAY = 300
KEEP_SHOTS = 10


class Hub:
    """Fan events out to every connected window; remember recent ones."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subs: list[queue.Queue] = []
        self.log: list[dict] = []
        self.seq = 0

    def emit(self, kind: str, data: dict):
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "kind": kind, "t": time.time(), **data}
            self.log.append(ev)
            shots = [e for e in self.log if e.get("screenshot")]
            for old in shots[:-KEEP_SHOTS]:
                old.pop("screenshot", None)
            del self.log[:-REPLAY]
            for q in list(self.subs):
                q.put(ev)

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.subs.append(q)
            return q, list(self.log)

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


HUB = Hub()
STATE = {"ready": False, "busy": False, "model": None, "load_error": None, "session": None,
         "commands": queue.Queue(), "started": time.time()}


# ------------------------------------------------------------------ worker


def worker(headless: bool, home: str):
    """The only thread that touches the model or the browser."""
    try:
        HUB.emit("status", {"text": "Loading Jobe…", "phase": "loading"})
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from jobe import load
        from jobe.browse.agent import Session
        from jobe.browse.policy import Policy
        t0 = time.perf_counter()
        bb = load(MODEL, device="auto")
        STATE["model"] = Path(MODEL).name
        session = Session(Policy(bb), HUB.emit, headless=headless, home=home)
        STATE["session"] = session
        STATE["ready"] = True
        HUB.emit("status", {"text": "Ready", "phase": "ready", "model": STATE["model"],
                            "load_s": time.perf_counter() - t0})
    except Exception as exc:  # noqa: BLE001
        STATE["load_error"] = "%s: %s" % (type(exc).__name__, exc)
        HUB.emit("error", {"text": "Could not load the model - " + STATE["load_error"]})
        return

    while True:
        cmd, arg = STATE["commands"].get()
        if cmd == "quit":
            session.close()
            return
        STATE["busy"] = True
        HUB.emit("busy", {"busy": True, "text": arg if cmd == "say" else cmd})
        try:
            if cmd == "say":
                session.say(arg)
            elif cmd == "close":
                session.close()
                HUB.emit("done", {"status": "closed"})
        finally:
            STATE["busy"] = False
            session.stop_requested = False
            HUB.emit("busy", {"busy": False})


# -------------------------------------------------------------------- pins
#
# Windows only. Three things were learned the hard way on 2026-09-24:
#   * SetWindowPos needs declared argument types. Undeclared, ctypes passes
#     HWND_TOPMOST (-1) as a 32-bit int, which 64-bit Windows reads as
#     0x00000000FFFFFFFF - not HWND_TOPMOST - and the call quietly does
#     nothing. The first launcher reported "pinned on top" for a window that
#     was not.
#   * A new browser window resets its own z-order while it finishes setting
#     up, undoing an early pin - so the launcher re-checks until a pin holds.
#   * Chromium stops painting a window it believes is fully covered. Opened
#     from a background process, the chat window started behind another one,
#     and raising it to topmost is a change Chromium does not re-check: it
#     stayed black (so did a one-pixel resize). Minimise-then-restore, without
#     taking focus, made it paint - see `reveal`.


def _user32():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetForegroundWindow.restype = wintypes.HWND
    return user32


def _windows(title: str = TITLE) -> list:
    """Visible top-level windows whose title is exactly `title`."""
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value == title:
                found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return found


def window_open(title: str = TITLE) -> bool:
    return bool(_windows(title))


def set_topmost(on: bool, title: str = TITLE) -> bool:
    """Pin or unpin the chat window above every other window. True if a window took it."""
    windows = _windows(title)
    if not windows:
        return False
    from ctypes import wintypes
    user32 = _user32()
    after = wintypes.HWND(-1 if on else -2)            # HWND_TOPMOST / HWND_NOTOPMOST
    flags = 0x0001 | 0x0002 | 0x0010                    # NOSIZE | NOMOVE | NOACTIVATE
    return any([bool(user32.SetWindowPos(h, after, 0, 0, 0, 0, flags)) for h in windows])


def is_topmost(title: str = TITLE) -> bool:
    """True when there is such a window and every one of them is pinned on top."""
    windows = _windows(title)
    if not windows:
        return False
    user32 = _user32()
    return all(user32.GetWindowLongW(h, -20) & 0x8 for h in windows)   # WS_EX_TOPMOST


def reveal(title: str = TITLE) -> None:
    """Minimise and restore, without focus, any such window that is not in front."""
    user32 = _user32() if os.name == "nt" else None
    for hwnd in _windows(title):
        if user32.GetForegroundWindow() == hwnd:
            continue                                    # the foreground window is never covered
        user32.ShowWindow(hwnd, 6)                      # SW_MINIMIZE
        time.sleep(0.3)
        user32.ShowWindow(hwnd, 4)                      # SW_SHOWNOACTIVATE


# -------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    server_version = "jobe-chat/0.1"

    def log_message(self, fmt, *args):
        pass

    # -- guards
    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        allowed = {"http://127.0.0.1:%d" % self.server.server_port,
                   "http://localhost:%d" % self.server.server_port}
        if host not in {a.split("//")[1] for a in allowed}:
            return False                       # DNS-rebinding guard
        # A browser that labels the request cross-site is answered with nothing.
        # (DeepSeek Harness refuses it outright at its carrier too.)
        if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            return False
        return origin is None or origin in allowed

    def _authorised(self) -> bool:
        return self._origin_ok() and secrets.compare_digest(
            self.headers.get("X-Jobe-Token", ""), TOKEN)

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routes
    def do_GET(self):
        if not self._origin_ok():
            return self._json(403, {"error": "forbidden"})
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # another site must not frame the window and steer clicks into it
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            return self.wfile.write(body)
        if path.startswith("/static/"):
            f = (STATIC / path[len("/static/"):]).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file():
                return self._json(404, {"error": "not found"})
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        if path == "/health":
            s = STATE["session"]
            return self._json(200, {"ready": STATE["ready"], "busy": STATE["busy"],
                                    "model": STATE["model"], "error": STATE["load_error"],
                                    "browser_open": bool(s and s.browser and s.browser.alive)})
        if path == "/events":
            # Read with fetch() rather than EventSource precisely so the token
            # travels as a header and never appears in a URL.
            if not secrets.compare_digest(self.headers.get("X-Jobe-Token", ""), TOKEN):
                return self._json(403, {"error": "token"})
            return self._stream()
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorised():
            return self._json(403, {"error": "forbidden"})
        # Media-type fence, from DeepSeek Harness's browser-trust note: a
        # cross-site "simple" POST (text/plain, a form) is sent without a CORS
        # preflight, so every command must be declared JSON - which such a
        # request cannot be - before anything is parsed.
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            return self._json(415, {"error": "commands are application/json"})
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad json"})
        path = self.path.split("?")[0]
        if path == "/say":
            text = (body.get("text") or "").strip()[:1000]
            if not text:
                return self._json(400, {"error": "empty"})
            if not STATE["ready"]:
                return self._json(503, {"error": "still loading"})
            HUB.emit("user", {"text": text})
            STATE["commands"].put(("say", text))
            return self._json(202, {"queued": True})
        if path == "/stop":
            s = STATE["session"]
            if s:
                s.stop_requested = True
            return self._json(200, {"stopping": True})
        if path == "/close":
            STATE["commands"].put(("close", None))
            return self._json(202, {"queued": True})
        if path == "/pin":
            return self._json(200, {"pinned": bool(body.get("on")),
                                    "found": set_topmost(bool(body.get("on")))})
        return self._json(404, {"error": "not found"})

    def _stream(self):
        q, backlog = HUB.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for ev in backlog:
                self._send_event(ev)
            while True:
                try:
                    ev = q.get(timeout=15)
                    self._send_event(ev)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    def _send_event(self, ev):
        self.wfile.write(b"data: " + json.dumps(ev, default=str).encode() + b"\n\n")
        self.wfile.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=7900)
    ap.add_argument("--headless", action="store_true", help="drive the browser without showing it")
    ap.add_argument("--home", default="https://duckduckgo.com")
    args = ap.parse_args(argv)
    from .browser import protect
    protect(args.port)                     # the agent's browser never loads this window's server
    threading.Thread(target=worker, args=(args.headless, args.home), daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True
    print("Jobe chat on http://127.0.0.1:%d/" % args.port, flush=True)
    try:
        httpd.serve_forever()
    finally:
        STATE["commands"].put(("quit", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
