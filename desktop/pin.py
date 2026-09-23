# -*- coding: utf-8 -*-
"""A small always-on-top window you talk to, with a switchable brain.

    > hi jobe, open browser
    > search for the latest news on github
    > open the top one

Same three lines, two backends:

  Jobe (local)   the frozen Qwen3.5-4B on your card, via jobe.server on :8911
  OpenRouter     any hosted model that exposes top_logprobs, via jobe.remote on
                 :8912 - the SAME readout protocol, the forward pass off-box

Both drive the SAME harness (jev-browser, unmodified) through the same
console, so what differs between a Jobe run and an OpenRouter run is exactly
one thing: which model read the distribution. That is what makes this a test
bench rather than a demo.

Three properties make the conversation work, and only the first is obvious:

  * The browser SESSION PERSISTS between lines. "open the top one" has a
    referent because the results page is still there.
  * The model never writes a URL, a selector or a plan - it never generates
    anything. Each line is classified into one of four fixed intents (open,
    act, read, close), a choice over a declared set, the one thing a readout
    does. Everything else becomes a goal for the agent.
  * Every action inside that goal is itself one forward pass.

Settings (backend, model, window position) persist in ~/.jobe-pin.json and
every line of every conversation is appended to ~/.jobe-pin-transcript.log.

    PYTHONPATH=src python -m jobe.server  --model <path> --port 8911     # local
    PYTHONPATH=src python -m jobe.remote  --model <id>   --port 8912     # hosted
    node console/server.mjs                     # CONSOLE_PORT=7899 JOBE_PORT=8911
    node console/server.mjs                     # CONSOLE_PORT=7898 JOBE_PORT=8912
    python desktop/pin.py
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import font as tkfont

BACKENDS = {
    "Jobe (local)": {"console": 7899, "decision": 8911, "hosted": False},
    "OpenRouter":   {"console": 7898, "decision": 8912, "hosted": True},
}
#: Non-reasoning, cheap, and in OpenRouter's list of models exposing top_logprobs.
#: A thinking model answers with a preamble instead of a letter and the remote
#: server's probe will refuse it - by design, with the reason.
DEFAULT_HOSTED_MODEL = "mistralai/mistral-nemo"

SETTINGS = os.path.join(os.path.expanduser("~"), ".jobe-pin.json")
TRANSCRIPT = os.path.join(os.path.expanduser("~"), ".jobe-pin-transcript.log")

BG = "#111418"
PANEL = "#181c22"
FIELD = "#1e242c"
LINE = "#2a323c"
INK = "#e6edf3"
DIM = "#98a4b0"
FAINT = "#6b7783"
ACCENT = "#6aa6ff"
HOSTED = "#c59aff"
OK = "#3fb950"
WARN = "#d29922"
BAD = "#e5484d"


#: The three populations this model has been measured to have: above 0.85 it is
#: 0.959 accurate, between 0.45 and 0.85 it is 0.643, below 0.45 it is 0.067.
def band(v: float) -> str:
    return OK if v >= 0.85 else WARN if v >= 0.45 else BAD


def load_settings() -> dict:
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    try:
        with open(SETTINGS, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except Exception:
        pass


def log_line(kind: str, text: str) -> None:
    try:
        with open(TRANSCRIPT, "a", encoding="utf-8") as fh:
            fh.write("%s\t%s\t%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), kind,
                                       text.replace("\n", " ")))
    except Exception:
        pass


class Pin(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.title("Jobe")
        self.configure(bg=BG)
        self.geometry(self.settings.get("geometry", "460x640+40+60"))
        self.minsize(380, 400)
        self.attributes("-topmost", True)
        self.pinned = True
        self.busy = False
        self.q: queue.Queue = queue.Queue()

        self.mono = tkfont.Font(family="Consolas", size=9)
        self.ui = tkfont.Font(family="Segoe UI", size=9)
        self.ui_b = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.head = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.you = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        self.backend = tk.StringVar(value=self.settings.get("backend", "Jobe (local)"))
        if self.backend.get() not in BACKENDS:
            self.backend.set("Jobe (local)")
        self.model = tk.StringVar(value=self.settings.get("model", DEFAULT_HOSTED_MODEL))

        self._build()
        self.after(120, self._drain)
        self._health()
        self._say_line("Tell me what to do. Try: open browser", FAINT)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ---------------------------------------------------------- addresses

    def _cfg(self) -> dict:
        return BACKENDS[self.backend.get()]

    def _console(self) -> str:
        return "http://127.0.0.1:%d" % self._cfg()["console"]

    def _decision(self) -> str:
        return "http://127.0.0.1:%d" % self._cfg()["decision"]

    def _tag(self) -> str:
        return ("openrouter:" + self.model.get().split("/")[-1]) if self._cfg()["hosted"] else "jobe"

    # ------------------------------------------------------------------ ui

    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(top, text="Jobe", font=self.head, bg=BG, fg=INK).pack(side="left")
        self.pin_btn = tk.Button(top, text="pinned", font=self.mono, bd=0,
                                 bg=PANEL, fg=ACCENT, activebackground=LINE,
                                 activeforeground=INK, padx=8, pady=2,
                                 cursor="hand2", command=self._toggle_pin)
        self.pin_btn.pack(side="right")
        self.status = tk.Label(top, text="connecting…", font=self.mono, bg=BG, fg=FAINT)
        self.status.pack(side="right", padx=8)

        # ---- the brain switch
        sw = tk.Frame(self, bg=BG)
        sw.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(sw, text="brain", font=self.mono, bg=BG, fg=FAINT).pack(side="left")
        for name in BACKENDS:
            tk.Radiobutton(sw, text=name, value=name, variable=self.backend,
                           command=self._swap, font=self.ui, bg=BG, fg=DIM,
                           selectcolor=BG, activebackground=BG, activeforeground=INK,
                           bd=0, highlightthickness=0, indicatoron=False,
                           padx=10, pady=3, cursor="hand2").pack(side="left", padx=(6, 0))
        self.model_row = tk.Frame(self, bg=BG)
        tk.Label(self.model_row, text="model", font=self.mono, bg=BG, fg=FAINT).pack(side="left")
        self.model_entry = tk.Entry(self.model_row, textvariable=self.model, font=self.mono,
                                    bg=FIELD, fg=INK, bd=0, insertbackground=INK,
                                    highlightthickness=1, highlightbackground=LINE,
                                    highlightcolor=HOSTED)
        self.model_entry.pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        self.model_entry.bind("<Return>", lambda e: self._set_model())
        tk.Button(self.model_row, text="use", font=self.ui_b, bd=0, bg=HOSTED, fg="#140a24",
                  activebackground="#d9bfff", padx=10, pady=3, cursor="hand2",
                  command=self._set_model).pack(side="right")

        # ---- transcript
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True, padx=12, pady=(4, 4))
        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview,
                          bg=BG, troughcolor=BG, bd=0, highlightthickness=0,
                          activebackground=LINE)
        self.feed = tk.Frame(self.canvas, bg=BG)
        self.feed.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.win = self.canvas.create_window((0, 0), window=self.feed, anchor="nw")
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfig(self.win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(
            int(-e.delta / 120), "units"))

        # ---- input
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=12, pady=(4, 12))
        self.entry = tk.Entry(bar, font=self.ui, bg=FIELD, fg=INK, bd=0,
                              insertbackground=INK, highlightthickness=1,
                              highlightbackground=LINE, highlightcolor=ACCENT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.entry.bind("<Return>", lambda e: self._send())
        self.entry.focus_set()
        self.send_btn = tk.Button(bar, text="Go", font=self.ui_b, bd=0, bg=ACCENT,
                                  fg="#08121f", activebackground="#8ab9ff",
                                  padx=14, pady=5, cursor="hand2", command=self._send)
        self.send_btn.pack(side="right", padx=(8, 0))
        # Always enabled, on purpose: the failure this fixes is a run that will
        # not finish, and a Stop that greys out while stuck is no Stop at all.
        self.stop_btn = tk.Button(bar, text="Stop", font=self.ui_b, bd=0, bg=PANEL,
                                  fg=BAD, activebackground=LINE, activeforeground=INK,
                                  padx=12, pady=5, cursor="hand2", command=self._stop)
        self.stop_btn.pack(side="right", padx=(8, 0))

        quick = tk.Frame(self, bg=BG)
        quick.pack(fill="x", padx=12, pady=(0, 10))
        for label in ("open browser", "what's on the page?", "close browser"):
            tk.Button(quick, text=label, font=self.mono, bd=0, bg=PANEL, fg=DIM,
                      activebackground=LINE, activeforeground=INK, padx=8, pady=2,
                      cursor="hand2",
                      command=lambda t=label: self._send(t)).pack(side="left", padx=(0, 6))
        self._swap(first=True)

    # -------------------------------------------------------------- output

    def _say_line(self, text, fg=DIM, font=None, indent=0):
        lbl = tk.Label(self.feed, text=text, font=font or self.mono, bg=BG, fg=fg,
                       anchor="w", justify="left", wraplength=390 - indent)
        lbl.pack(fill="x", padx=(indent, 0), pady=1)
        self._scroll()
        return lbl

    def _you(self, text):
        f = tk.Frame(self.feed, bg=BG)
        f.pack(fill="x", pady=(10, 2))
        tk.Label(f, text="> " + text, font=self.you, bg=BG, fg=INK, anchor="w",
                 justify="left", wraplength=340).pack(side="left", fill="x", expand=True)
        tk.Label(f, text=self._tag(), font=self.mono, bg=BG,
                 fg=HOSTED if self._cfg()["hosted"] else ACCENT).pack(side="right")
        self._scroll()

    def _bar(self, name, p, winner=False):
        row = tk.Frame(self.feed, bg=BG)
        row.pack(fill="x", pady=1, padx=(14, 0))
        tk.Label(row, text=name[:20], font=self.mono, bg=BG,
                 fg=INK if winner else FAINT, anchor="w", width=18).pack(side="left")
        tk.Label(row, text="%.2f" % p, font=self.mono, bg=BG,
                 fg=INK if winner else FAINT, width=5).pack(side="right")
        track = tk.Frame(row, bg=LINE, height=3)
        track.pack(side="left", fill="x", expand=True, padx=6)
        track.pack_propagate(False)
        tk.Frame(track, bg=band(p) if winner else FAINT, height=3).place(
            relwidth=max(0.02, min(1.0, p)), relheight=1)
        self._scroll()

    def _scroll(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    # ------------------------------------------------------------- actions

    def _persist(self):
        self.settings.update(backend=self.backend.get(), model=self.model.get(),
                             geometry=self.geometry())
        save_settings(self.settings)

    def _quit(self):
        self._persist()
        self.destroy()

    def _swap(self, first=False):
        hosted = self._cfg()["hosted"]
        if hosted:
            self.model_row.pack(fill="x", padx=12, pady=(0, 6), before=self.canvas.master)
        else:
            self.model_row.pack_forget()
        if not first:
            self._say_line("brain -> %s" % self._tag(), HOSTED if hosted else ACCENT, self.ui_b)
            log_line("brain", self._tag())
        self._persist()
        self._health()

    def _toggle_pin(self):
        self.pinned = not self.pinned
        self.attributes("-topmost", self.pinned)
        self.pin_btn.config(text="pinned" if self.pinned else "unpinned",
                            fg=ACCENT if self.pinned else FAINT)

    def _set_model(self):
        """Ask the remote server to switch, and show what its probe said."""
        wanted = self.model.get().strip()
        if not wanted:
            return
        self._say_line("switching hosted model to %s …" % wanted, FAINT)

        def work():
            try:
                req = urllib.request.Request(
                    self._decision() + "/model", method="POST",
                    data=json.dumps({"model": wanted}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read())
                self.q.put(("line", ("now %s — probe answered %s in %.0f ms, %d of 2 letters in window"
                                     % (d["model"], d.get("probe_choice"), d.get("probe_ms", 0),
                                        d.get("letters_in_window", 0)), OK)))
                self.q.put(("persist", None))
            except urllib.error.HTTPError as e:
                try:
                    msg = json.loads(e.read()).get("error", "")
                except Exception:
                    msg = "HTTP %d" % e.code
                self.q.put(("line", ("refused: %s" % msg[:220], BAD)))
            except urllib.error.URLError:
                self.q.put(("line", ("remote server not running on :8912 — set OPENROUTER_API_KEY "
                                     "and start `python -m jobe.remote`", BAD)))
            except Exception as e:
                self.q.put(("line", (str(e)[:200], BAD)))
        threading.Thread(target=work, daemon=True).start()

    def _stop(self):
        console = self._console()

        def work():
            try:
                req = urllib.request.Request(console + "/stop", method="POST", data=b"")
                urllib.request.urlopen(req, timeout=30).read()
                self.q.put(("line", ("stopped — lock cleared, browser closed", WARN)))
            except Exception as e:
                self.q.put(("line", ("stop failed: %s" % str(e)[:90], BAD)))
            finally:
                self.q.put(("idle", None))
        threading.Thread(target=work, daemon=True).start()

    def _send(self, text=None):
        if self.busy:
            return
        text = (text if text is not None else self.entry.get()).strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._you(text)
        log_line("you@" + self._tag(), text)
        self.busy = True
        self.send_btn.config(text="…", state="disabled")
        console, decision = self._console(), self._decision()

        def work():
            try:
                req = urllib.request.Request(
                    console + "/say", method="POST",
                    data=json.dumps({"text": text}).encode(),
                    headers={"Content-Type": "application/json"})
                buf = ""
                with urllib.request.urlopen(req, timeout=3600) as r:
                    while True:
                        chunk = r.read1(2048)
                        if not chunk:
                            break
                        buf += chunk.decode("utf-8", "replace")
                        while "\n\n" in buf:
                            block, buf = buf.split("\n\n", 1)
                            ev = dat = None
                            for ln in block.splitlines():
                                if ln.startswith("event: "):
                                    ev = ln[7:]
                                elif ln.startswith("data: "):
                                    dat = json.loads(ln[6:])
                            if ev:
                                self.q.put(("stream", (ev, dat)))
            except urllib.error.URLError as e:
                self.q.put(("line", ("console not running on %s — %s"
                                     % (console.split("//")[1], str(e.reason)[:80]), BAD)))
            except Exception as e:
                self.q.put(("line", (str(e)[:200], BAD)))
            finally:
                self.q.put(("idle", None))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------- plumbing

    def _health(self):
        # Captured NOW, and checked again when the result lands. A poll fired
        # while OpenRouter was selected can return after the user has flipped
        # back to Jobe, and it would otherwise stamp "server down on :8912"
        # over a perfectly healthy local backend for the next ten seconds.
        asked = self.backend.get()
        decision, console, hosted, port = (self._decision(), self._console(),
                                           self._cfg()["hosted"], self._cfg()["decision"])

        def work():
            try:
                with urllib.request.urlopen(decision + "/health", timeout=5) as r:
                    h = json.loads(r.read())
                name = (h.get("model") or "").replace("\\", "/").split("/")[-1]
                if hosted:
                    self.q.put(("status", ("%s · hosted" % name, OK, asked)))
                else:
                    free = h.get("free_vram_mb") or 0
                    self.q.put(("status", ("%s · %d MiB" % (name, free),
                                           OK if free > 600 else WARN, asked)))
            except Exception:
                self.q.put(("status", ("decision server down on :%d%s"
                                       % (port, " (set OPENROUTER_API_KEY)" if hosted else ""),
                                       BAD, asked)))
            try:
                with urllib.request.urlopen(console + "/health", timeout=4) as r:
                    lock = (json.loads(r.read()) or {}).get("lock") or {}
                self.q.put(("busy", (lock.get("text", ""), (lock.get("for_ms") or 0) / 1000)
                            if lock.get("busy") else None))
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()
        if not getattr(self, "_health_armed", False):
            self._health_armed = True
            self.after(10000, self._health_tick)

    def _health_tick(self):
        self._health_armed = False
        self._health()

    def _drain(self):
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                if payload[2] == self.backend.get():      # drop results for a backend we left
                    self.status.config(text=payload[0], fg=payload[1])
            elif kind == "line":
                self._say_line(payload[0], payload[1])
                log_line("note", payload[0])
            elif kind == "persist":
                self._persist()
            elif kind == "idle":
                self.busy = False
                self.send_btn.config(text="Go", state="normal")
            elif kind == "busy":
                self.stop_btn.config(text="Stop (%ds)" % int(payload[1]) if payload else "Stop")
            elif kind == "stream":
                self._stream(*payload)
        self.after(120, self._drain)

    def _stream(self, ev, d):
        if ev == "intent":
            msg = "read as: %s   (%.2f, %d ms)" % (d["intent"], d.get("confidence", 0), d.get("ms", 0))
            self._say_line(msg, band(d.get("confidence", 0)), self.ui_b)
            log_line("intent", msg)
        elif ev == "status":
            if d.get("phase") == "opening":
                self._say_line("opening " + str(d.get("url", "")), FAINT)
        elif ev == "round":
            msg = "%s%s  ->  %s" % (d["tool"], " ← " + d["value"] if d.get("value") else "",
                                    (d.get("el") or "")[:40])
            self._say_line(msg, DIM, self.ui, indent=14)
            self._bar("target", d["p_target"], winner=True)
            log_line("round", "%s p=%.2f" % (msg, d["p_target"]))
        elif ev == "page":
            self._say_line(d["text"][:1200], DIM, indent=14)
        elif ev == "done":
            good = d["status"] in ("done", "likely_done", "ready", "read", "closed")
            msg = "%s — %d decisions, %.1fs" % (d["status"], d.get("calls", 0), d.get("ms", 0) / 1000)
            self._say_line(msg, OK if good else BAD, self.ui_b)
            log_line("done@" + self._tag(), msg + (" | " + d["info"] if d.get("info") else ""))
            if d.get("info"):
                self._say_line(d["info"], DIM, indent=14)
            if d.get("url"):
                self._say_line(d["url"][:70], FAINT, indent=14)
        elif ev == "error":
            self._say_line(d.get("message", "error"), BAD)
            log_line("error", d.get("message", ""))


if __name__ == "__main__":
    Pin().mainloop()
