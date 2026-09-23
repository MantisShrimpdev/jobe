# -*- coding: utf-8 -*-
"""The same readout, against a hosted model, when the local card is too small.

The protocol does not actually require the weights to be on your machine. It
requires a probability for each declared answer letter at one position, and
OpenAI-compatible APIs can give that through `logprobs` + `top_logprobs`. On
OpenRouter, 151 of 456 models expose it.

So this serves the identical `/v1/systemone` endpoint as `jobe.server`, with the
identical prompt, letters and restricted softmax - only the forward pass moves
off-box. Anything already pointed at the local server works unchanged.

    set OPENROUTER_API_KEY=...            (never passed on the command line)
    python -m jobe.remote --model openai/gpt-oss-20b --port 8912

WHAT THIS COSTS, measured in EveryAppKit's docs/SEMANTIC_DECISION_BENCHMARK.md
and stated in this project's README as a reason to run locally:

  * **Only a top-N window comes back**, not the full vocabulary. If none of the
    declared letters lands in the top 20, the decision is UNMEASURED rather
    than wrong, and this returns 422 instead of guessing. Locally that never
    happens - every option is always visible.
  * **Reasoning models do not work.** Their first token is a preamble, not the
    answer, so the readout reads the wrong position. `enable_thinking=False` is
    not reachable through an API. Pick a non-thinking model.
  * The provider sees your evidence. That is the trade for not needing the VRAM.

It is not a replacement for the local path; it is what you use when an 8.9 GB
model will not share a 10 GB card with a browser.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from .prompt import Decision, DecisionError, build_messages
from .server import STATE, Handler, digest, options_for, record
from .slots import LETTERS, MAX_OPTIONS

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY_ENV = "OPENROUTER_API_KEY"


class RemoteError(RuntimeError):
    """The provider could not give a usable distribution."""


def call(messages, model, key, endpoint, top_logprobs=20, timeout=60) -> dict:
    """One completion of one token, with the distribution at that position."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }).encode()
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "X-Title": "jobe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def distribution(payload, n_options: int) -> tuple[dict[str, float], int]:
    """Restrict the returned window to the declared letters and renormalise.

    Mirrors `jobe.readout.restricted_softmax`, except the input is a truncated
    window rather than the whole vocabulary - which is precisely the weakness
    this module carries and the local path does not.
    """
    try:
        choice = payload["choices"][0]
        top = choice["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RemoteError("no logprobs in the response; this model does not "
                          "expose them") from exc

    wanted = {LETTERS[i]: i for i in range(n_options)}
    seen: dict[str, float] = {}
    for entry in top:
        tok = (entry.get("token") or "").strip()
        if tok in wanted and tok not in seen:
            seen[tok] = entry["logprob"]
    if not seen:
        got = ", ".join(repr((e.get("token") or "")[:4]) for e in top[:8])
        raise RemoteError(
            "none of the %d answer letters appeared in the top %d tokens (saw %s); "
            "the decision is unmeasured, not wrong" % (n_options, len(top), got))

    top_lp = max(seen.values())
    exps = {k: math.exp(v - top_lp) for k, v in seen.items()}
    total = sum(exps.values())
    return ({k: v / total for k, v in exps.items()}, len(seen))


def remote_score(decision: Decision, cfg: dict):
    """A `Readout`-shaped result, from somebody else's GPU."""
    decision.validate()
    started = time.perf_counter()
    payload = call(build_messages(decision), cfg["model"], cfg["key"],
                   cfg["endpoint"], cfg["top_logprobs"], cfg["timeout"])
    probs, covered = distribution(payload, len(decision.options))
    letters = {LETTERS[i]: o.id for i, o in enumerate(decision.options)}
    by_id = {letters[k]: v for k, v in probs.items()}
    for option in decision.options:            # unseen options are not impossible
        by_id.setdefault(option.id, 0.0)
    usage = payload.get("usage") or {}
    return _Remote(
        decision_id=decision.id,
        option_ids=tuple(o.id for o in decision.options),
        probabilities=tuple(by_id[o.id] for o in decision.options),
        input_tokens=usage.get("prompt_tokens", 0),
        total_seconds=time.perf_counter() - started,
        covered=covered,
    )


class _Remote:
    """Just enough of `Readout` for the server to serve it."""

    flat = True
    passes = 1

    def __init__(self, decision_id, option_ids, probabilities, input_tokens,
                 total_seconds, covered):
        self.decision_id = decision_id
        self.option_ids = option_ids
        self.probabilities = probabilities
        self.input_tokens = input_tokens
        self.total_seconds = total_seconds
        self.covered = covered

    @property
    def choice(self):
        best = max(range(len(self.option_ids)), key=lambda i: self.probabilities[i])
        return self.option_ids[best]

    @property
    def scores(self):
        return dict(zip(self.option_ids, self.probabilities))

    def confidence(self):
        k = len(self.probabilities)
        if k < 2:
            return 0.0
        return (max(self.probabilities) - 1.0 / k) / (1.0 - 1.0 / k)


#: The live configuration. `remote_score` reads the model off this at call time,
#: so switching it switches every later decision - no restart, no reload.
CFG: dict = {}


def probe(cfg: dict):
    """One real decision. Proves the model exposes logprobs AND answers with a
    letter first rather than a reasoning preamble - the two ways a hosted
    model silently cannot serve this protocol."""
    from .prompt import Option
    return remote_score(Decision(
        id="probe", evidence="A service returned HTTP 500 after a deploy.",
        criterion="Is this an incident?",
        options=(Option("true", "true: it is an incident"),
                 Option("false", "false: it is not"))), cfg)


def install(cfg: dict) -> None:
    """Swap the local forward pass for the remote one, keeping everything else.

    The transport, the option mapping, the 422 policy and the ledger are
    `jobe.server`'s and are not duplicated here - only the scorer changes, so a
    remote answer and a local one are the same shape and land in the same
    records with the same fields.
    """
    from . import server as srv

    CFG.clear()
    CFG.update(cfg)

    def scorer(model, tokenizer, decision, **kw):
        try:
            return remote_score(decision, CFG)
        except RemoteError as exc:
            raise DecisionError(str(exc)) from exc   # 422, not 500: a declared limit

    srv.score = scorer
    srv.score_wide = lambda m, t, d, **kw: scorer(m, t, d)
    STATE["backbone"] = _Backbone(cfg["model"])


class RemoteHandler(Handler):
    """`jobe.server`'s handler plus one verb: change the hosted model live.

        POST /model  {"model": "mistralai/mistral-nemo"}

    The switch is probed before it is accepted. A model that cannot serve the
    protocol - no logprobs, or a thinking preamble - is refused with the reason,
    and the previous model stays in place. This is what lets a test bench flip
    between hosted models from a dropdown without anybody restarting anything.
    """

    def do_GET(self):
        if self.path.rstrip("/") == "/model":
            self._json(200, {"model": CFG.get("model"), "endpoint": CFG.get("endpoint")})
            return
        super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") != "/model":
            super().do_POST()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            wanted = (body.get("model") or "").strip()
            if not wanted:
                self._json(400, {"error": "model is required"})
                return
            trial = dict(CFG, model=wanted)
            t0 = time.perf_counter()
            r = probe(trial)
            CFG["model"] = wanted
            STATE["backbone"] = _Backbone(wanted)
            record({"ts": time.time(), "status": 200, "key": "model-switch",
                    "decider": wanted, "latency_ms": (time.perf_counter() - t0) * 1000})
            self._json(200, {"model": wanted, "probe_ms": (time.perf_counter() - t0) * 1000,
                             "probe_choice": r.choice, "letters_in_window": r.covered})
        except urllib.error.HTTPError as e:
            self._json(422, {"error": "provider refused: HTTP %d %s"
                             % (e.code, e.read().decode(errors="replace")[:200])})
        except RemoteError as e:
            self._json(422, {"error": "%s cannot serve this protocol: %s" % (wanted, e)})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": "%s: %s" % (type(e).__name__, e)})


class _Backbone:
    def __init__(self, name):
        self.name = name
        self.device = "remote"
        self.adapter = None
        self.model = None
        self.tokenizer = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="openai/gpt-oss-20b",
                    help="an OpenRouter model that supports top_logprobs, and that "
                         "does NOT emit a reasoning preamble")
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--port", type=int, default=8912)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--ledger", default=os.environ.get("JOBE_LEDGER", "runs/remote.jsonl"))
    args = ap.parse_args(argv)

    key = os.environ.get(KEY_ENV)
    if not key:
        raise SystemExit(
            "%s is not set.\n"
            "Set it in your shell or a .env you control - it is deliberately not a "
            "command-line flag, so it cannot end up in shell history or a process list."
            % KEY_ENV)

    cfg = {"model": args.model, "key": key, "endpoint": args.endpoint,
           "top_logprobs": args.top_logprobs, "timeout": args.timeout}
    STATE["ledger"] = args.ledger or None
    STATE["max_tokens"] = 10 ** 9      # the provider enforces its own context limit
    if STATE["ledger"]:
        os.makedirs(os.path.dirname(os.path.abspath(STATE["ledger"])) or ".",
                    exist_ok=True)
    install(cfg)

    print("probing %s …" % args.model, flush=True)
    t0 = time.perf_counter()
    try:
        r = probe(cfg)
    except urllib.error.HTTPError as e:
        raise SystemExit("provider refused: HTTP %d %s" % (e.code, e.read()[:200]))
    except RemoteError as e:
        raise SystemExit(
            "%s cannot serve this protocol: %s\n"
            "Pick a model that exposes top_logprobs and does not emit a reasoning "
            "preamble before its answer." % (args.model, e))
    print("  ok in %.2fs — %s at %.3f, %d of 2 letters in the window"
          % (time.perf_counter() - t0, r.choice, max(r.probabilities), r.covered),
          flush=True)
    print("ready on http://%s:%d/v1/systemone   model: %s (remote)   ledger: %s"
          % (args.host, args.port, args.model, STATE["ledger"] or "off"), flush=True)
    ThreadingHTTPServer((args.host, args.port), RemoteHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
