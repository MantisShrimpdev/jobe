# -*- coding: utf-8 -*-
"""A TypeSafe-compatible decision endpoint, served by the local readout.

    POST /v1/systemone   {"state": ..., "questions": {"<key>": {...}}}
    ->                   {"model": ..., "answers": {"<key>": {...}}, "usage": {...}}

Anything built against the hosted decision API points here unchanged. That is
the whole reason this exists: `browser-use/jev-ultrafast` and
`Ying-Kai-Liao/jev-browser` both speak this format and both ship their own
benchmarks, so serving it turns their harnesses into evaluation we did not have
to write.

Adapted from AXIOM v1's `axiom/server.py` (same author, project retired
2026-09-23); the transport is its shape, the readout and everything below are
this project's.

THE ANSWERS MUST MATCH THE BENCHMARKED ONES. The option mapping here is the same
one `bench/jobe_direct.py` uses for JevBench, deliberately duplicated in shape
rather than diverging: if the served answer differs from the scored answer, the
placement number stops describing the thing being served.

Order averaging is available and OFF by default, because it was measured and it
does not pay - over the 231 public tasks the declared order scores 186, the
reversed 181, and the average 185, for twice the forward passes
(`bench/RESULTS.md`).

    python -m jobe.server --model D:/Coding/models/qwen35-4b --port 8901
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import load, pick_device
from .prompt import Decision, DecisionError, Option
from .readout import score
from .slots import MAX_OPTIONS, SlotError
from .wide import score_wide

STATE: dict = {"backbone": None, "ledger": None, "lock": threading.Lock(),
               "gpu": threading.Lock(), "wide": False, "max_tokens": 4096, "n": 0}


# --------------------------------------------------------------------- guards


def free_vram_mb() -> int | None:
    """What nvidia-smi says, not what torch says.

    TheLab's playbook is explicit about the difference: on this Windows box the
    torch view read 1.5 GB while nvidia-smi read 9.9 GB at 100% utilisation.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (int(x) for x in out.replace(",", " ").split())
        return total - used
    except Exception:
        return None


def kernel_path(model) -> dict:
    """Which implementation actually bound, recorded on every answer.

    Playbook rule, paid for the hard way: a number measured on the reference
    PyTorch path is not comparable to one measured on the Triton path, and
    nothing in the output tells you which ran. Hopper's server refuses to start
    on the slow path; this one records it and says so loudly.
    """
    out = {}
    try:
        mod = sys.modules.get(type(model).__module__)
        for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
            fn = getattr(mod, name, None)
            cells = {}
            if fn is not None and getattr(fn, "__closure__", None):
                cells = dict(zip(fn.__code__.co_freevars,
                                 [c.cell_contents for c in fn.__closure__]))
            impl = cells.get("implementation")
            out[name] = getattr(impl, "__module__", "unknown") if impl is not None else "unknown"
    except Exception:
        pass
    return out


# ------------------------------------------------------------------- mapping


def options_for(question: dict) -> list[Option]:
    """A question's declared options, in the id: description shape.

    Showing the id as well as the description is not cosmetic - our own
    conformance run scored 0.779 while the id was hidden and 0.805 once the
    mapping became `<id>: <description>`, which is six hard items.
    """
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        c = criteria or {}
        pairs = [(k, c.get(k) or "The proposition is %s." % k) for k in ("true", "false")]
    elif kind == "choice":
        pairs = [(str(k), v or str(k)) for k, v in (criteria or {}).items()]
    elif kind == "score":
        pairs = [(str(i), lvl) for i, lvl in enumerate(criteria or [])]
    else:
        raise DecisionError("unknown question type %r; expected choice, noul or score" % kind)
    return [Option(id=k, description="%s: %s" % (k, v)) for k, v in pairs]


def _brief(entry: dict) -> str:
    """One element as jev-browser itself describes it (page-model.mjs `brief`)."""
    name = (entry.get("label") or entry.get("text") or entry.get("placeholder")
            or entry.get("name") or entry.get("near") or entry.get("href") or "")
    return '%s "%s"' % (entry.get("tag", "?"), str(name)[:50])


def referents(state, question) -> dict[str, str]:
    """id -> description, for a choice whose options are bare indices into the state.

    jev-browser asks "which entry of `page.elements` (by its `i`)" and sends the
    criteria as `{"4": null}`: the option is an index with no description, and
    the element it names lives in the state. A model that reads all 255 options
    at once can cross-reference that. The letter readout cannot, because it
    answers with a LETTER and the option text is the only thing binding that
    letter to an element. Measured on DuckDuckGo: the search box was offered as
    "4: 4", indistinguishable from the other 156 elements, and the narrowing
    tree above it summarised its groups as "0: 0 | 1: 1 | 2: 2 ...". "search
    nike" failed with the search box absent from the target's top five.

    JevBench never reaches this - every JevBench option carries a real
    description - so this only ever fires on the browser loop's index questions.
    """
    if not isinstance(state, dict):
        return {}
    page = state.get("page") if isinstance(state.get("page"), dict) else state
    wants_groups = "page.groups" in (question.get("instructions") or "")
    out: dict[str, str] = {}
    if wants_groups:
        for g in page.get("groups") or []:
            if isinstance(g, dict) and "g" in g:
                out[str(g["g"])] = "group: " + str(g.get("summary", ""))[:160]
    else:
        for e in page.get("elements") or []:
            if isinstance(e, dict) and "i" in e:
                out[str(e["i"])] = _brief(e)
    return out


def describe_indices(options: list[Option], question: dict, state) -> list[Option]:
    """Give index-only options the description of the entry they index.

    Only options the caller left EMPTY are touched, so anything with a real
    description - every JevBench option - passes through byte-identical.
    """
    criteria = question.get("criteria")
    if question.get("type") != "choice" or not isinstance(criteria, dict):
        return options
    empty = {str(k) for k, v in criteria.items() if not v}
    if not empty:
        return options
    refs = referents(state, question)
    if not refs:
        return options
    return [Option(id=o.id, description="%s: %s" % (o.id, refs[o.id]))
            if o.id in empty and o.id in refs else o for o in options]


def answer_for(backbone, state, question, key: str, scorer=None) -> dict:
    """One question, one forward pass, in the wire format the callers expect.

    With `--wide`, a question too wide for the answer slots is narrowed instead
    of refused. Narrowing is off by default on purpose: the 422 is a DECLARED
    limit that callers can rely on, and quietly answering it with an estimator
    that has not been validated against the flat readout would be worse than
    refusing. When it is on, every narrowed answer says so in `_meta`.
    """
    options = describe_indices(options_for(question), question, state)
    if len(options) == 1:
        # The caller declared exactly one option. The protocol needs two - there
        # is nothing to read a distribution over - but the ANSWER is not in
        # doubt, and 422-ing it fails a whole run over a degenerate case. Real
        # callers hit this constantly: jev-browser asks which of `task.values`
        # to type, and a goal with a single value has one option.
        only = options[0]
        return {"type": question["type"], "choice": only.id,
                "probabilities": {only.id: 1.0}, "confidence": 1.0,
                "_meta": {"input_tokens": 0, "n_options": 1, "narrowed": False,
                          "degenerate": True}}
    decision = Decision(
        id=key, evidence=state, criterion=question.get("instructions", ""),
        options=tuple(options), ordinal=question.get("type") == "score")
    limit = STATE["max_tokens"]
    with STATE["gpu"]:
        readout = (score_wide(backbone.model, backbone.tokenizer, decision,
                              scorer=scorer, max_tokens=limit)
                   if STATE["wide"] else
                   score(backbone.model, backbone.tokenizer, decision, max_tokens=limit))
    probs = dict(readout.scores)
    kind = question["type"]
    if kind == "noul":
        answer = {"type": "noul", "noul": probs.get("true", 0.0)}
    elif kind == "choice":
        answer = {"type": "choice", "choice": readout.choice, "probabilities": probs}
    else:
        answer = {"type": "score", "probabilities": probs}
    answer["confidence"] = readout.confidence()
    answer["_meta"] = {
        "input_tokens": readout.input_tokens,
        "n_options": len(options),
        "narrowed": getattr(readout, "flat", True) is False,
    }
    if answer["_meta"]["narrowed"]:
        # Say which options were never examined individually. Their probability
        # is a shared-out group mass, so ranking them against each other is not
        # informative and a caller walking the ranking should know.
        answer["_meta"].update(passes=readout.passes, depth=readout.depth,
                               estimated=sorted(readout.estimated))
    return answer


def smoke(backbone) -> float:
    """One real decision before announcing ready.

    Two jobs. It turns a backbone that cannot run here at all into a refusal to
    start rather than a 500 on somebody's first request - this Qwen3.5 build
    binds fla's Triton delta-rule kernel, which dies on a CPU tensor, and the
    only symptom was a per-request error. And it pays the kernel warm-up here
    instead of inside a caller's first decision; in the browser demo that first
    call cost 703 seconds.
    """
    t0 = time.perf_counter()
    readout = score(backbone.model, backbone.tokenizer, Decision(
        id="smoke", evidence="A service started returning HTTP 500 after a deploy.",
        criterion="Is this an incident?",
        options=(Option("true", "true: it is an incident"),
                 Option("false", "false: it is not an incident"))))
    if abs(sum(readout.probabilities) - 1.0) > 1e-6:
        raise RuntimeError("smoke decision returned an unnormalised distribution")
    return time.perf_counter() - t0


# -------------------------------------------------------------------- ledger


def record(row: dict) -> None:
    """Every decision, including the failures, appended from exactly one place.

    A sibling project printed its escalations without recording them, and its
    promotion gate then read off a ledger that under-reported. One writer, and
    it writes on the error paths too.
    """
    path = STATE["ledger"]
    if not path:
        return
    with STATE["lock"]:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


class Handler(BaseHTTPRequestHandler):
    server_version = "jobe/0.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        bb = STATE["backbone"]
        if self.path.rstrip("/") in ("/health", ""):
            self._json(200, {
                "ok": bb is not None, "model": getattr(bb, "name", None),
                "device": getattr(bb, "device", None), "adapter": getattr(bb, "adapter", None),
                "decisions_served": STATE["n"],
                "wide": STATE["wide"], "max_options": MAX_OPTIONS,
                "max_tokens": STATE["max_tokens"],
                "kernels": kernel_path(bb.model) if bb else {},
                "free_vram_mb": free_vram_mb(),
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/systemone":
            self._json(404, {"error": "not found"})
            return
        started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            record({"ts": time.time(), "status": 400, "error": str(exc)[:200]})
            self._json(400, {"error": "bad request: %s" % exc})
            return

        bb, state = STATE["backbone"], body.get("state")
        questions = body.get("questions") or {}
        # Every question in a request is about the SAME state, which is the case
        # the prefix cache exists for: encode the evidence once and score each
        # question as a suffix off it. jev-browser asks ten per round over a
        # ~2,900-token page, so re-encoding it each time is the dominant cost.
        scorer = None
        if STATE["wide"] and bb is not None:
            from .prefix import PrefixScorer
            # The limit has to be passed HERE too. PrefixScorer keeps its own
            # copy and defaults to 4096, so a server started with --max-tokens
            # 8192 still refused a 6,449-token page with
            # "PrefixError: prefix of 6449 tokens exceeds the limit 4096" -
            # the flag appeared to do nothing, which is worse than not having it.
            scorer = PrefixScorer(bb.model, bb.tokenizer,
                                  max_tokens=STATE["max_tokens"])
        answers, tokens = {}, 0
        for key, question in questions.items():
            t0 = time.perf_counter()
            try:
                answer = answer_for(bb, state, question, key, scorer=scorer)
            except (DecisionError, SlotError) as exc:
                # A declared limit is 422, not 500. Three consecutive 500s end a
                # run on some harnesses, and a cap reported as a crash is
                # indistinguishable from one.
                record({"ts": time.time(), "key": key, "status": 422,
                        "state_sha": digest(state), "error": str(exc)[:200],
                        "n_options": len(question.get("criteria") or [])})
                self._json(422, {"error": str(exc)})
                return
            except Exception as exc:
                record({"ts": time.time(), "key": key, "status": 500,
                        "state_sha": digest(state), "error": "%s: %s" % (type(exc).__name__, exc)})
                self._json(500, {"error": "%s: %s" % (type(exc).__name__, exc)})
                return
            answers[key] = answer
            tokens += answer["_meta"]["input_tokens"]
            STATE["n"] += 1
            record({
                "ts": time.time(), "key": key, "status": 200,
                "state_sha": digest(state), "question": question.get("instructions", "")[:300],
                "type": question.get("type"),
                "options": [o.id for o in options_for(question)],
                "chosen": answer.get("choice") or answer.get("type"),
                "probabilities": answer.get("probabilities") or {"noul": answer.get("noul")},
                "confidence": answer["confidence"],
                "decider": getattr(bb, "name", None),
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": answer["_meta"]["input_tokens"],
                "n_options": answer["_meta"]["n_options"],
                "narrowed": answer["_meta"]["narrowed"],
                "passes": answer["_meta"].get("passes", 1),
                "outcome": None,
            })

        self._json(200, {
            "model": getattr(bb, "name", None),
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "latency_s": time.perf_counter() - started,
        })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.environ.get("JOBE_MODEL", "Qwen/Qwen3.5-4B"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ledger", default=os.environ.get("JOBE_LEDGER", "runs/ledger.jsonl"),
                    help="one appended record per decision; empty string disables")
    ap.add_argument("--wide", action="store_true",
                    help="narrow choices wider than the %d answer slots instead of "
                         "refusing them with 422. The narrowed distribution is an "
                         "estimator that has NOT been validated against the flat "
                         "readout; every narrowed answer is flagged in _meta."
                         % MAX_OPTIONS)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="refuse prompts longer than this rather than truncating. The "
                         "4096 default is this project's, not the model's: Qwen3.5 takes "
                         "262144 and jev-browser sends states up to ~32768, so a real web "
                         "page routinely blows 4096. Raising it costs VRAM for the KV "
                         "cache, which on a full card is the thing that OOMs.")
    ap.add_argument("--min-free-mb", type=int, default=9000,
                    help="refuse to start on a card this full; 0 disables")
    args = ap.parse_args(argv)

    # The guard is about VRAM, so it only means anything on CUDA. Applying it
    # on CPU would refuse to start over a card the run never touches.
    device = pick_device(args.device)
    free = free_vram_mb() if device == "cuda" else None
    if args.min_free_mb and free is not None and free < args.min_free_mb:
        raise SystemExit(
            "%d MiB free; this backbone needs about %d and will spill, which makes every "
            "latency this server reports meaningless. Free the card or pass --min-free-mb 0."
            % (free, args.min_free_mb))

    STATE["wide"] = args.wide
    STATE["max_tokens"] = args.max_tokens
    STATE["ledger"] = args.ledger or None
    if STATE["ledger"]:
        os.makedirs(os.path.dirname(os.path.abspath(STATE["ledger"])) or ".", exist_ok=True)

    print("loading %s ..." % args.model, flush=True)
    STATE["backbone"] = load(args.model, device=args.device)
    kernels = kernel_path(STATE["backbone"].model)
    for name, impl in kernels.items():
        fast = impl not in ("unknown",) and not impl.startswith("transformers.")
        print("  %-32s %s%s" % (name, impl, "" if fast else "   <- REFERENCE PATH, slow"),
              flush=True)
    try:
        warm = smoke(STATE["backbone"])
    except Exception as exc:  # noqa: BLE001 - any failure here means it cannot serve
        raise SystemExit(
            "the backbone loaded but could not decide - %s: %s\n"
            "On --device cpu this is expected for Qwen3.5: it binds fla's Triton "
            "delta-rule kernel, which cannot take a CPU tensor." % (type(exc).__name__, exc))
    print("  smoke decision ok in %.2fs; kernels warm" % warm, flush=True)
    print("ready on http://%s:%d/v1/systemone   device: %s   wide: %s   ledger: %s"
          % (args.host, args.port, device, "on (narrowed answers are estimates)"
             if args.wide else "off (>%d options -> 422)" % MAX_OPTIONS,
             STATE["ledger"] or "off"), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
