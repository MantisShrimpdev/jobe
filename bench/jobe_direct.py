"""In-process JevBench adapter for Jobe: a frozen-backbone decision readout.

Drop this file into `jevbench/adapters/` and register it in `cli.py` (see
bench/README-submission.md). Uses Jobe's own code path — `jobe.load` and
`jobe.score` — on the declared backbone. One forward pass per decision, no
generation; the distribution is the restricted softmax over the declared
answer-letter logits, read from the FULL vocabulary. `probs_source` is
"native": these are the model's own logits, not verbalised probabilities and
not an API's top-N logprob window.

Mapping from the canonical record mirrors `semif_direct.py`, deliberately, so
the numbers are comparable in shape to how SemIf itself was mapped:

  noul   -> options "true"/"false" (that order); result remapped to "yes"/"no"
  choice -> one option per criteria key
  score  -> options "0".."k-1", description = level text
  every description prefixed "<id>: "

FAILURE POLICY. The runner aborts a whole run after three consecutive
failures, but exempts `status_code == 422`. So:

  * out-of-contract tasks — more than 16 options, a tokenizer that cannot
    support the answer-slot contract, a prompt over the token limit — return
    ok=False, status=422. They are recorded as unprocessable, not skipped, and
    do not count toward the abort.
  * anything else (CUDA OOM, a driver fault) returns ok=False with no status,
    which DOES count toward the abort — because three of those in a row is an
    infrastructure problem and stopping is the right call.

Nothing here raises past `run()`.
"""

from __future__ import annotations

import time

from .base import DecisionResult

JOBE_PROMPT_VERSION = "jobe-options-v1"


class JobeDirectAdapter:
    name = "jobe_direct"
    cost_basis = "self_hosted_gpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None,
                 max_tokens=4096, device="auto"):
        # `endpoint` is the backbone: a HuggingFace id or a local path.
        self.endpoint = endpoint or "Qwen/Qwen3.5-4B"
        self.model = model or self.endpoint
        self.revision = revision
        self.max_tokens = max_tokens
        self.device = device
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self._backbone = None
        self.load_s = None

    # ------------------------------------------------------------- lifecycle

    def load(self):
        """Load the frozen backbone once. Called before the clock under
        JEVBENCH_WARM_LOAD=1; otherwise lazily on the first decision."""
        if self._backbone is None:
            from jobe import load
            t0 = time.perf_counter()
            self._backbone = load(self.endpoint, revision=self.revision, device=self.device)
            self.load_s = time.perf_counter() - t0
        return self._backbone

    def reserve_estimate(self, task) -> float:
        return 0.0

    # --------------------------------------------------------------- mapping

    @staticmethod
    def build_decision(task):
        from jobe import Decision, Option

        qtype = task.question["type"]
        crit = task.question.get("criteria")
        if qtype == "noul":
            c = crit or {}
            pairs = [(k, c.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
        elif qtype == "choice":
            pairs = [(k, v or k) for k, v in (crit or {}).items()]
        else:
            pairs = [(str(i), lvl) for i, lvl in enumerate(crit or [])]
        return Decision(
            id=task.id,
            evidence=task.state,
            criterion=task.question["instructions"],
            options=tuple(Option(id=k, description=f"{k}: {v}") for k, v in pairs),
            ordinal=qtype == "score",
        )

    # ------------------------------------------------------------------- run

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        qtype = task.question["type"]

        # 1. Map. A malformed or over-wide task is out of contract: 422.
        try:
            decision = self.build_decision(task)
            res.request_body = {
                "id": decision.id,
                "criterion": decision.criterion,
                "options": [{"id": o.id, "description": o.description} for o in decision.options],
                "prompt_version": JOBE_PROMPT_VERSION,
            }
            decision.validate()
        except Exception as e:  # noqa: BLE001 - DecisionError, or a broken record
            res.status = 422
            res.error = f"unprocessable: {type(e).__name__}: {str(e)[:250]}"
            return res

        # 2. Load. A failure here is infrastructure, so it counts.
        try:
            backbone = self.load()
        except Exception as e:  # noqa: BLE001
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res

        # 3. Score. Contract failures are 422; everything else counts.
        from jobe import score
        from jobe.slots import SlotError

        t0 = time.perf_counter()
        try:
            out = score(backbone.model, backbone.tokenizer, decision, max_tokens=self.max_tokens)
        except (SlotError, ValueError) as e:
            # SlotError: the tokenizer cannot support single-token answer slots
            # for this prompt. ValueError: the prompt exceeds max_tokens (no
            # truncation is ever performed - dropping evidence changes answers).
            res.latency_s = time.perf_counter() - t0
            res.status = 422
            res.error = f"unprocessable: {type(e).__name__}: {str(e)[:250]}"
            return res
        except Exception as e:  # noqa: BLE001 - OOM, driver, anything real
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:250]}"
            return res
        res.latency_s = time.perf_counter() - t0

        # 4. Report, in the canonical label space.
        scores = out.scores
        if qtype == "noul":
            res.probs = {"yes": scores["true"], "no": scores["false"]}
        else:
            res.probs = dict(scores)
        res.usage = {"input_tokens": out.input_tokens, "output_tokens": 0}
        res.raw = {
            "answer": {
                "option_ids": list(out.option_ids),
                "probabilities": list(out.probabilities),
                "option_logits": list(out.option_logits),
                "input_tokens": out.input_tokens,
                "forward_seconds": out.forward_seconds,
                "prompt_sha256": out.prompt_sha256,
                "prompt_version": out.prompt_version,
            },
            "runtime": {
                "backbone": backbone.name,
                "revision": backbone.revision,
                "device": backbone.device,
                "dtype": backbone.dtype,
                "attn_implementation": backbone.attn_implementation,
                "readout": out.meta.get("readout"),
                "probability_origin": "native-option-logit-softmax",
                "probability_status": out.probability_status,
                "temperature": 1.0,
                "order_averaging": "none",
            },
        }
        res.ok = True
        return res
