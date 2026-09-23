# -*- coding: utf-8 -*-
"""The decision endpoint: transport, mapping, status codes and the ledger.

No GPU here. `jobe.server.score` is replaced by a stub that still calls
`decision.validate()` and still returns a REAL `Readout`, so `choice`, `scores`
and `confidence()` are the shipping implementations and only the forward pass is
fake. That is the seam worth faking: everything this module actually owns is
exercised, and a wrong answer cannot hide behind a stubbed helper.
"""

from __future__ import annotations

import ast
import json
import pathlib
import textwrap
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from jobe import Decision, Option
from jobe.readout import Readout
from jobe import server as srv

HERE = pathlib.Path(__file__).resolve().parents[1]


def fake_score(model, tokenizer, decision, **kw):
    """A forward pass that isn't. Contract checks stay real."""
    decision.validate()
    n = len(decision.options)
    # Deliberately lopsided and not uniform, so an argmax bug cannot pass by
    # coincidence: first option gets the mass, the rest split the remainder.
    head = 0.7
    probs = tuple([head] + [(1.0 - head) / (n - 1)] * (n - 1))
    return Readout(
        decision_id=decision.id,
        option_ids=tuple(o.id for o in decision.options),
        probabilities=probs,
        option_logits=tuple(0.0 for _ in probs),
        input_tokens=11,
        forward_seconds=0.001,
        total_seconds=0.002,
        prompt_sha256="0" * 64,
    )


class FakeBackbone:
    """Stands in for `LoadedModel`. The ledger reads `name` off it, so it has
    one - a record that cannot say which model decided is not a record."""

    name = "test/fake-backbone"
    device = "cpu"
    adapter = None
    model = object()
    tokenizer = object()


@pytest.fixture()
def endpoint(tmp_path, monkeypatch):
    """A real socket, a real handler, a real ledger file."""
    monkeypatch.setattr(srv, "score", fake_score)
    ledger = tmp_path / "ledger.jsonl"
    srv.STATE.update({"backbone": FakeBackbone(), "ledger": str(ledger), "n": 0})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    try:
        yield base, ledger
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def post(base, body):
    req = urllib.request.Request(
        base + "/v1/systemone", method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def ask(question, state="The checkout page returns a 500 when applying a coupon."):
    return {"state": state, "questions": {"q": question}}


# ------------------------------------------------------------ question types


def test_choice_returns_a_declared_id_and_a_full_distribution(endpoint):
    base, _ = endpoint
    status, body = post(base, ask({
        "type": "choice", "instructions": "Which team owns this?",
        "criteria": {"billing": "Payments and refunds", "infra": "Outages and deploys"}}))
    assert status == 200
    answer = body["answers"]["q"]
    assert answer["type"] == "choice"
    assert answer["choice"] in ("billing", "infra")
    assert set(answer["probabilities"]) == {"billing", "infra"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-9
    assert body["usage"] == {"input_tokens": 11, "output_tokens": 0}


def test_noul_reports_p_true_and_not_a_choice(endpoint):
    """The wire format wants a single probability, not an argmax over true/false."""
    base, _ = endpoint
    status, body = post(base, ask({
        "type": "noul", "instructions": "Is this a production outage?"}))
    assert status == 200
    answer = body["answers"]["q"]
    assert answer["type"] == "noul"
    assert "choice" not in answer
    # "true" is presented first, and the stub puts its mass on the first option.
    assert answer["noul"] == pytest.approx(0.7)


def test_score_is_keyed_by_level_index(endpoint):
    base, _ = endpoint
    status, body = post(base, ask({
        "type": "score", "instructions": "How severe is this?",
        "criteria": ["trivial", "minor", "major", "critical"]}))
    assert status == 200
    answer = body["answers"]["q"]
    assert answer["type"] == "score"
    assert list(answer["probabilities"]) == ["0", "1", "2", "3"]


def test_confidence_is_chance_corrected_not_raw_pmax(endpoint):
    """Two options at p=0.7 is a weaker signal than four at p=0.7, and the
    number has to say so or it cannot be thresholded across question shapes."""
    base, _ = endpoint
    _, two = post(base, ask({"type": "noul", "instructions": "Outage?"}))
    _, four = post(base, ask({
        "type": "score", "instructions": "Severity?",
        "criteria": ["a", "b", "c", "d"]}))
    c2 = two["answers"]["q"]["confidence"]
    c4 = four["answers"]["q"]["confidence"]
    assert c2 == pytest.approx((0.7 - 1 / 2) / (1 - 1 / 2))
    assert c4 == pytest.approx((0.7 - 1 / 4) / (1 - 1 / 4))
    assert c4 > c2


def test_several_questions_in_one_request(endpoint):
    base, _ = endpoint
    status, body = post(base, {"state": "x", "questions": {
        "a": {"type": "noul", "instructions": "Outage?"},
        "b": {"type": "choice", "instructions": "Owner?",
              "criteria": {"billing": "Payments", "infra": "Deploys"}}}})
    assert status == 200
    assert set(body["answers"]) == {"a", "b"}
    assert body["usage"]["input_tokens"] == 22


# --------------------------------------------------------------- refusals


def test_over_the_letter_cap_is_422_not_500(endpoint):
    """The 16-option ceiling is a declared limit of the letter protocol. A
    harness that aborts after three consecutive 500s must not be told a
    contract refusal is a crash."""
    base, _ = endpoint
    status, body = post(base, ask({
        "type": "choice", "instructions": "Pick one.",
        "criteria": {"opt%02d" % i: "option %d" % i for i in range(17)}}))
    assert status == 422
    assert "17" in body["error"]


def test_unknown_question_type_is_422(endpoint):
    base, _ = endpoint
    status, body = post(base, ask({"type": "vibes", "instructions": "?"}))
    assert status == 422
    assert "vibes" in body["error"]


def test_malformed_body_is_400(endpoint):
    base, _ = endpoint
    req = urllib.request.Request(base + "/v1/systemone", method="POST",
                                 data=b"{not json", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400


def test_unknown_path_is_404(endpoint):
    base, _ = endpoint
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/v1/nope", timeout=10)
    assert e.value.code == 404


def test_health_reports_readiness(endpoint):
    base, _ = endpoint
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        body = json.loads(r.read())
    assert "decisions_served" in body and "kernels" in body


# ----------------------------------------------------------------- ledger


def test_every_decision_is_recorded_including_the_refusals(endpoint):
    """One writer, and it writes on the error paths too - a ledger that only
    records successes silently under-reports exactly the cases worth auditing."""
    base, ledger = endpoint
    post(base, ask({"type": "noul", "instructions": "Outage?"}))
    post(base, ask({"type": "choice", "instructions": "Pick.",
                    "criteria": {"o%02d" % i: str(i) for i in range(17)}}))
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == [200, 422]
    ok = rows[0]
    assert ok["options"] == ["true", "false"]
    assert ok["confidence"] == pytest.approx((0.7 - 0.5) / 0.5)
    assert ok["latency_ms"] > 0
    assert ok["outcome"] is None          # filled in later by whoever learns it
    assert rows[1]["n_options"] == 17
    # The same evidence must hash the same way, or the ledger cannot be grouped.
    post(base, ask({"type": "noul", "instructions": "Outage?"}))
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["state_sha"] == rows[2]["state_sha"]


# --------------------------------------------- the served answer == the benchmarked answer


def benchmarked_build_decision():
    """Lift `build_decision` out of the JevBench adapter without importing it.

    `bench/jobe_direct.py` does `from .base import DecisionResult`, so it only
    imports inside a jevbench checkout. The mapping itself has no dependencies,
    so parse the file and exec just that function.
    """
    src = (HERE / "bench" / "jobe_direct.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_decision":
            node.decorator_list = []          # drop @staticmethod
            body = textwrap.dedent(ast.get_source_segment(src, node))
            ns: dict = {}
            exec(compile(ast.parse(body), "<jobe_direct>", "exec"), ns)  # noqa: S102
            return ns["build_decision"]
    raise AssertionError("build_decision not found in bench/jobe_direct.py")


class Task:
    def __init__(self, tid, state, question):
        self.id, self.state, self.question = tid, state, question


@pytest.mark.parametrize("question", [
    {"type": "noul", "instructions": "Is this an outage?"},
    {"type": "noul", "instructions": "Is this an outage?",
     "criteria": {"true": "It is down.", "false": "It is up."}},
    {"type": "choice", "instructions": "Who owns it?",
     "criteria": {"billing": "Payments", "infra": "Deploys", "web": "Frontend"}},
    {"type": "score", "instructions": "Severity?",
     "criteria": ["trivial", "minor", "major"]},
])
def test_server_mapping_equals_the_jevbench_adapter_mapping(question):
    """If these diverge, the published placement stops describing what is served.

    That is the whole reason the option text is `"<id>: <description>"` and the
    noul options are ordered true-then-false: those choices were measured, and a
    second copy of the mapping is free to drift away from them silently.
    """
    state = "The checkout page returns a 500 when applying a coupon."
    theirs: Decision = benchmarked_build_decision()(Task("t1", state, question))
    ours = Decision(
        id="t1", evidence=state, criterion=question["instructions"],
        options=tuple(srv.options_for(question)),
        ordinal=question["type"] == "score")
    assert ours.options == theirs.options
    assert ours.ordinal == theirs.ordinal
    assert ours.evidence == theirs.evidence
    assert ours.criterion == theirs.criterion


def test_option_descriptions_carry_the_id_prefix():
    """Hiding the id cost 0.779 vs 0.805 on our own conformance run."""
    options = srv.options_for({"type": "choice", "instructions": "?",
                               "criteria": {"billing": "Payments"}})
    assert options == [Option("billing", "billing: Payments")]


# ------------------------------------------------------------- startup gate


def test_smoke_returns_the_warm_up_cost(monkeypatch):
    monkeypatch.setattr(srv, "score", fake_score)
    assert srv.smoke(FakeBackbone()) >= 0.0


def test_smoke_propagates_a_backbone_that_cannot_decide(monkeypatch):
    """A backbone that cannot run here must refuse to start, not 500 on the
    first request. Running Qwen3.5 on CPU does exactly this: it binds fla's
    Triton delta-rule kernel, which rejects a CPU tensor.
    """
    def triton_on_cpu(*a, **kw):
        raise ValueError("Pointer argument cannot be accessed from Triton (cpu tensor?)")

    monkeypatch.setattr(srv, "score", triton_on_cpu)
    with pytest.raises(ValueError, match="Triton"):
        srv.smoke(FakeBackbone())


def test_smoke_rejects_an_unnormalised_distribution(monkeypatch):
    def broken(model, tokenizer, decision, **kw):
        r = fake_score(model, tokenizer, decision, **kw)
        return Readout(**{**r.__dict__, "probabilities": (0.3, 0.3)})

    monkeypatch.setattr(srv, "score", broken)
    with pytest.raises(RuntimeError, match="unnormalised"):
        srv.smoke(FakeBackbone())
