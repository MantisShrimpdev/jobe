# -*- coding: utf-8 -*-
"""Q2 + Q3: "reason only when unsure" - a thinking pass over the readout's
LEAST-CONFIDENT hard tasks, then the routing curve.

The routed set is exactly what a confidence router would send: the lowest-
confidence FRAC of the hard tier by the readout's own top probability
(official-harness results). Each routed task gets one thinking pass with the
model card's thinking-mode recipe (t=1.0, top_p=0.95, top_k=20,
presence_penalty=1.5). Two shapes:

  sampled   : think freely up to --cap tokens; answer parsed after </think>.
              If it never closes, the router FALLS BACK to the readout answer.
  budgeted  : think up to --budget tokens, then force </think> and read a
              greedy answer. Bounded latency by construction.

Rows are appended to --out as they finish (resume-safe: done ids are skipped).
Aborts if another process takes the GPU (the user shares this card).

  python bench/reason_route.py --tasks <jevbench>/datasets/public/hard.jsonl --official bench/runs/2026-09-22-public231/results.jsonl --mode budgeted --budget 512 --frac 0.3 --out route_b512.jsonl
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

from thelab.core.gpu import require_free_gpu, used_by_others_gb  # noqa: E402

from jobe import Decision, Option, build_messages, load  # noqa: E402
from jobe.slots import LETTERS  # noqa: E402

READOUT_MS = 1.0  # fallback only; the real per-task latency_s comes from the official results


class PresencePenalty(LogitsProcessor):
    """logits[t] -= penalty for every t already generated (output tokens only)."""

    def __init__(self, penalty, prompt_len):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1]:
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
            scores = scores - mask.to(scores.dtype) * self.penalty
        return scores


def options_for(task):
    q = task["question"]; crit = q.get("criteria")
    if q["type"] == "noul":
        c = crit or {}; pairs = [(k, c.get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif q["type"] == "choice":
        pairs = [(k, v or k) for k, v in (crit or {}).items()]
    else:
        pairs = [(str(i), lvl) for i, lvl in enumerate(crit or [])]
    return [Option(id=k, description=f"{k}: {v}") for k, v in pairs]


def canonical(kind, oid):
    return ("yes" if oid == "true" else "no") if kind == "noul" else oid


LETTER_RE = re.compile(r"\b([A-P])\b")


def parse(kind, opts, tail):
    m = LETTER_RE.search(tail)
    if m and LETTERS.index(m.group(1)) < len(opts):
        return canonical(kind, opts[LETTERS.index(m.group(1))].id)
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True, help="jevbench datasets/public/hard.jsonl")
ap.add_argument("--official", required=True, help="results.jsonl from the official-harness run")
ap.add_argument("--mode", choices=["sampled", "budgeted"], required=True)
ap.add_argument("--cap", type=int, default=2048)
ap.add_argument("--budget", type=int, default=512)
ap.add_argument("--frac", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--model", required=True, help="local path or hub id of the backbone")
ap.add_argument("--out", required=True)
ap.add_argument("--report-only", action="store_true")
args = ap.parse_args()

# ---- routed set: least-confident FRAC of hard, by the readout's own top prob ----
official = [json.loads(l) for l in open(args.official, encoding="utf-8") if l.strip()]
hard_rows = [r for r in official if r["task_id"].startswith("hard-") and r.get("ok") and r.get("probs")]
hard_rows.sort(key=lambda r: max(r["probs"].values()))
n_hard = len(hard_rows)
k_route = round(n_hard * args.frac)
routed_rows = hard_rows[:k_route]
tasks = {json.loads(l)["id"]: json.loads(l) for l in open(args.tasks, encoding="utf-8") if l.strip()}
by_id = {r["task_id"]: r for r in hard_rows}

done = {}
if os.path.exists(args.out):
    for l in open(args.out, encoding="utf-8"):
        if l.strip():
            row = json.loads(l); done[row["task_id"]] = row

todo = [r for r in routed_rows if r["task_id"] not in done]
if args.limit:
    todo = todo[: args.limit]
cutoff = max(routed_rows[-1]['probs'].values()) if routed_rows else float('nan')
print(f"hard n={n_hard}  routed={k_route} (conf <= {cutoff:.3f})  "
      f"done={len(done)}  todo={len(todo)}  mode={args.mode} cap={args.cap} budget={args.budget}")

if todo and not args.report_only:
    require_free_gpu(3000)
    bb = load(args.model, device="auto", attn_implementation=args.attn)
    tok, model, dev = bb.tokenizer, bb.model, next(bb.model.parameters()).device
    pad = tok.pad_token_id or tok.eos_token_id
    print(f"attention: {bb.attn_implementation}")
    fh = open(args.out, "a", encoding="utf-8")
    hdr = f"{'task':30s} {'exp':>8s} {'read':>8s} {'reason':>8s} {'ok':>3s} {'term':>4s} {'new':>5s} {'sec':>6s}"
    print(hdr)
    for i, r in enumerate(todo):
        if i and used_by_others_gb() > 3.0:
            print(f"another process took the GPU ({used_by_others_gb():.1f} GB) - stopping after {i} tasks; rerun to resume")
            break
        task = tasks[r["task_id"]]
        kind = task["question"]["type"]
        opts = options_for(task)
        d = Decision(id=task["id"], evidence=task["state"], criterion=task["question"]["instructions"],
                     options=tuple(opts), ordinal=(kind == "score"))
        prompt = tok.apply_chat_template(build_messages(d), tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        torch.manual_seed(0)
        t0 = time.perf_counter()
        max_new = args.cap if args.mode == "sampled" else args.budget
        with torch.inference_mode():
            out = model.generate(torch.tensor([ids], device=dev),
                                 attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=dev),
                                 max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                                 logits_processor=LogitsProcessorList([PresencePenalty(1.5, len(ids))]),
                                 pad_token_id=pad)
        new = out[0, len(ids):]
        text = tok.decode(new, skip_special_tokens=False)
        terminated = "</think>" in text
        forced = False
        if terminated:
            tail = text.split("</think>", 1)[1]
            think_tokens = len(tok.encode(text.split("</think>", 1)[0], add_special_tokens=False))
        elif args.mode == "budgeted":
            forced = True
            closed = text + "\n</think>\n\n"
            ids2 = ids + tok.encode(closed, add_special_tokens=False)
            with torch.inference_mode():
                out2 = model.generate(torch.tensor([ids2], device=dev),
                                      attention_mask=torch.ones((1, len(ids2)), dtype=torch.long, device=dev),
                                      max_new_tokens=8, do_sample=False, pad_token_id=pad)
            tail = tok.decode(out2[0, len(ids2):], skip_special_tokens=True)
            think_tokens = int(new.numel())
        else:
            tail = ""
            think_tokens = int(new.numel())
        pred = parse(kind, opts, tail)
        if pred is None and terminated and args.mode == "budgeted":
            # closed the think but the letter fell outside the budget: answer from the closed think
            ids2 = ids + tok.encode(text.split("</think>", 1)[0] + "</think>\n\n", add_special_tokens=False)
            with torch.inference_mode():
                out2 = model.generate(torch.tensor([ids2], device=dev),
                                      attention_mask=torch.ones((1, len(ids2)), dtype=torch.long, device=dev),
                                      max_new_tokens=8, do_sample=False, pad_token_id=pad)
            tail = tok.decode(out2[0, len(ids2):], skip_special_tokens=True)
            pred = parse(kind, opts, tail)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        exp = str(task["expected"])
        row = dict(task_id=task["id"], family=task["family"], type=kind, expected=exp,
                   readout_pred=r["predicted"], readout_correct=bool(r["correct"]),
                   readout_conf=max(r["probs"].values()),
                   reason_pred=pred, reason_correct=(pred is not None and pred == exp),
                   terminated=terminated, forced_close=forced, think_tokens=think_tokens,
                   new_tokens=int(new.numel()), seconds=sec, mode=args.mode,
                   cap=args.cap if args.mode == "sampled" else args.budget)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        done[row["task_id"]] = row
        print(f"{task['id'][:30]:30s} {exp[:8]:>8s} {str(r['predicted'])[:8]:>8s} {str(pred)[:8]:>8s} "
              f"{'Y' if row['reason_correct'] else ('-' if pred is None else 'n'):>3s} "
              f"{'Y' if terminated else ('F' if forced else 'n'):>4s} {int(new.numel()):5d} {sec:6.1f}")
    fh.close()

# ---- report: reasoning vs readout on the routed set, then the routing curve ----
rows = [done[r["task_id"]] for r in routed_rows if r["task_id"] in done]
if not rows:
    sys.exit("nothing to report")
n = len(rows)
print(f"\n=== routed set: {n} least-confident hard tasks (of {n_hard}) ===")
read_acc = sum(r["readout_correct"] for r in rows) / n
reas_acc = sum(r["reason_correct"] for r in rows) / n
fb_acc = sum((r["reason_correct"] if r["reason_pred"] is not None else r["readout_correct"]) for r in rows) / n
print(f"readout acc {read_acc:.3f} | reasoning acc {reas_acc:.3f} | reasoning-with-fallback {fb_acc:.3f}")
print(f"terminated {sum(r['terminated'] for r in rows)}/{n}, forced-close {sum(r['forced_close'] for r in rows)}, "
      f"no-answer {sum(r['reason_pred'] is None for r in rows)}")
print(f"mean {sum(r['seconds'] for r in rows)/n:.1f} s/task, mean {sum(r['new_tokens'] for r in rows)/n:.0f} new tokens")
fixed = sum(1 for r in rows if not r["readout_correct"] and r["reason_correct"])
broke = sum(1 for r in rows if r["readout_correct"] and r["reason_pred"] is not None and not r["reason_correct"])
print(f"fixed {fixed} wrong -> right; broke {broke} right -> wrong")
by_fam = {}
for r in rows:
    f = by_fam.setdefault(r["family"], [0, 0, 0])
    f[0] += 1; f[1] += r["readout_correct"]; f[2] += (r["reason_correct"] if r["reason_pred"] is not None else r["readout_correct"])
print("by family (n, readout, routed):", {k: (v[0], v[1], v[2]) for k, v in sorted(by_fam.items())})

# routing curve over the full hard tier: route the least-confident X%, readout for the rest
total_correct_readout = sum(1 for r in hard_rows if r["correct"])
ran = {r["task_id"] for r in rows}
mean_reason_s = sum(r["seconds"] for r in rows) / n
lat_read = [r.get("latency_s") or 0 for r in hard_rows]
read_s = (sum(lat_read) / len(lat_read)) if any(lat_read) else READOUT_MS
print(f"\n=== routing curve (hard tier, n={n_hard}; readout alone {total_correct_readout/n_hard:.3f}) ===")
print(f"{'route':>6s} {'n':>4s} {'acc':>6s} {'delta':>7s} {'mean s/task':>12s}  (readout {read_s:.2f}s, reason {mean_reason_s:.1f}s)")
for pct in (10, 20, 30, 40, 50):
    k = round(n_hard * pct / 100)
    if k > k_route:
        break
    sub = hard_rows[:k]
    if not all(r["task_id"] in ran for r in sub):
        print(f"{pct:5d}%  incomplete"); continue
    correct = total_correct_readout
    for r in sub:
        d = done[r["task_id"]]
        routed_ok = d["reason_correct"] if d["reason_pred"] is not None else d["readout_correct"]
        correct += int(routed_ok) - int(r["correct"])
    acc = correct / n_hard
    lat = read_s + (k / n_hard) * mean_reason_s
    print(f"{pct:5d}% {k:4d} {acc:6.3f} {acc - total_correct_readout/n_hard:+7.3f} {lat:12.1f}")
