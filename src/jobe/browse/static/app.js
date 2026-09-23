// Jobe chat window. Streams events from the local server and renders each decision.
"use strict";

const TOKEN = window.JOBE_TOKEN;
const $ = (s) => document.querySelector(s);
const feed = $("#feed"), input = $("#input"), send = $("#send"), stop = $("#stop");
const statusPill = $("#status"), pin = $("#pin"), empty = $("#empty");
const brainBtn = $("#brain"), brainName = $("#brainName"), brainMenu = $("#brainMenu");
const brainNote = $("#brainNote"), remoteForm = $("#remoteForm"), remoteModel = $("#remoteModel");
const KEEP_IMAGES = 12;              // older step screenshots are dropped from the DOM
let lastSeq = 0, ready = false, busy = false, lastCard = null, history = [], histIdx = -1;
let brainKey = null;                 // "kind:name" of what answers now

// ------------------------------------------------------------------ helpers
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
// Follow the conversation unless the person has scrolled up to read something.
// Only the person's own input can unstick it. The jump is instant and
// synchronous: a smooth scroll is an animation, and animations do not run in a
// window that is covered or minimised - the log would stop following.
let stick = true;
const nearBottom = () => feed.scrollHeight - feed.scrollTop - feed.clientHeight < 80;
["wheel", "touchmove", "keydown"].forEach((type) => feed.addEventListener(type, () => {
  setTimeout(() => { stick = nearBottom(); }, 120);
}, { passive: true }));
feed.addEventListener("scroll", () => { if (nearBottom()) stick = true; });
function toBottom() { if (stick) feed.scrollTop = feed.scrollHeight; }
function add(node) {
  if (empty && empty.isConnected) empty.remove();
  feed.appendChild(node);
  // screenshots arrive as images that size themselves after decoding
  node.querySelectorAll?.("img").forEach((img) => img.addEventListener("load", toBottom, { once: true }));
  toBottom();
  return node;
}
function pct(p) { return Math.round(p * 100) + "%"; }
function short(line) {
  if (!line) return "";
  const m = line.match(/^\[\d+\]\s+(\S+)\s+"(.*?)"/);
  return m ? m[2] || m[1] : line;
}
function setStatus(phase, text) {
  statusPill.dataset.phase = phase;
  statusPill.querySelector("b").textContent = text;
}
function pruneImages() {
  const imgs = feed.querySelectorAll(".shot");
  for (let i = 0; i < imgs.length - KEEP_IMAGES; i++) imgs[i].remove();
}
async function post(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json", "X-Jobe-Token": TOKEN },
    body: JSON.stringify(body || {}),
  });
  if (r.status === 403) { location.reload(); return new Promise(() => {}); }   // stale token: see connect()
  return r.ok ? r.json() : Promise.reject(await r.text());
}

// ------------------------------------------------------------------ render
function screenshot(ev) {
  if (!ev.screenshot) return null;
  const wrap = el("div", "shot");
  const img = el("img");
  img.alt = ev.title ? "Screenshot of " + ev.title : "Screenshot";
  img.src = "data:image/jpeg;base64," + ev.screenshot;
  wrap.appendChild(img);
  const r = ev.rect, vp = ev.viewport;
  if (r && vp && vp[0] && vp[1]) {
    const box = el("div", "mark-box");
    const pad = 3;
    box.style.left = ((r.x - pad) / vp[0] * 100) + "%";
    box.style.top = ((r.y - pad) / vp[1] * 100) + "%";
    box.style.width = ((r.w + pad * 2) / vp[0] * 100) + "%";
    box.style.height = ((r.h + pad * 2) / vp[1] * 100) + "%";
    wrap.appendChild(box);
  }
  return wrap;
}

function decisionCard(ev) {
  const card = el("article", "card");
  const head = el("div", "card-head");
  const op = el("span", "op", ev.operation.replace("_", " "));
  op.dataset.op = ev.operation;
  head.appendChild(op);
  const tgt = el("span", "target");
  tgt.textContent = ev.target_line ? short(ev.target_line)
    : { DONE: "Goal complete", BLOCKED: "Nothing here can make progress", SUBMIT: "Press Enter",
        WAIT: "Waiting for the page", SCROLL_DOWN: "Scroll down", SCROLL_UP: "Scroll up" }[ev.operation] || "";
  if (ev.target_line) tgt.title = ev.target_line;
  head.appendChild(tgt);
  const t = ev.timings || {};
  head.appendChild(el("span", "ms", Math.round(t.total_ms || 0) + " ms"));
  card.appendChild(head);
  const shot = screenshot(ev);
  if (shot) card.appendChild(shot);
  const probs = el("div", "probs");
  const ranked = Object.entries(ev.op_probs || {}).sort((a, b) => b[1] - a[1]).slice(0, 4);
  ranked.forEach(([k, p], i) => {
    const chip = el("span", "prob" + (k === ev.operation ? " top" : ""), k.replace("_", " ") + " " + pct(p));
    probs.appendChild(chip);
  });
  if (ev.target_line && ev.target_p != null) {
    probs.appendChild(el("span", "prob top", "target " + pct(ev.target_p)));
  }
  card.appendChild(probs);
  return card;
}

function result(status, ev) {
  const map = {
    done: ["ok", "Done"], ready: ["ok", "Browser ready"], read: ["ok", "Read the page"],
    closed: ["ok", "Browser closed"], blocked: ["bad", "Blocked"], stuck: ["warn", "Stuck"],
    stopped: ["warn", "Stopped"], step_budget: ["warn", "Out of steps"],
  };
  const [cls, label] = map[status] || ["warn", status];
  const r = el("div", "result " + cls);
  r.appendChild(el("b", null, label));
  const bits = [];
  if (ev.steps) bits.push(ev.steps + (ev.steps === 1 ? " step" : " steps"));
  if (ev.ms) bits.push((ev.ms / 1000).toFixed(1) + " s");
  if (ev.text) bits.push(ev.text);
  if (bits.length) r.appendChild(el("span", null, "· " + bits.join(" · ")));
  return r;
}

function line(text, cls) {
  const l = el("div", "line" + (cls ? " " + cls : ""));
  l.appendChild(el("span", "dot"));
  const s = el("span");
  s.textContent = text;
  l.appendChild(s);
  return l;
}

// A confirmation is answered by whatever happens next. Its buttons go then: an
// "Allow" left on an old card - a replayed one especially - would send "yes"
// and approve whatever action happens to be paused NOW.
function settleConfirms() {
  feed.querySelectorAll(".card.confirm .actions").forEach((a) => a.remove());
}

function handle(ev) {
  if (ev.seq && ev.seq <= lastSeq) return;
  if (ev.seq) lastSeq = ev.seq;
  if (["user", "decision", "done", "confirm", "error"].includes(ev.kind)) settleConfirms();
  switch (ev.kind) {
    case "status":
      if (ev.phase === "loading") { setStatus("loading", "Loading"); }
      else if (ev.phase === "ready") {
        ready = true; setStatus("ready", "Ready"); send.disabled = !input.value.trim();
        if (ev.model) statusPill.title = "Model: " + ev.model;
        if (ev.brain) {
          const key = ev.brain.kind + ":" + ev.brain.name;
          if (brainKey && key !== brainKey) {
            add(line("Now answering: " + (ev.brain.kind === "remote" ? ev.brain.name + " on OpenRouter"
                                                                        : "Jobe on this PC")));
          }
          brainKey = key;
          showBrain(ev.brain);
        }
      }
      else if (ev.text) add(line(ev.text));
      break;
    case "busy":
      busy = !!ev.busy;
      stop.hidden = !busy;
      setStatus(busy ? "busy" : (ready ? "ready" : "loading"), busy ? "Working" : (ready ? "Ready" : "Loading"));
      break;
    case "user":
      add(el("div", "msg-user", ev.text));
      break;
    case "intent":
      if (ev.intent !== "act") add(line("Understood as “" + ev.intent + "” · " + pct(ev.probabilities[ev.intent] || 0)));
      break;
    case "page": {
      const card = el("article", "card");
      const head = el("div", "card-head");
      head.appendChild(el("span", "op", "PAGE"));
      head.appendChild(el("span", "target", ev.title || ev.url || ""));
      card.appendChild(head);
      const shot = screenshot(ev);
      if (shot) card.appendChild(shot);
      if (ev.text) card.appendChild(el("div", "card-note", ev.text));
      add(card); pruneImages();
      break;
    }
    case "decision":
      lastCard = add(decisionCard(ev)); pruneImages();
      break;
    case "typing": {
      const l = line("Typing ", "typing");
      const c = el("code", null, ev.text);
      l.lastChild.appendChild(c);
      add(l);
      break;
    }
    case "acted":
      if (lastCard && ev.change && ev.change !== "nothing visible changed") {
        lastCard.appendChild(el("div", "card-note", "→ " + ev.change));
      } else if (lastCard && ev.changed === false) {
        lastCard.appendChild(el("div", "card-note", "→ nothing visible changed"));
      }
      break;
    case "note":
      add(line(ev.text));
      break;
    case "confirm": {
      const card = el("article", "card confirm");
      card.appendChild(el("div", "card-head", "Needs your OK"));
      card.appendChild(el("div", "card-body", ev.what + " — " + ev.text));
      const actions = el("div", "actions");
      const yes = el("button", "btn primary", "Allow");
      const no = el("button", "btn", "Cancel");
      yes.onclick = () => { say("yes"); actions.remove(); };
      no.onclick = () => { say("cancel"); actions.remove(); };
      actions.append(yes, no);
      card.appendChild(actions);
      add(card);
      break;
    }
    case "done":
      add(result(ev.status, ev));
      break;
    case "error":
      add(el("div", "error", ev.text));
      if (!ready) setStatus("error", "Error");
      break;
  }
}

// ------------------------------------------------------------------ stream
async function connect() {
  for (let delay = 400; ; delay = Math.min(delay * 2, 5000)) {
    try {
      const r = await fetch("/events", { headers: { "X-Jobe-Token": TOKEN } });
      // A restarted server mints a new token; this page still holds the old one.
      // Reloading fetches the page - and the token - again.
      if (r.status === 403) { setStatus("loading", "Restarting"); setTimeout(() => location.reload(), 800); return; }
      if (!r.ok || !r.body) throw new Error("status " + r.status);
      delay = 400;
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
          const data = chunk.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("");
          if (data) { try { handle(JSON.parse(data)); } catch (e) { console.error(e); } }
        }
      }
    } catch (e) {
      setStatus("error", "Reconnecting");
    }
    await new Promise((res) => setTimeout(res, delay));
  }
}

// ------------------------------------------------------------------ input
async function say(text) {
  text = (text || "").trim();
  if (!text) return;
  if (!ready) { add(line("Still loading the model - one moment, then try again.")); return; }
  history.push(text); histIdx = -1;
  try { await post("/say", { text }); }
  catch (e) { add(el("div", "error", "Could not send: " + e)); }
}

$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = ""; send.disabled = true;
  say(text);
});
input.addEventListener("input", () => { send.disabled = !ready || !input.value.trim(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "ArrowUp" && history.length) {
    histIdx = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1);
    input.value = history[histIdx]; send.disabled = !ready;
    e.preventDefault();
  } else if (e.key === "Escape" && busy) {
    post("/stop");
  }
});
stop.addEventListener("click", () => post("/stop"));
document.querySelectorAll("[data-say]").forEach((b) => b.addEventListener("click", () => say(b.dataset.say)));
pin.addEventListener("click", async () => {
  const on = pin.getAttribute("aria-pressed") !== "true";
  try {
    await post("/pin", { on });
    pin.setAttribute("aria-pressed", String(on));
    pin.title = on ? "Keep on top" : "Not on top";
  } catch (e) { /* ignore */ }
});

// ------------------------------------------------------------------ brain
function showBrain(b) {
  brainBtn.dataset.kind = b.kind;
  brainName.textContent = b.kind === "remote" ? b.name.split("/").pop() : "Jobe";
  brainBtn.title = b.kind === "remote" ? "Answering: " + b.name + " on OpenRouter"
                                       : "Answering: Jobe (Qwen3.5-4B) on this PC";
}
function note(text, bad) { brainNote.textContent = text || ""; brainNote.classList.toggle("bad", !!bad); }
function errorText(e) { try { return JSON.parse(e).error || String(e); } catch (_) { return String(e); } }
function closeMenu() { brainMenu.hidden = true; brainBtn.setAttribute("aria-expanded", "false"); }

brainBtn.addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!brainMenu.hidden) return closeMenu();
  brainMenu.hidden = false; brainBtn.setAttribute("aria-expanded", "true");
  try {
    const h = await (await fetch("/health")).json();
    const off = !h.remote_available;
    remoteModel.disabled = off; remoteForm.querySelector("button").disabled = off;
    note(off ? "Set OPENROUTER_API_KEY as a user environment variable to use a hosted model."
             : "A hosted model sees every page Jobe looks at.");
  } catch (_) { note(""); }
});
brainMenu.addEventListener("click", (e) => e.stopPropagation());
document.addEventListener("click", closeMenu);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

brainMenu.querySelector('[data-brain="local"]').addEventListener("click", async () => {
  closeMenu();
  if (brainBtn.dataset.kind === "local") return;          // already answering
  add(line("Switching to Jobe on this PC…"));              // before the post: the switch can beat its reply
  try { await post("/brain", { model: "local" }); }
  catch (e) { add(el("div", "error", "Could not switch: " + errorText(e))); }
});
remoteForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const model = remoteModel.value.trim();
  if (!model) return;
  note("Checking " + model + " with one real decision…");
  try {
    await post("/brain", { model });
    closeMenu(); note("");
    // Idle, the switch lands at once and says "Now answering: ..." itself;
    // mid-goal it waits, so say that it is coming.
    if (busy) add(line(model + " passed a test decision - switching after this goal."));
  } catch (err) { note(errorText(err), true); }
});

connect();
input.focus();
