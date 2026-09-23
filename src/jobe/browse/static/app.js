// Jobe chat window. Streams events from the local server and renders each decision.
"use strict";

const TOKEN = window.JOBE_TOKEN;
const $ = (s) => document.querySelector(s);
const feed = $("#feed"), input = $("#input"), send = $("#send"), stop = $("#stop");
const statusPill = $("#status"), pin = $("#pin"), empty = $("#empty");
const brainBtn = $("#brain"), brainName = $("#brainName"), brainMenu = $("#brainMenu");
const brainNote = $("#brainNote"), remoteForm = $("#remoteForm"), remoteModel = $("#remoteModel");
const bar = $("#bar"), composer = $("#composer"), miniLine = $("#miniLine"), miniBtn = $("#mini");
const whereBtn = $("#where"), whereText = $("#whereText"), suggestRow = $("#suggest");
const micBtn = $("#mic"), quitBtn = $("#quit"), newBtn = $("#newchat"), off = $("#off");
const KEEP_IMAGES = 12;              // older step screenshots are dropped from the DOM
let lastSeq = 0, ready = false, busy = false, lastCard = null, history = [], histIdx = -1;
let brainKey = null;                 // "kind:name" of what answers now
let queued = [];                     // typed before the model was ready; sent the moment it is
let mini = false;                    // the window is a bar
let lastStatus = null;               // how the last request ended, for the suggestions
const where = { url: null, title: null, open: false };   // the browser, as the events tell it

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
  if (empty.isConnected) empty.remove();
  feed.appendChild(node);
  // screenshots arrive as images that size themselves after decoding
  node.querySelectorAll?.("img").forEach((img) => img.addEventListener("load", toBottom, { once: true }));
  toBottom();
  renderSuggest();
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
function setMiniLine(text) { miniLine.textContent = text; miniLine.title = text; }
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
function errorText(e) { try { return JSON.parse(e).error || String(e); } catch (_) { return String(e); } }
function store(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* private window */ } }
function stored(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }

// ------------------------------------------------------------ compact steps
// Only the newest step keeps its screenshot open; older ones fold to one line
// and open again on a click. A long task otherwise fills the window with
// pictures of pages already left behind.
function stepCard(card) {
  card.classList.add("step");
  feed.querySelectorAll(".card.step:not(.collapsed)").forEach((c) => c.classList.add("collapsed"));
  return card;
}
feed.addEventListener("click", (e) => {
  const head = e.target.closest(".card.step .card-head");
  if (head) { head.parentElement.classList.toggle("collapsed"); }
});

// ------------------------------------------------------------ where it is
function host(url) { try { return new URL(url).hostname.replace(/^www\./, ""); } catch (_) { return url; } }
function setWhere(url, title) {
  if (!url || url === "about:blank") return;
  where.url = url; where.open = true;
  if (title) where.title = title;
  whereText.textContent = host(url) + (where.title ? "  ·  " + where.title : "");
  whereBtn.hidden = false;
  renderSuggest();
}
whereBtn.addEventListener("click", () => say("show browser"));

// ------------------------------------------------------------ suggestions
const ENGINE = /(^|\.)(bing|google|duckduckgo|brave|startpage|ecosia|yahoo)\./;
function isResults(url) {
  try {
    const u = new URL(url);
    return ENGINE.test(u.hostname) && ["q", "p", "query"].some((k) => u.searchParams.get(k));
  } catch (_) { return false; }
}
function suggestions() {
  if (!ready || busy) return [];
  if (!where.open) return ["open browser"];
  const out = [];
  if (["stuck", "blocked", "step_budget"].includes(lastStatus) && history.length) out.push("try again");
  if (isResults(where.url)) out.push("open the top one", "open the second one", "go back");
  else out.push("go back", "scroll down", "show browser");
  return out;
}
function renderSuggest() {
  const list = suggestions();
  suggestRow.replaceChildren(...list.map((text) => {
    const b = el("button", null, text);
    b.type = "button";
    b.addEventListener("click", () => say(text === "try again" ? history[history.length - 1] : text,
                                          text === "try again"));
    return b;
  }));
  suggestRow.hidden = !list.length || empty.isConnected;
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

const OP_TEXT = { DONE: "Goal complete", BLOCKED: "Nothing here can make progress", SUBMIT: "Press Enter",
                  WAIT: "Waiting for the page", SCROLL_DOWN: "Scroll down", SCROLL_UP: "Scroll up" };
function decisionCard(ev) {
  const card = el("article", "card");
  const head = el("div", "card-head");
  const op = el("span", "op", ev.operation.replace("_", " "));
  op.dataset.op = ev.operation;
  head.appendChild(op);
  const tgt = el("span", "target");
  tgt.textContent = ev.target_line ? short(ev.target_line) : OP_TEXT[ev.operation] || "";
  if (ev.target_line) tgt.title = ev.target_line;
  head.appendChild(tgt);
  const t = ev.timings || {};
  head.appendChild(el("span", "ms", Math.round(t.total_ms || 0) + " ms"));
  card.appendChild(head);
  const shot = screenshot(ev);
  if (shot) card.appendChild(shot);
  const probs = el("div", "probs");
  Object.entries(ev.op_probs || {}).sort((a, b) => b[1] - a[1]).slice(0, 4).forEach(([k, p]) => {
    probs.appendChild(el("span", "prob" + (k === ev.operation ? " top" : ""), k.replace("_", " ") + " " + pct(p)));
  });
  if (ev.target_line && ev.target_p != null) probs.appendChild(el("span", "prob top", "target " + pct(ev.target_p)));
  card.appendChild(probs);
  return card;
}

const RESULT = {
  done: ["ok", "Done"], ready: ["ok", "Browser ready"], read: ["ok", "Read the page"],
  closed: ["ok", "Browser closed"], blocked: ["bad", "Blocked"], stuck: ["warn", "Stuck"],
  stopped: ["warn", "Stopped"], step_budget: ["warn", "Out of steps"],
};
function result(status, ev) {
  const [cls, label] = RESULT[status] || ["warn", status];
  const r = el("div", "result " + cls);
  r.appendChild(el("b", null, label));
  const bits = [];
  if (ev.steps) bits.push(ev.steps + (ev.steps === 1 ? " step" : " steps"));
  if (ev.ms) bits.push((ev.ms / 1000).toFixed(1) + " s");
  if (ev.text) bits.push(ev.text);
  if (bits.length) r.appendChild(el("span", null, "· " + bits.join(" · ")));
  setMiniLine(label + (bits.length ? " · " + bits.join(" · ") : ""));
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

// A question is answered by whatever happens next. Its buttons go then: an
// "Allow" left on an old card - a replayed one especially - would send "yes"
// and approve whatever action happens to be paused NOW.
function settleQuestions() {
  feed.querySelectorAll(".card.confirm .actions, .card.ask .actions").forEach((a) => a.remove());
}

function askCard(ev) {
  const card = el("article", "card ask");
  card.appendChild(el("div", "card-head", ev.question || "Which one did you mean?"));
  const actions = el("div", "actions col");
  (ev.options || []).forEach((o) => {
    const b = el("button", "btn choice");
    b.type = "button";
    b.append(el("span", "choice-label", o.label), el("span", "choice-p", pct(o.p)));
    b.addEventListener("click", () => { actions.remove(); post("/pick", { id: o.id }).catch(() => {}); });
    actions.appendChild(b);
  });
  const none = el("button", "btn quiet", "Neither - stop");
  none.type = "button";
  none.addEventListener("click", () => { actions.remove(); post("/pick", { id: "stop" }).catch(() => {}); });
  actions.appendChild(none);
  card.appendChild(actions);
  return card;
}

function handle(ev) {
  if (ev.seq && ev.seq <= lastSeq) return;
  if (ev.seq) lastSeq = ev.seq;
  if (["user", "decision", "done", "confirm", "ask", "error", "cleared"].includes(ev.kind)) settleQuestions();
  switch (ev.kind) {
    case "status":
      if (ev.phase === "loading") { setStatus("loading", "Loading"); setMiniLine(ev.text || "Loading…"); }
      else if (ev.phase === "ready") {
        const first = !ready;
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
        if (first) { setMiniLine("Ready - ask me something"); flushQueue(); }
        renderSuggest();
      }
      else if (ev.text) { add(line(ev.text)); setMiniLine(ev.text); }
      break;
    case "busy":
      busy = !!ev.busy;
      stop.hidden = !busy;
      setStatus(busy ? "busy" : (ready ? "ready" : "loading"), busy ? "Working" : (ready ? "Ready" : "Loading"));
      if (busy) setMiniLine("Working on it…");
      renderSuggest();
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
      head.appendChild(el("span", "target", ev.why && ev.why !== "opened" ? ev.why : (ev.title || ev.url || "")));
      card.appendChild(head);
      const shot = screenshot(ev);
      if (shot) card.appendChild(shot);
      if (ev.text) card.appendChild(el("div", "card-note", ev.text));
      add(stepCard(card)); pruneImages();
      setWhere(ev.url, ev.title);
      break;
    }
    case "decision":
      lastCard = add(stepCard(decisionCard(ev))); pruneImages();
      setWhere(ev.url, ev.title);
      setMiniLine(ev.operation.replace("_", " ") + (ev.target_line ? " " + short(ev.target_line) : ""));
      break;
    case "typing": {
      const l = line("Typing ", "typing");
      l.lastChild.appendChild(el("code", null, ev.text));
      add(l);
      break;
    }
    case "acted":
      if (lastCard && ev.change && ev.change !== "nothing visible changed") {
        lastCard.appendChild(el("div", "card-note", "→ " + ev.change));
      } else if (lastCard && ev.changed === false) {
        lastCard.appendChild(el("div", "card-note", "→ nothing visible changed"));
      }
      setWhere(ev.url);
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
      needsYou("Needs your OK");
      break;
    }
    case "ask":
      add(askCard(ev));
      needsYou("Not sure - which one did you mean?");
      break;
    case "done":
      lastStatus = ev.status;
      if (ev.status === "closed") { where.open = false; whereBtn.hidden = true; }
      if (ev.url) setWhere(ev.url);
      add(result(ev.status, ev));
      if (ev.captcha) needsYou("Needs you in the browser");
      break;
    case "error":
      add(el("div", "error", ev.text));
      if (!ready) setStatus("error", "Error");
      needsYou(ev.text);
      break;
    case "cleared":
      feed.replaceChildren(empty);
      lastCard = null; lastStatus = null;
      setMiniLine("New chat - ask me something");
      renderSuggest();
      break;
    case "quit":
      ready = false;
      off.hidden = false;
      setStatus("error", "Off");
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
      if (off.hidden) setStatus("error", "Reconnecting");
    }
    await new Promise((res) => setTimeout(res, delay));
  }
}

// ------------------------------------------------------------------ input
// Sent while the model was still loading, a message used to be answered "try
// again" and dropped. Now it waits its turn and goes the moment Jobe is ready.
async function say(text, again) {
  text = (text || "").trim();
  if (!text) return;
  if (!ready) {
    if (!off.hidden) return;
    queued.push(text);
    add(line("Queued “" + text + "” - it goes as soon as the model is ready."));
    return;
  }
  if (!again) { history.push(text); histIdx = -1; }
  try { await post("/say", { text }); }
  catch (e) { add(el("div", "error", "Could not send: " + errorText(e))); }
}
async function flushQueue() {
  while (queued.length) await say(queued.shift());
}

$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = ""; send.disabled = true;
  say(text);
});
input.addEventListener("input", () => { send.disabled = !input.value.trim(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "ArrowUp" && history.length) {
    histIdx = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1);
    input.value = history[histIdx]; send.disabled = false;
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
newBtn.addEventListener("click", () => post("/clear").catch(() => {}));

// Quit asks twice, in place: the first click arms the button, a second within
// three seconds stops everything. A dialog would be one more window.
let quitTimer = null;
quitBtn.addEventListener("click", () => {
  if (!quitBtn.classList.contains("arm")) {
    quitBtn.classList.add("arm");
    quitBtn.title = "Click again to quit Jobe";
    quitTimer = setTimeout(() => { quitBtn.classList.remove("arm"); quitBtn.title = "Quit Jobe"; }, 3000);
    return;
  }
  clearTimeout(quitTimer);
  post("/quit").catch(() => {});
});

// ------------------------------------------------------------------ mini
// A bar, like the Gemini box: the header's live line and the input. The page
// knows how tall its bar is; the server resizes the window, which a page cannot.
function applyMini(on) {
  mini = on;
  document.body.classList.toggle("mini", on);
  miniBtn.setAttribute("aria-pressed", String(on));
  miniBtn.title = on ? "Open the full window" : "Shrink to a bar";
  miniBtn.setAttribute("aria-label", miniBtn.title);
  store("jobe.mini", on ? "1" : "0");
}
function setMini(on) {
  if (on === mini) return;
  if (on) store("jobe.fullHeight", String(window.outerHeight));
  applyMini(on);
  setTimeout(() => {
    const content = on ? bar.offsetHeight + composer.offsetHeight : 0;
    post("/window", { mode: on ? "mini" : "full", content, chrome: window.outerHeight - window.innerHeight,
                      dpr: window.devicePixelRatio || 1, restore: Number(stored("jobe.fullHeight")) || null })
      .catch(() => {});
    if (!on) toBottom();
  }, 0);
}
// When Jobe needs the person - an OK, a pick, a wall, an error - the bar opens up.
function needsYou(text) { setMiniLine(text); if (mini) setMini(false); }
miniBtn.addEventListener("click", () => setMini(!mini));

// ------------------------------------------------------------------ voice
// Recorded here as 16 kHz mono WAV and transcribed on this PC by Juno's Whisper
// sidecar, through the server. Chrome's own recognition would send it to Google.
let rec = null;
function micState(state, hint) {
  micBtn.dataset.state = state;
  micBtn.setAttribute("aria-pressed", String(state === "live"));
  input.placeholder = hint || "Ask Jobe to do something…";
}
async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true,
                                                                   noiseSuppression: true } });
  } catch (e) {
    add(el("div", "error", "The microphone is blocked - allow it for this window, then try again."));
    return;
  }
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const tap = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  tap.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(tap); tap.connect(ctx.destination);
  rec = { stream, ctx, source, tap, chunks, rate: ctx.sampleRate,
          limit: setTimeout(() => finishRecording(), 60000) };
  micState("live", "Listening… click the mic again when you are done");
  post("/voice").catch(() => {});                  // the model starts loading while you speak
}
function toWav(chunks, rate) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const ratio = rate / 16000, n = Math.floor(total / ratio);
  const pcm = new Int16Array(n);
  const all = new Float32Array(total);
  let at = 0;
  chunks.forEach((c) => { all.set(c, at); at += c.length; });
  for (let i = 0; i < n; i++) {                    // average each 16 kHz sample's span
    const a = Math.floor(i * ratio), b = Math.min(total, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < b; j++) s += all[j];
    const v = Math.max(-1, Math.min(1, s / Math.max(1, b - a)));
    pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  const buf = new ArrayBuffer(44 + pcm.length * 2), dv = new DataView(buf);
  const text = (o, s) => [...s].forEach((ch, k) => dv.setUint8(o + k, ch.charCodeAt(0)));
  text(0, "RIFF"); dv.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, 16000, true); dv.setUint32(28, 32000, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); text(36, "data"); dv.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}
async function finishRecording() {
  if (!rec) return;
  const r = rec; rec = null;
  clearTimeout(r.limit);
  r.tap.disconnect(); r.source.disconnect();
  r.stream.getTracks().forEach((t) => t.stop());
  await r.ctx.close();
  micState("busy", "Transcribing…");
  try {
    const res = await fetch("/transcribe", { method: "POST", body: toWav(r.chunks, r.rate),
                                             headers: { "Content-Type": "audio/wav", "X-Jobe-Token": TOKEN } });
    const out = await res.json();
    if (!res.ok || out.error) throw new Error(out.error || "status " + res.status);
    const said = (out.text || "").trim();
    if (said) say(said); else add(line("I didn't catch anything - try again a little closer to the mic."));
  } catch (e) {
    add(el("div", "error", "Speech-to-text: " + (e.message || e)));
  } finally {
    micState("idle");
  }
}
micBtn.addEventListener("click", () => { if (micBtn.dataset.state === "busy") return; rec ? finishRecording() : startRecording(); });
micState("idle");

// ------------------------------------------------------------------ brain
function showBrain(b) {
  brainBtn.dataset.kind = b.kind;
  brainName.textContent = b.kind === "remote" ? b.name.split("/").pop() : "Jobe";
  brainBtn.title = b.kind === "remote" ? "Answering: " + b.name + " on OpenRouter"
                                       : "Answering: Jobe (Qwen3.5-4B) on this PC";
}
function note(text, bad) { brainNote.textContent = text || ""; brainNote.classList.toggle("bad", !!bad); }
function closeMenu() { brainMenu.hidden = true; brainBtn.setAttribute("aria-expanded", "false"); }

brainBtn.addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!brainMenu.hidden) return closeMenu();
  brainMenu.hidden = false; brainBtn.setAttribute("aria-expanded", "true");
  try {
    const h = await (await fetch("/health")).json();
    const noKey = !h.remote_available;
    remoteModel.disabled = noKey; remoteForm.querySelector("button").disabled = noKey;
    note(noKey ? "Set OPENROUTER_API_KEY as a user environment variable to use a hosted model."
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

// A window reopened at bar height (the browser remembers its size) starts as a bar.
if (stored("jobe.mini") === "1" && window.innerHeight < 220) applyMini(true);
connect();
input.focus();
