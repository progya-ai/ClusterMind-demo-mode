/* ClusterMind web GUI. Plain JS, no external dependencies.
   All data from the server is inserted with textContent (never innerHTML). */
"use strict";

const $ = (s) => document.querySelector(s);
const MODELS = ["qwen1.5b", "llama1b", "qwen7b"];
const S = { snapshot: null, snapshots: [], status: null, jobs: new Map(), runRows: [] };

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
const badge = (v) => el("span", { class: `badge b-${String(v).replace(/[^a-z_]/gi, "")}`, text: String(v) });
const pct = (x) => (x === null || x === undefined ? "" : (100 * x).toFixed(1) + "%");
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 1800); }

async function api(method, url, body, isForm) {
  const opt = { method, headers: { "X-CM-Request": "1" } };
  if (body !== undefined) {
    if (isForm) opt.body = body;
    else { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  }
  let r;
  try { r = await fetch(url, opt); } catch (e) { return { ok: false, status: 0, data: { error: "Server not reachable." } }; }
  let data = null;
  try { data = await r.json(); } catch (e) { data = { error: `HTTP ${r.status}` }; }
  return { ok: r.ok, status: r.status, data };
}

function table(tbl, cols, rows, onClick) {
  tbl.replaceChildren();
  const thead = el("thead", {}, el("tr", {}, cols.map((c) => el("th", { class: c.num ? "num" : null, text: c.label }))));
  const tb = el("tbody");
  for (const r of rows) {
    const tr = el("tr", { class: onClick ? "clickable" : null });
    for (const c of cols) {
      const v = c.render ? c.render(r) : r[c.key];
      tr.append(el("td", { class: [c.num ? "num" : "", c.wrap ? "wrap" : ""].join(" ").trim() || null }, v instanceof Node ? v : (v ?? "")));
    }
    if (onClick) tr.addEventListener("click", () => onClick(r));
    tb.append(tr);
  }
  if (!rows.length) tb.append(el("tr", {}, el("td", { colspan: cols.length, class: "muted", text: "No rows." })));
  tbl.append(thead, tb);
}

function setErr(id, msg, input) {
  $(id).textContent = msg || "";
  document.querySelectorAll(".invalid").forEach((x) => x.classList.remove("invalid"));
  if (msg && input) input.classList.add("invalid");
}

function ctx() {
  const seed = $("#ctx-seed").value.trim();
  return { snapshot: $("#ctx-snapshot").value, model: $("#ctx-model").value, seed };
}
function validSeed() {
  const s = $("#ctx-seed").value.trim();
  if (!/^\d{1,9}$/.test(s)) { $("#ctx-seed").classList.add("invalid"); return false; }
  $("#ctx-seed").classList.remove("invalid"); return true;
}

// ------------------------------------------------------------------ jobs
// Operations run as background jobs on the server; we poll for status.
// Long console tasks are listed with their status and an optional log in the Tasks tab.
function trackJob(snap) {
  return new Promise((resolve) => {
    const j = { id: snap.id, title: snap.title, command: snap.command, status: snap.status, next: 0, resolve,
      kind: snap.kind, name: snap.name, lines: [], started: Date.now(), ended: null, open: false };
    S.jobs.set(j.id, j);
    ingest(j, snap);
    renderJobs();
  });
}
function ingest(j, snap) {
  if (snap.lines && snap.lines.length) {
    j.lines.push(...snap.lines);
    if (j.lines.length > 20000) j.lines.splice(0, j.lines.length - 20000);
  }
  j.next = snap.next; j.status = snap.status;
  if (!["queued", "running", "cancelling"].includes(snap.status) && j.resolve) {
    j.ended = Date.now();
    const r = j.resolve; j.resolve = null; r(snap); renderJobs(); refreshStatus();
  }
}
async function poll() {
  const active = [...S.jobs.values()].filter((j) => j.resolve);
  await Promise.all(active.map(async (j) => {
    const r = await api("GET", `/api/jobs/${j.id}?since=${j.next}`);
    if (r.ok) ingest(j, r.data);
  }));
  if (active.length) renderJobs();
  setTimeout(poll, active.length ? 700 : 1500);
}
function dur(j) { return `${(((j.ended || Date.now()) - j.started) / 1000).toFixed(0)} s`; }
function renderJobs() {
  const n = [...S.jobs.values()].filter((j) => j.resolve).length;
  $("#pill-jobs").lastChild.textContent = `${n} running`;
  $("#pill-jobs .dot").classList.toggle("on", n > 0);
  const box = $("#jobs-list");
  if (!box) return;
  const list = [...S.jobs.values()].filter((j) => j.kind === "command").reverse();
  $("#jobs-card").hidden = list.length === 0;
  box.replaceChildren(...list.map((j) => {
    const logBtn = el("button", { class: "btn small", type: "button", text: j.open ? "Hide log" : "Log" });
    logBtn.addEventListener("click", () => { j.open = !j.open; renderJobs(); });
    const row = el("div", { class: "jobrow" },
      el("div", { class: "row" }, badge(j.status), el("code", { class: "jobcmd", title: j.command, text: j.command }),
        el("span", { class: "muted", text: dur(j) }), el("span", { class: "right" }),
        j.status === "running" ? el("button", { class: "btn small danger", type: "button", text: "Cancel", onclick: () => cancelJob(j.id) }) : null,
        logBtn,
        el("button", { class: "btn small", type: "button", text: "Download log", onclick: () => downloadText(`${j.name}-${j.id}.log`, j.lines.join("\n") + "\n") })));
    if (j.open) {
      const pre = el("pre", { class: "evidence joblog", text: j.lines.join("\n") });
      row.append(pre);
      requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
    }
    return row;
  }));
}
function downloadText(name, txt) {
  const a = el("a", { href: URL.createObjectURL(new Blob([txt], { type: "text/plain" })), download: name });
  document.body.append(a); a.click(); a.remove();
}
async function cancelJob(id) {
  const r = await api("POST", `/api/jobs/${id}/cancel`);
  if (!r.ok) toast(r.data.error || "Cannot cancel");
}
// start a job-returning request; returns final job snapshot or {error}
async function runJob(method, url, body, isForm) {
  const r = await api(method, url, body, isForm);
  if (!r.ok) return { error: r.data.error || `HTTP ${r.status}` };
  return trackJob(r.data);
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
function showView(v) {
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", t.dataset.view === v ? "true" : "false"));
  document.querySelectorAll(".view").forEach((x) => (x.hidden = x.id !== `view-${v}`));
  try { localStorage.setItem("cm-view", v); } catch (e) { /* ignore */ }
  if (v === "files") loadFiles();
  if (v === "audit") loadAudit();
  if (v === "runs" && !$("#run-file").options.length) loadRuns();
}

// ------------------------------------------------------------------ public demo mode
function applyDemo() {
  if (document.body.classList.contains("demo")) return;
  document.body.classList.add("demo");
  $("#demo-banner").hidden = false;
  ["ask", "tasks", "files", "audit"].forEach((v) => { const t = document.querySelector(`.tab[data-view="${v}"]`); if (t) t.hidden = true; });
  $("#ctx-model").closest(".field").hidden = true;
  $("#ctx-seed").closest(".field").hidden = true;
  $("#form-batch").closest(".card").hidden = true;
  $("#pill-jobs").hidden = true;
  $("#pill-trace .dot").classList.add("on");
  $("#pill-trace").lastChild.textContent = "public demo";
  $("#pill-models").lastChild.textContent = "rule-based parser, no AI model";
  $("#snap-hint").textContent = "Real node records from the Alibaba PAI trace. Green = busy, yellow = idle, red = recent failures. Click a node to try a shutdown on it.";
  $("#act-hint").textContent = "Keyword parser (stands in for the AI model) → whitelist and node check → typed node-name confirmation → busy nodes are blocked (no override). Execution is simulated.";
  const cur = document.querySelector('.tab[aria-selected="true"]');
  if (cur && cur.hidden) showView("dashboard");
}

// ------------------------------------------------------------------ status / dashboard
async function refreshStatus() {
  const r = await api("GET", "/api/status");
  if (!r.ok) return;
  const st = (S.status = r.data);
  if (st.demo) { applyDemo(); }
  const present = st.models.filter((m) => m.present).map((m) => m.model);
  if (!st.demo) {
    $("#pill-trace .dot").classList.toggle("on", st.trace_available);
    $("#pill-trace").lastChild.textContent = st.trace_available ? "trace parquet" : "no trace parquet";
    $("#pill-models .dot").classList.toggle("on", present.length > 0);
    $("#pill-models").lastChild.textContent = `${present.length}/3 models`;
  }
  const stat = (k, v, s) => el("div", { class: "stat" }, el("div", { class: "k", text: k }), el("div", { class: "v", text: v }), el("div", { class: "s", text: s || "" }));
  const base = [stat("Snapshots", st.snapshots, "40 × 8 nodes"), stat("Questions", st.questions, "real cluster snapshots, 8 question types")];
  $("#stats").replaceChildren(...(st.demo
    ? [...base, stat("Mode", "Public demo", "recorded Alibaba GPU data"), stat("Checker", "Real", "same code as the full app")]
    : [...base,
      stat("Models present", `${present.length}/3`, present.join(", ") || "run task 'download-models'"),
      stat("Trace drift", st.trace_available ? "real" : "off", st.trace_available ? "state re-read from the trace" : "run task 'prepare-data'"),
      stat("Audit entries", st.audit_entries, "runs/demo_audit.jsonl")]));
  const MODEL_NAMES = { qwen1_5b: "Qwen2.5-1.5B", qwen15b: "Qwen2.5-1.5B", llama1b: "Llama-3.2-1B", qwen7b: "Qwen2.5-7B (larger)" };
  const niceModel = (c) => { const k = String(c).split("_")[0].replace(/\./g, ""); return MODEL_NAMES[k] || c; };
  table($("#headline"), [
    { label: "AI model", render: (x) => niceModel(x.config) },
    { label: "Raw accuracy", num: true, render: (x) => pct(x.ungated_acc) },
    { label: "Old check: answers delivered", num: true, render: (x) => pct(x.v0_coverage) },
    { label: "Old check: delivered accuracy", num: true, render: (x) => pct(x.v0_delivered_acc) },
    { label: "New checker: answers delivered", num: true, render: (x) => pct(x.v2_coverage) },
    { label: "New checker: delivered accuracy", num: true, render: (x) => pct(x.v2_delivered_acc) }], st.headline);
  const m = st.models.find((x) => x.model === $("#ctx-model").value);
  $("#ctx-info").textContent = m && !m.present ? `${m.file} not downloaded: Ask and Action need it.` : "";
}

async function loadSnapshots() {
  const r = await api("GET", "/api/snapshots");
  if (!r.ok) return;
  S.snapshots = r.data;
  const sel = $("#ctx-snapshot");
  sel.replaceChildren(...r.data.map((s) => el("option", { value: s.snapshot_id, text: `${s.snapshot_id} · ${s.idle} idle${s.with_failures ? ` · ${s.with_failures} w/ failures` : ""}` })));
  let saved = null; try { saved = localStorage.getItem("cm-snap"); } catch (e) { /* ignore */ }
  if (saved && r.data.some((s) => s.snapshot_id === saved)) sel.value = saved;
  await loadSnapshot();
}
async function loadSnapshot() {
  const sid = $("#ctx-snapshot").value;
  try { localStorage.setItem("cm-snap", sid); } catch (e) { /* ignore */ }
  const r = await api("GET", `/api/snapshots/${encodeURIComponent(sid)}`);
  if (!r.ok) return;
  S.snapshot = r.data;
  $("#snap-title").textContent = `${sid} · evidence ID ${r.data.evidence_hash}`;
  $("#snap-evidence").textContent = r.data.evidence_text;
  const demoMode = !!(S.status && S.status.demo);
  const askAbout = (x) => {
    if (demoMode) { $("#act-i").value = `Shut down ${x.node}`; showView("act"); $("#act-i").focus(); return; }
    $("#ask-q").value = `What is the GPU utilization of ${x.node}?`; showView("ask"); $("#ask-q").focus();
  };
  $("#node-cards").replaceChildren(...r.data.nodes.map((x) => {
    const busy = x.running_instances > 0;
    const state = x.failed_last_hour > 0 ? "fail" : (busy ? "busy" : "idle");
    const w = Math.max(0, Math.min(100, Math.round(x.gpu_util_pct / 5) * 5));
    const metric = (v, l) => el("div", { class: "m" }, el("div", { class: "mv", text: v }), el("div", { class: "ml", text: l }));
    return el("button", { class: `node-card state-${state}`, type: "button", onclick: () => askAbout(x) },
      el("div", { class: "node-id", text: "GPU node" }),
      el("div", { class: "node-name", text: x.node }),
      el("div", { class: "node-metrics" }, metric(`${x.gpu_util_pct.toFixed(0)}%`, "GPU util"), metric(x.gpu_mem_gb.toFixed(1), "Mem GB"), metric(String(x.running_instances), "Running")),
      el("div", { class: "bar-wrap" }, el("div", { class: `bar w-${w}` })),
      el("div", { class: "node-foot" }, badge(busy ? "busy" : "idle"),
        el("span", { text: x.failed_last_hour > 0 ? `${x.failed_last_hour} failed in last hour` : "no recent failures" })));
  }));
  table($("#nodes"), [
    { label: "node", key: "node" },
    { label: "state", render: (x) => badge(x.running_instances > 0 ? "busy" : "idle") },
    { label: "running", key: "running_instances", num: true },
    { label: "GPU util %", num: true, render: (x) => x.gpu_util_pct.toFixed(1) },
    { label: "GPU mem GB", num: true, render: (x) => x.gpu_mem_gb.toFixed(1) },
    { label: "failed / 1 h", key: "failed_last_hour", num: true }], r.data.nodes,
  (x) => { $("#ask-q").value = `What is the GPU utilization of ${x.node}?`; showView("ask"); $("#ask-q").focus(); });
  $("#ask-samples").replaceChildren(...r.data.questions.map((q) => el("option", { value: q.question })));
  const chips = r.data.questions.slice(0, 8);
  $("#ask-chips").replaceChildren(...chips.map((q) => el("button", { class: "chip", type: "button", text: q.question, onclick: () => ($("#ask-q").value = q.question) })));
  const idle = r.data.nodes.find((n) => n.running_instances === 0), busy = r.data.nodes.find((n) => n.running_instances > 0);
  const ex = [idle && `Shut down ${idle.node}`, busy && `Shut down ${busy.node}`, busy && `${busy.node} has been idle for hours, shut it down.`,
    "Shut down the idle node, it has been sitting unused", "Shut down Node-15", busy && `Send an alert about ${busy.node}`].filter(Boolean);
  $("#act-chips").replaceChildren(...ex.map((t) => el("button", { class: "chip", type: "button", text: t, onclick: () => ($("#act-i").value = t) })));
}
$("#ctx-snapshot").addEventListener("change", loadSnapshot);
$("#ctx-model").addEventListener("change", refreshStatus);

// ------------------------------------------------------------------ result cards
function claimsTable(claims) {
  const t = el("table");
  table(t, [
    { label: "", render: (c) => badge(c.supported ? "ok" : "bad") },
    { label: "kind", key: "kind" }, { label: "node", render: (c) => c.node ?? "" },
    { label: "metric", render: (c) => c.metric ?? "" }, { label: "check", key: "why", wrap: true }], claims);
  t.querySelectorAll(".b-ok").forEach((b) => b.classList.add("b-supported"));
  t.querySelectorAll(".b-bad").forEach((b) => b.classList.add("b-unsupported"));
  return el("div", { class: "tablewrap" }, t);
}
function verdictCard(r, extra) {
  const cls = r.delivered ? "ok" : (r.v2 === "unverifiable" ? "warn" : "bad");
  return el("div", { class: `result ${cls}` },
    el("div", {}, badge(r.v2), " ", el("strong", { text: r.delivered ? "Delivered" : "Shown with warning" }),
      r.latency_s !== undefined ? el("span", { class: "muted", text: `  ·  ${r.model} · ${r.latency_s.toFixed(1)} s` }) : null),
    el("div", { class: "answer", text: r.answer }),
    el("dl", { class: "kv" },
      el("dt", { text: "evidence hash h(E)" }), el("dd", { class: "mono", text: r.evidence_hash }),
      el("dt", { text: "v0 (prototype)" }), el("dd", {}, badge(r.v0)),
      r.v1s ? el("dt", { text: "v1s" }) : null, r.v1s ? el("dd", {}, badge(r.v1s)) : null,
      el("dt", { text: "evidence nodes" }), el("dd", { text: (r.evidence_nodes || []).join(", ") }), extra || null),
    el("h3", { text: `Checked claims (${r.claims.length})` }),
    r.claims.length ? claimsTable(r.claims) : el("p", { class: "hint", text: "No checkable claim found: UNVERIFIABLE." }),
    el("details", {}, el("summary", { text: "Evidence text r(E)" }), el("pre", { class: "evidence", text: r.evidence_text })));
}

// ------------------------------------------------------------------ ask
$("#form-ask").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#ask-q").value.trim();
  if (!q) return setErr("#ask-err", "Enter a question.", $("#ask-q"));
  if (q.length > 500) return setErr("#ask-err", "Question is too long (maximum 500 characters).", $("#ask-q"));
  if (!validSeed()) return setErr("#ask-err", "Seed must be a non-negative integer.");
  setErr("#ask-err", "");
  const c = ctx();
  $("#ask-btn").disabled = true;
  $("#ask-result").replaceChildren(el("p", { class: "hint", text: `Running ${c.model} on CPU… (usually 5–60 s)` }));
  const j = await runJob("POST", "/api/ask", { ...c, question: q, retrieval: $("#ask-retrieval").value, style: $("#ask-style").value });
  $("#ask-btn").disabled = false;
  if (j.error || j.status !== "succeeded") { $("#ask-result").replaceChildren(); return setErr("#ask-err", j.error || "The operation failed."); }
  $("#ask-result").replaceChildren(verdictCard(j.result));
});

// ------------------------------------------------------------------ check
$("#form-check").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const a = $("#chk-a").value.trim();
  if (!a) return setErr("#chk-err", "Enter an answer to check.", $("#chk-a"));
  setErr("#chk-err", "");
  const j = await runJob("POST", "/api/check", { snapshot: $("#ctx-snapshot").value, question: $("#chk-q").value, answer: a, retrieval: $("#chk-retrieval").value });
  if (j.error || j.status !== "succeeded") return setErr("#chk-err", j.error || "The operation failed.");
  $("#chk-result").replaceChildren(verdictCard(j.result, [el("dt", { text: "checker time" }), el("dd", { text: `${j.result.check_us} µs` })]));
});
$("#demo-check-btn").addEventListener("click", async () => {
  const j = await runJob("POST", "/api/demo-check");
  if (j.error) return setErr("#chk-err", j.error);
  const t = el("table");
  table(t, [{ label: "question", key: "question", wrap: true }, { label: "answer", key: "answer", wrap: true },
    { label: "v0 (prototype)", render: (x) => badge(x.v0) }, { label: "v2 (new checker)", render: (x) => badge(x.v2) }], j.result,
  (x) => { $("#chk-q").value = x.question; $("#chk-a").value = x.answer; $("#ctx-snapshot").value = "S003"; loadSnapshot(); });
  $("#chk-result").replaceChildren(el("h3", { text: "Example checks on snapshot S003 (click a row to load it)" }), el("div", { class: "tablewrap" }, t));
});

// batch
let batchRows = [];
$("#form-batch").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = $("#batch-file").files[0];
  if (!f) return setErr("#batch-err", "Choose a file.", $("#batch-file"));
  if (!/\.(csv|jsonl|json)$/i.test(f.name)) return setErr("#batch-err", "Upload a .csv, .jsonl or .json file.", $("#batch-file"));
  if (f.size > 2 * 1024 * 1024) return setErr("#batch-err", "File too large (maximum 2 MB).", $("#batch-file"));
  setErr("#batch-err", "");
  const fd = new FormData(); fd.append("file", f);
  const j = await runJob("POST", "/api/batch-check", fd, true);
  if (j.error || j.status !== "succeeded") return setErr("#batch-err", j.error || "The operation failed.");
  batchRows = j.result.rows;
  $("#batch-dl").href = j.result.download;
  $("#batch-out").hidden = false;
  renderBatch();
});
function renderBatch() {
  const q = $("#batch-filter").value.toLowerCase();
  const rows = batchRows.filter((r) => !q || JSON.stringify(r).toLowerCase().includes(q));
  const ok = batchRows.filter((r) => !r.error);
  $("#batch-summary").textContent = `${rows.length} of ${batchRows.length} rows shown · delivered ${ok.filter((r) => r.delivered === true).length}/${ok.length} · invalid ${batchRows.length - ok.length}`;
  table($("#batch-table"), [{ label: "#", key: "row", num: true }, { label: "snapshot", key: "snapshot" },
    { label: "question", key: "question", wrap: true }, { label: "answer", key: "answer", wrap: true },
    { label: "v0", render: (x) => (x.v0 ? badge(x.v0) : "") }, { label: "v2", render: (x) => (x.v2 ? badge(x.v2) : "") },
    { label: "error", key: "error", wrap: true }], rows.slice(0, 1000));
}
$("#batch-filter").addEventListener("input", renderBatch);

// ------------------------------------------------------------------ action
$("#form-act").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const ins = $("#act-i").value.trim();
  if (!ins) return setErr("#act-err", "Enter an instruction.", $("#act-i"));
  if (!validSeed()) return setErr("#act-err", "Seed must be a non-negative integer.");
  setErr("#act-err", "");
  const c = ctx();
  $("#act-btn").disabled = true;
  $("#act-result").replaceChildren(el("p", { class: "hint", text: `Asking ${(S.status && S.status.demo) ? "the keyword parser" : c.model} for a proposal…` }));
  const j = await runJob("POST", "/api/actions/propose", { ...c, instruction: ins });
  $("#act-btn").disabled = false;
  if (j.error || j.status !== "succeeded") { $("#act-result").replaceChildren(); return setErr("#act-err", j.error || "The operation failed."); }
  renderProposal(j.result);
});
function stateLine(s) { return `running=${s.running_instances} · GPU util ${s.gpu_util_pct}% · GPU mem ${s.gpu_mem_gb} GB`; }
function renderProposal(p) {
  const box = el("div", { class: `result ${p.status === "dropped" ? "bad" : p.status === "logged" ? "ok" : "warn"}` },
    el("div", {}, badge(p.status), " ", el("strong", { text: p.status === "dropped" ? `Dropped: ${p.reason}` : `${p.action} on ${p.node}` })),
    el("h3", { text: (S.status && S.status.demo) ? "Parser output (raw)" : "Model proposal (raw)" }), el("pre", { class: "evidence", text: p.raw }));
  if (p.status === "logged") box.append(el("p", { text: "Non-disruptive action: logged (simulated)." }));
  if (p.status === "needs_confirmation") {
    box.append(el("dl", { class: "kv" },
      el("dt", { text: "proposal-time state" }), el("dd", { text: stateLine(p.proposal_state) }),
      el("dt", { text: "prototype util-proxy" }), el("dd", { text: p.util_proxy_busy ? "would say busy" : "would say idle (GPU util ≤ 10%)" })));
    const typed = el("input", { type: "text", id: "act-typed", autocomplete: "off", placeholder: p.node, "aria-label": "Type the node name" });
    const delayChoices = (S.status && S.status.demo) ? [["0", "immediately"]]
      : [["0", "immediately"], ["300", "after 5 min"], ["900", "after 15 min"], ["1800", "after 30 min"], ["3600", "after 1 h"]];
    const delay = el("select", { id: "act-delay" }, delayChoices.map(([v, t]) => el("option", { value: v, text: t })));
    const err = el("div", { class: "err", role: "alert" });
    const out = el("div");
    const btn = el("button", { class: "btn primary", type: "button", text: "Confirm" });
    btn.addEventListener("click", async () => {
      if (!typed.value.trim()) { err.textContent = "Type the node name to confirm."; typed.classList.add("invalid"); return; }
      typed.classList.remove("invalid"); err.textContent = ""; btn.disabled = true;
      const j = await runJob("POST", "/api/actions/confirm", { proposal_id: p.id, typed: typed.value, delay: delay.value });
      if (j.error || j.status !== "succeeded") { err.textContent = j.error || "The operation failed."; btn.disabled = false; return; }
      const c = j.result;
      out.replaceChildren(el("div", { class: `result ${c.status === "executed" ? "ok" : c.status === "blocked" ? "bad" : "warn"}` },
        el("div", {}, badge(c.status), " ", el("strong", { text: c.status === "executed" ? `EXECUTED (simulated): ${c.node} marked shut down`
          : c.status === "blocked" ? `BLOCKED: ${c.node} is running ${c.confirm_state.running_instances} instance(s). No override.`
            : "Typed confirmation did not match; nothing executed." })),
        c.confirm_state ? el("p", { class: "hint", text: `Confirmation-time state (${c.confirm_state_source}): ${stateLine(c.confirm_state)}` }) : null));
      refreshStatus();
    });
    box.append(el("div", { class: "confirmbox" },
      el("strong", { text: `Disruptive action. Type “${p.node}” to confirm.` }),
      el("p", { class: "hint", text: (S.status && S.status.demo) ? "The busy-node check runs on the node's state when you confirm. Busy nodes are blocked." : "The busy-node check runs on the state re-read at confirmation time. A delay re-derives it from the trace (needs the trace parquet)." }),
      el("div", { class: "row" }, el("div", { class: "field grow" }, el("label", { for: "act-typed", text: "Node name" }), typed),
        el("div", { class: "field" }, el("label", { for: "act-delay", text: "Confirm" }), delay), btn), err, out));
  }
  $("#act-result").replaceChildren(box);
}

// ------------------------------------------------------------------ runs
async function loadRuns() {
  const r = await api("GET", "/api/runs");
  if (!r.ok) return;
  $("#run-file").replaceChildren(...r.data.map((f) => el("option", { value: f, text: f })));
  if (r.data.length) loadRunRows();
}
async function loadRunRows() {
  const f = $("#run-file").value;
  $("#run-summary").textContent = "Loading…";
  const [root, ...rest] = f.split("/");
  $("#run-dl").href = `/download/${root}/${rest.join("/")}`;
  const r = await api("GET", `/api/runs/rows?run=${encodeURIComponent(f)}`);
  if (!r.ok) { $("#run-summary").textContent = r.data.error; return; }
  S.runRows = r.data;
  const seeds = [...new Set(r.data.map((x) => x.seed))].sort((a, b) => a - b);
  const tpls = [...new Set(r.data.map((x) => x.template))].sort();
  $("#run-seed").replaceChildren(el("option", { value: "", text: "all" }), ...seeds.map((s) => el("option", { value: String(s), text: String(s) })));
  $("#run-tpl").replaceChildren(el("option", { value: "", text: "all" }), ...tpls.map((s) => el("option", { value: s, text: s })));
  renderRuns();
}
function renderRuns() {
  const q = $("#run-q").value.toLowerCase(), seed = $("#run-seed").value, tpl = $("#run-tpl").value, v2 = $("#run-v2").value, cor = $("#run-correct").value;
  const rows = S.runRows.filter((x) => (!seed || String(x.seed) === seed) && (!tpl || x.template === tpl) && (!v2 || x.v2 === v2)
    && (!cor || String(x.correct) === cor) && (!q || (x.question + " " + x.answer + " " + x.qid).toLowerCase().includes(q)));
  const del = rows.filter((x) => x.delivered);
  $("#run-summary").textContent = `${rows.length} answers · accuracy ${pct(rows.filter((x) => x.correct).length / (rows.length || 1))} · v2 coverage ${pct(del.length / (rows.length || 1))} · v2 delivered accuracy (pooled) ${del.length ? pct(del.filter((x) => x.correct).length / del.length) : "n/a"}`;
  table($("#run-table"), [{ label: "qid", key: "qid" }, { label: "seed", key: "seed", num: true },
    { label: "question", key: "question", wrap: true }, { label: "answer", key: "answer", wrap: true },
    { label: "correct", render: (x) => badge(x.correct) }, { label: "v2", render: (x) => badge(x.v2) }], rows.slice(0, 500),
  (x) => { $("#ctx-snapshot").value = x.snapshot; loadSnapshot(); $("#chk-q").value = x.question; $("#chk-a").value = x.answer; showView("check"); });
}
["#run-seed", "#run-tpl", "#run-v2", "#run-correct"].forEach((s) => $(s).addEventListener("change", renderRuns));
$("#run-q").addEventListener("input", renderRuns);
$("#run-file").addEventListener("change", loadRunRows);

// ------------------------------------------------------------------ tasks
const sel = (id, opts, def) => el("select", { id }, opts.map((o) => el("option", { value: o, text: o, selected: o === def })));
const TASK_FORMS = {
  "download-models": () => el("div", { class: "checks" }, MODELS.map((m) => el("label", {}, el("input", { type: "checkbox", value: m, class: "dl-model", checked: m === "qwen1.5b" }), m))),
  generate: () => [field("Model", sel("t-gen-model", MODELS, "llama1b")), field("Evidence", sel("t-gen-ret", ["full", "tfidf"], "full")),
    field("Prompt", sel("t-gen-style", ["contract", "original"], "contract")), field("Seeds", el("input", { id: "t-gen-seeds", type: "text", value: "0" })),
    field("Limit (questions, optional)", el("input", { id: "t-gen-limit", type: "number", min: "1", max: "320", placeholder: "all 320" }))],
  judge: () => [field("Judge", sel("t-j-model", MODELS, "qwen7b")), field("Item set", el("input", { id: "t-j-items", type: "text", value: "gen:llama1b_full_contract:0", size: "30" }))],
  actions: () => [field("Model", sel("t-a-model", MODELS, "qwen1.5b"))],
};
const TASK_ARGS = {
  "download-models": () => ({ models: [...document.querySelectorAll(".dl-model:checked")].map((x) => x.value) }),
  generate: () => ({ model: $("#t-gen-model").value, retrieval: $("#t-gen-ret").value, style: $("#t-gen-style").value, seeds: $("#t-gen-seeds").value, limit: $("#t-gen-limit").value }),
  judge: () => ({ judge: $("#t-j-model").value, itemset: $("#t-j-items").value }),
  actions: () => ({ model: $("#t-a-model").value }),
};
function field(label, input) { return el("div", { class: "field" }, el("label", { for: input.id, text: label }), input); }
async function loadCommands() {
  const r = await api("GET", "/api/commands");
  if (!r.ok) return;
  $("#task-cards").replaceChildren(...r.data.map((c) => {
    const err = el("div", { class: "err", role: "alert" });
    const btn = el("button", { class: "btn primary", type: "button", text: "Run" });
    btn.addEventListener("click", async () => {
      err.textContent = "";
      const args = TASK_ARGS[c.name] ? TASK_ARGS[c.name]() : {};
      if (c.name === "generate" && !/^\d{1,2}(,\d{1,2}){0,4}$/.test(args.seeds.replace(/\s/g, ""))) { err.textContent = "Seeds: up to 5 comma-separated integers, e.g. 0,1,2."; return; }
      if (c.name === "reproduce-all" && !window.confirm("reproduce-all runs every experiment and takes many hours. Start it?")) return;
      btn.disabled = true;
      const j = await runJob("POST", "/api/tasks", { command: c.name, args });
      btn.disabled = false;
      if (j.error) err.textContent = j.error;
      else if (j.status !== "succeeded") err.textContent = `Task ${j.status}. Open its log under Recent tasks.`;
      else toast(`${c.label}: done`);
      if (["tables", "tables-rerun", "prepare-data", "download-models"].includes(c.name)) refreshStatus();
    });
    const form = TASK_FORMS[c.name] ? TASK_FORMS[c.name]() : null;
    return el("div", { class: "card" },
      el("div", { class: "row" }, el("div", { class: "field grow" }, el("strong", { text: c.label }), el("span", { class: "hint", text: c.description })),
        c.long ? el("span", { class: "pill", text: "long-running" }) : null),
      form ? el("div", { class: "row" }, form) : null,
      el("div", { class: "row" }, el("code", { class: "muted", text: `./run.sh ${c.name}` }), el("span", { class: "right" }), btn), err);
  }));
}

// ------------------------------------------------------------------ files & audit
let files = [];
async function loadFiles() {
  const r = await api("GET", "/api/files");
  if (r.ok) { files = r.data; renderFiles(); }
}
function renderFiles() {
  const q = $("#files-q").value.toLowerCase();
  table($("#files-table"), [{ label: "file", key: "label" }, { label: "size", num: true, render: (f) => (f.bytes < 1024 ? `${f.bytes} B` : `${(f.bytes / 1024).toFixed(1)} KB`) },
    { label: "modified", render: (f) => new Date(f.mtime * 1000).toLocaleString() },
    { label: "", render: (f) => el("a", { class: "btn small", href: f.url, download: f.path.split("/").pop(), text: "Download" }) }],
  files.filter((f) => !q || f.label.toLowerCase().includes(q)));
}
$("#files-q").addEventListener("input", renderFiles);
$("#files-refresh").addEventListener("click", loadFiles);

let auditTimer = null;
async function loadAudit() {
  const r = await api("GET", `/api/audit?limit=1000&q=${encodeURIComponent($("#audit-q").value)}`);
  if (!r.ok) return;
  table($("#audit-table"), [{ label: "time", key: "time" }, { label: "via", render: (x) => x.via || "cli" }, { label: "type", key: "type" },
    { label: "snapshot", key: "snapshot" }, { label: "node / question", wrap: true, render: (x) => x.node || x.question || x.instruction || "" },
    { label: "result", render: (x) => (x.type === "answer" ? badge(x.v2) : el("span", { text: x.result || "" })) }], r.data);
}
$("#audit-q").addEventListener("input", () => { clearTimeout(auditTimer); auditTimer = setTimeout(loadAudit, 250); });
$("#audit-refresh").addEventListener("click", loadAudit);

// ------------------------------------------------------------------ boot
(async function boot() {
  await Promise.all([refreshStatus(), loadSnapshots()]);
  if (!(S.status && S.status.demo)) await loadCommands();
  let v = "dashboard"; try { v = localStorage.getItem("cm-view") || v; } catch (e) { /* ignore */ }
  const tabEl = document.querySelector(`.tab[data-view="${v}"]`);
  if (document.getElementById(`view-${v}`) && !(tabEl && tabEl.hidden)) showView(v);
  poll();
})();
