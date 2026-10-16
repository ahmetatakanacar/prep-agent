"use strict";

/* prep-agent web UI. Plain DOM, no framework. Everything that comes from the server or from a
   CSV file is inserted as text, never as HTML. Every sentence of the interface comes from
   i18n.js through `t`. */

/* ================================================================ helpers */

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key.startsWith("--")) el.style.setProperty(key, value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  return add(el, kids);
}

function add(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// Element.append stringifies arrays; this one flattens them like `h` does.
function put(el, ...kids) {
  return add(el, kids);
}

const clear = (el) => el.replaceChildren();
const pad2 = (n) => String(n).padStart(2, "0");

/* ================================================================ language */

const LANGS = ["en", "tr"];
const LOCALES = { en: "en-US", tr: "tr-TR" };
let lang = "en";

function initialLang() {
  try {
    const stored = localStorage.getItem("prep-lang");
    if (LANGS.includes(stored)) return stored;
  } catch {
    /* storage can be blocked; the browser language decides then */
  }
  return (navigator.language || "").toLowerCase().startsWith("tr") ? "tr" : "en";
}

function lookup(key) {
  const own = window.I18N[lang][key];
  return own !== undefined ? own : window.I18N.en[key];
}

const hasText = (key) => window.I18N[lang][key] !== undefined;

function t(key, vars) {
  const text = lookup(key);
  if (text === undefined) {
    console.warn("missing text:", key);
    return key;
  }
  return vars ? text.replace(/\{(\w+)\}/g, (_, name) => (vars[name] != null ? vars[name] : "")) : text;
}

// Singular and plural texts: `key.one` and `key.other`.
const tn = (key, n, vars) => t(`${key}.${n === 1 ? "one" : "other"}`, { n: fmtInt(n), ...vars });

const fmtInt = (n) => Number(n).toLocaleString(LOCALES[lang]);
const pct = (r) => `${(r * 100).toFixed(r > 0 && r < 0.001 ? 2 : 1)}%`;
const fmtScore = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3));

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${pad2(Math.floor(s / 60))}:${pad2(s % 60)}`;
}

function scrollToEl(el, focus) {
  el.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
  if (focus) {
    if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "-1");
    el.focus({ preventScroll: true });
  }
}

function announce(text) {
  const live = document.getElementById("live");
  live.textContent = "";
  window.setTimeout(() => (live.textContent = text), 50);
}

const esc = (text) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/* ================================================================ state */

const state = {
  config: null,
  snap: null,
  result: null,
  // What the user typed into the brief; kept so that a re-render or a language change does not
  // lose it.
  brief: { goal: "", target: "", family: "", prefilled: null },
  qChoices: {},
  pollTimer: null,
  clockTimer: null,
  clock: null,
  questionSig: "",
  questions: null,
  activeQ: 0,
  generation: 0,
};

const sheets = {};
const SHEET_ORDER = ["data", "profile", "brief", "agent", "outcome"];

function stageOf(snap) {
  if (!snap) return "data";
  if (snap.status === "ready") return "brief";
  if (snap.status === "running") return "running";
  if (snap.status === "waiting") return "question";
  if (snap.status === "error") return "error";
  if (snap.status === "failed") return state.result ? "failed" : "running";
  return state.result ? "outcome" : "running";
}

function setStage(name) {
  document.getElementById("tb-stage").textContent = t(`stage.${name}`);
}

/* ================================================================ api */

class ApiError extends Error {
  constructor(message, status, code, params) {
    super(message);
    this.status = status;
    this.code = code;
    this.params = params || {};
  }
}

// The message of an error in the current language; the server's English text is the fallback.
function errorText(error) {
  if (error.code && lookup(`api.${error.code}`) !== undefined) return t(`api.${error.code}`, error.params);
  return error.message;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("network", 0, "network");
  }
  if (!response.ok) {
    let body = {};
    try {
      body = await response.json();
    } catch {
      /* the body was not JSON; the status text will do */
    }
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(detail, response.status, body.code, body.params);
  }
  return response.status === 204 ? null : response.json();
}

const postJson = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/* ================================================================ banner */

function showBanner({ title, message, kind = "error", action }) {
  const slot = document.getElementById("banner");
  clear(slot);
  const banner = h(
    "div",
    { class: `banner ${kind}`, role: kind === "error" ? "alert" : "status" },
    h("div", {}, h("strong", {}, title), h("p", {}, message)),
    action,
    h("button", { type: "button", class: "btn quiet small", onclick: clearBanner }, t("common.dismiss")),
  );
  put(slot, banner);
  banner.scrollIntoView({ block: "nearest" });
}

const clearBanner = () => clear(document.getElementById("banner"));

/* ================================================================ theme and language */

const THEMES = ["auto", "light", "dark"];

function applyTheme(pref) {
  const root = document.documentElement;
  root.dataset.pref = pref;
  root.dataset.theme = pref === "auto" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : pref;
  const button = document.getElementById("theme-toggle");
  button.textContent = t("theme.label", { value: t(`theme.${pref}`) });
  button.setAttribute("aria-label", t("theme.aria"));
}

function initTheme() {
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = THEMES[(THEMES.indexOf(document.documentElement.dataset.pref) + 1) % THEMES.length];
    try {
      localStorage.setItem("prep-theme", next);
    } catch {
      /* private mode: the choice just does not persist */
    }
    applyTheme(next);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (document.documentElement.dataset.pref === "auto") applyTheme("auto");
  });
}

function setLang(next) {
  lang = next;
  document.documentElement.lang = next;
  try {
    localStorage.setItem("prep-lang", next);
  } catch {
    /* the choice just does not persist */
  }
  const button = document.getElementById("lang-toggle");
  button.textContent = t("lang.label", { value: next.toUpperCase() });
  button.setAttribute("aria-label", t("lang.aria"));
}

function initLang() {
  setLang(initialLang());
  document.getElementById("lang-toggle").addEventListener("click", () => {
    const next = LANGS[(LANGS.indexOf(lang) + 1) % LANGS.length];
    swapPrefilledGoal(next);
    setLang(next);
    renderAll();
  });
}

// A goal that is still the text a sample put there follows the language; one the user edited
// does not.
function swapPrefilledGoal(nextLang) {
  const b = state.brief;
  if (!b.prefilled || b.goal !== b.prefilled.text) return;
  const sample = b.prefilled.sample;
  const text = nextLang === "tr" && sample.goal_tr ? sample.goal_tr : sample.goal;
  b.goal = text;
  b.prefilled = { sample, text };
}

function applyStatic() {
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  document.getElementById("tb-mode").textContent = state.config ? t(state.config.fake ? "tb.mode_demo" : "tb.mode_live") : "";
  document.getElementById("tb-data").textContent = state.snap ? state.snap.filename : t("tb.none");
  applyTheme(document.documentElement.dataset.pref || "auto");
}

/* ================================================================ sheets */

function sheetHead(no, title, meta, id) {
  return h(
    "div",
    { class: "sheet-head" },
    h("span", { class: "sheet-no" }, no),
    h("h2", { class: "sheet-title", id }, title),
    meta ? h("span", { class: "sheet-meta" }, meta) : null,
  );
}

function showSheets(...names) {
  for (const name of SHEET_ORDER) sheets[name].hidden = !names.includes(name);
}

/* The page is drawn from `state`; this is the one place that decides what is shown. */
function renderAll() {
  applyStatic();
  const snap = state.snap;
  renderDataSheet();
  if (!snap) {
    showSheets("data");
    setStage("data");
    return;
  }
  renderProfileSheet(snap);
  renderBriefSheet(snap);
  const shown = ["profile", "brief"];
  if (state.result && (snap.status === "done" || snap.status === "failed")) {
    renderOutcome(snap, state.result);
    shown.push("outcome");
  } else if (snap.status !== "ready") {
    buildAgentSheet();
    applySnapshot(snap, true);
    shown.push("agent");
  }
  showSheets(...shown);
  setStage(stageOf(snap));
}

/* ================================================================ first screen */

function diagram() {
  const label = (text) => esc(text);
  return `
<svg class="diagram" viewBox="0 0 440 372" role="img" aria-label="${esc(t("diagram.aria"))}">
  <g fill="none" stroke-width="2">
    <line class="d-ink" x1="140" y1="10" x2="140" y2="176"/>
    <line class="d-train" x1="140" y1="184" x2="140" y2="366"/>
    <line class="d-test" x1="172" y1="184" x2="172" y2="366" stroke-dasharray="6 5"/>
  </g>
  <rect class="d-node d-ink" x="124" y="30" width="32" height="22"/>
  <text x="140" y="45" text-anchor="middle" class="t-strong">01</text>
  <text x="188" y="45" class="t-strong">dedupe</text>
  <rect class="d-node d-ink" x="124" y="92" width="32" height="22"/>
  <text x="140" y="107" text-anchor="middle" class="t-strong">02</text>
  <text x="188" y="107" class="t-strong">drop_columns</text>
  <text x="124" y="150" text-anchor="end" class="d-label t-dim">&#9650; ${label(t("diagram.all"))}</text>
  <rect class="d-rule" x="0" y="176" width="440" height="6"/>
  <rect class="d-chip" x="212" y="164" width="116" height="30"/>
  <text x="270" y="184" text-anchor="middle" class="t-strong" style="font-size:14px;letter-spacing:.2em">${label(t("diagram.split"))}</text>
  <text x="124" y="214" text-anchor="end" class="d-label t-train t-strong">&#9660; ${label(t("diagram.train"))}</text>
  <rect class="d-node d-train" x="124" y="226" width="32" height="22"/>
  <text x="140" y="241" text-anchor="middle" class="t-train t-strong">03</text>
  <text x="188" y="241" class="t-strong">impute</text>
  <rect class="d-node d-train" x="124" y="272" width="32" height="22"/>
  <text x="140" y="287" text-anchor="middle" class="t-train t-strong">04</text>
  <text x="188" y="287" class="t-strong">encode</text>
  <rect class="d-node d-train" x="124" y="318" width="32" height="22"/>
  <text x="140" y="333" text-anchor="middle" class="t-train t-strong">05</text>
  <text x="188" y="333" class="t-strong">scale</text>
  <text x="112" y="287" text-anchor="end" class="t-train">${label(t("diagram.fit"))}</text>
  <line class="d-test" x1="172" y1="356" x2="182" y2="356" stroke-width="2"/>
  <text x="188" y="360" class="t-test">${label(t("diagram.test"))}</text>
</svg>`;
}

function renderDataSheet() {
  const sheet = sheets.data;
  clear(sheet);
  if (state.snap) return;

  const limit = fmtBytes(state.config.max_upload_bytes);
  const input = h("input", {
    type: "file",
    accept: ".csv,text/csv",
    id: "file-input",
    "aria-label": t("dz.aria"),
    onchange: (event) => uploadFile(event.target.files[0]),
  });
  const dropzone = h(
    "div",
    { class: "dropzone" },
    input,
    h("span", { class: "dz-title" }, t("dz.title")),
    h("span", { class: "dz-sub" }, t("dz.sub", { limit })),
  );
  for (const type of ["dragenter", "dragover"]) dropzone.addEventListener(type, () => dropzone.classList.add("over"));
  for (const type of ["dragleave", "drop"]) dropzone.addEventListener(type, () => dropzone.classList.remove("over"));

  const samples = state.config.samples;
  const sampleList = samples.length
    ? h(
        "ul",
        { class: "samples" },
        samples.map((sample) => {
          const note = sample.id === "customers" ? t("sample.customers.note") : sample.note;
          const meta = [
            sample.n_rows ? `${fmtInt(sample.n_rows)} × ${sample.n_columns}` : null,
            sample.task_type,
            sample.target ? `${t("sum.target").toLowerCase()} ${sample.target}` : null,
          ]
            .filter(Boolean)
            .join(" · ");
          return h(
            "li",
            {},
            h(
              "button",
              { type: "button", class: "sample", onclick: () => useSample(sample) },
              h("span", { class: "sample-name" }, sample.label),
              h("span", { class: "sample-go" }, t("sample.use")),
              h("span", { class: "sample-meta" }, meta),
              note ? h("span", { class: "sample-note" }, note) : null,
            ),
          );
        }),
      )
    : h("p", { class: "hint" }, t("sample.none"));

  const art = h("div", {});
  // The diagram is built from this file's constants and dictionary texts only; no data from the
  // server or a CSV file goes into it.
  art.innerHTML = diagram();
  put(
    sheet,
    h(
      "div",
      { class: "hero" },
      h(
        "div",
        {},
        h("p", { class: "eyebrow" }, t("hero.eyebrow")),
        h("h1", { id: "h-data" }, t("hero.title_a"), h("em", {}, t("hero.title_b"))),
        h("p", { class: "lede" }, t("hero.lede")),
        h("div", { class: "privacy" }, h("strong", {}, t("privacy.title")), h("p", {}, t("privacy.body"))),
      ),
      art,
    ),
    h(
      "div",
      { class: "intake" },
      h("div", {}, h("p", { class: "label" }, t("intake.upload")), dropzone),
      h("div", {}, h("p", { class: "label" }, t("intake.sample")), sampleList),
    ),
  );
}

async function uploadFile(file) {
  if (!file) return;
  if (file.size > state.config.max_upload_bytes) {
    showBanner({
      title: t("b.too_large_title"),
      message: t("b.too_large", { name: file.name, size: fmtBytes(file.size), limit: fmtBytes(state.config.max_upload_bytes) }),
    });
    return;
  }
  clearBanner();
  const form = new FormData();
  form.append("file", file);
  await openSession(() => api("/api/sessions", { method: "POST", body: form }), null);
}

async function useSample(sample) {
  clearBanner();
  await openSession(() => postJson("/api/sessions/sample", { id: sample.id }), sample);
}

async function openSession(create, sample) {
  sheets.data.setAttribute("aria-busy", "true");
  try {
    const snap = await create();
    state.snap = snap;
    state.result = null;
    const goal = sample ? (lang === "tr" && sample.goal_tr ? sample.goal_tr : sample.goal) : "";
    state.brief = {
      goal,
      target: sample && sample.target ? sample.target : "",
      family: "",
      prefilled: sample && goal ? { sample, text: goal } : null,
    };
    renderAll();
    announce(
      t("loaded.announce", {
        file: snap.filename,
        rows: tn("rows", snap.profile.n_rows),
        columns: tn("columns", snap.profile.n_columns),
      }),
    );
    scrollToEl(sheets.profile, false);
  } catch (error) {
    showBanner({ title: t("b.cannot_read"), message: errorText(error) });
  } finally {
    sheets.data.removeAttribute("aria-busy");
  }
}

function startOver() {
  stopPolling();
  stopClock();
  const id = state.snap && state.snap.id;
  if (id) fetch(`/api/sessions/${id}`, { method: "DELETE" }).catch(() => {});
  state.generation += 1;
  Object.assign(state, {
    snap: null,
    result: null,
    questionSig: "",
    questions: null,
    qChoices: {},
    brief: { goal: "", target: "", family: "", prefilled: null },
  });
  clearBanner();
  for (const name of ["profile", "brief", "agent", "outcome"]) clear(sheets[name]);
  renderAll();
  window.scrollTo({ top: 0 });
}

/* ================================================================ 01 profile */

// Warnings that say a column is a candidate to be dropped. The others only inform.
const CANDIDATE_WARNINGS = { possible_pii: "alert", possible_id: "cand", constant: "cand", high_missing: "cand" };

function warningTag(code) {
  const kind = CANDIDATE_WARNINGS[code] || "";
  return h("span", { class: `tag ${kind}`, title: lookup(`warn.${code}`) || code }, code);
}

function renderProfileSheet(snap) {
  const sheet = sheets.profile;
  clear(sheet);
  const p = snap.profile;
  const rows = p.columns.map((c) => {
    const pii = c.warnings.includes("possible_pii");
    const shown = c.sample_values.slice(0, 3);
    const isTarget = c.name === state.brief.target;
    return h(
      "tr",
      { "data-column": c.name, class: `${pii ? "has-pii" : ""} ${isTarget ? "is-target" : ""}` },
      h("td", { class: "name" }, c.name, isTarget ? [" ", h("span", { class: "tag train" }, t("col.target"))] : null),
      h("td", { "data-label": t("col.type") }, h("span", { class: "tag" }, c.inferred_type)),
      h(
        "td",
        { class: "num", "data-label": t("col.missing") },
        c.n_missing ? pct(c.missing_ratio) : "0",
        c.n_missing
          ? h(
              "span",
              { class: `missbar ${c.warnings.includes("high_missing") ? "high" : ""}`, "aria-hidden": "true" },
              h("i", { style: `width:${Math.max(4, c.missing_ratio * 100)}%` }),
            )
          : null,
      ),
      h("td", { class: "num", "data-label": t("col.unique") }, fmtInt(c.n_unique)),
      h("td", { "data-label": t("col.warnings") }, h("div", { class: "warns" }, c.warnings.map(warningTag))),
      h("td", { class: "samplevals", "data-label": t("col.samples") }, pii ? t("col.hidden") : shown.length ? shown.join(", ") : "—"),
    );
  });

  const meta = [
    h("span", {}, [snap.filename, tn("rows", p.n_rows), tn("columns", p.n_columns), tn("dups", p.n_duplicate_rows)].join(" · ")),
    h("button", { type: "button", class: "btn quiet small", onclick: startOver }, t("profile.another")),
  ];
  put(
    sheet,
    sheetHead("01", t("profile.title"), meta, "h-profile"),
    h(
      "div",
      { class: "tablewrap" },
      h(
        "table",
        { class: "ptable" },
        h("caption", { class: "sr-only" }, t("profile.caption")),
        h(
          "thead",
          {},
          h("tr", {}, ["col.column", "col.type", "col.missing", "col.unique", "col.warnings", "col.samples"].map((k) => h("th", { scope: "col" }, t(k)))),
        ),
        h("tbody", {}, rows),
      ),
    ),
    h(
      "p",
      { class: "legend-line" },
      h("span", { class: "tag cand" }, t("warn.legend_candidate")),
      " ",
      h("span", { class: "tag" }, t("warn.legend_info")),
    ),
  );
}

function markTarget(name) {
  state.brief.target = name;
  for (const row of sheets.profile.querySelectorAll("tbody tr")) {
    const isTarget = row.dataset.column === name;
    row.classList.toggle("is-target", isTarget);
    const cell = row.firstChild;
    const mark = cell.querySelector(".tag");
    if (isTarget && !mark) put(cell, " ", h("span", { class: "tag train" }, t("col.target")));
    if (!isTarget && mark) mark.remove();
  }
}

/* ================================================================ 02 brief */

const FAMILIES = [
  ["", "fam.undecided", "fam.hint_undecided"],
  ["tree", "fam.tree", "fam.hint_tree"],
  ["linear", "fam.linear", "fam.hint_linear"],
  ["any", "fam.any", "fam.hint_any"],
];

function renderBriefSheet(snap) {
  const sheet = sheets.brief;
  clear(sheet);
  put(sheet, sheetHead("02", t("brief.title"), null, "h-brief"));
  if (snap.status !== "ready") {
    put(sheet, briefSummary());
    return;
  }
  const b = state.brief;

  const goal = h("textarea", {
    class: "textarea",
    id: "goal",
    name: "goal",
    rows: "3",
    required: true,
    "aria-describedby": "goal-hint goal-error",
    placeholder: t("brief.goal_placeholder"),
    oninput: (event) => {
      b.goal = event.target.value;
    },
  });
  goal.value = b.goal;
  const goalError = h("p", { class: "hint field-error", id: "goal-error", role: "alert" });

  const target = h(
    "select",
    { class: "select", id: "target", name: "target", onchange: (event) => markTarget(event.target.value) },
    h("option", { value: "" }, t("brief.target_auto")),
    snap.profile.columns.map((c) => h("option", { value: c.name }, c.name)),
  );
  target.value = snap.profile.columns.some((c) => c.name === b.target) ? b.target : "";

  const familyHint = h("p", { class: "hint", id: "family-hint" });
  const familyInputs = FAMILIES.map(([value, label, hint]) => {
    const input = h("input", {
      type: "radio",
      name: "family",
      value,
      checked: b.family === value,
      onchange: () => {
        b.family = value;
        familyHint.textContent = t(hint);
      },
    });
    return h("label", { class: "seg-opt" }, input, h("span", {}, t(label)));
  });
  familyHint.textContent = t(FAMILIES.find(([value]) => value === b.family)[2]);

  const form = h(
    "form",
    {
      class: "brief",
      novalidate: true,
      onsubmit: (event) => {
        event.preventDefault();
        startRun(goal, goalError);
      },
    },
    h(
      "div",
      {},
      h("div", { class: "field" }, h("label", { class: "label", for: "goal" }, t("brief.goal")), goal, h("p", { class: "hint", id: "goal-hint" }, t("brief.goal_hint")), goalError),
      h("div", { class: "field" }, h("label", { class: "label", for: "target" }, t("brief.target")), target, h("p", { class: "hint" }, t("brief.target_hint"))),
      h("fieldset", { class: "field" }, h("legend", { class: "label" }, t("brief.family")), h("div", { class: "seg", role: "presentation" }, familyInputs), familyHint),
      lang === "tr" && hasText("brief.langnote") ? h("p", { class: "hint note-line" }, t("brief.langnote")) : null,
      h("div", { class: "actions" }, h("button", { type: "submit", class: "btn primary", id: "plan-button" }, t("brief.plan")), h("span", { class: "keyhints" }, h("kbd", {}, "Ctrl"), " + ", h("kbd", {}, "Enter"))),
    ),
    h(
      "div",
      {},
      h("p", { class: "label" }, t("brief.next")),
      h("ol", { class: "hint next-list" }, [1, 2, 3, 4].map((n) => h("li", {}, t(`brief.next_${n}`)))),
    ),
  );
  goal.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  put(sheet, form);
}

function briefSummary() {
  const b = state.brief;
  const family = b.family ? t(`fam.${b.family}`) : t("sum.family_auto");
  const pair = (label, value, cls) => h("span", { class: "sum-item" }, h("b", {}, label), h("span", { class: cls || "" }, value));
  return h(
    "div",
    { class: "summary" },
    h(
      "p",
      { class: "sum-line" },
      pair(t("sum.goal"), b.goal, "sum-goal"),
      pair(t("sum.target"), b.target || t("sum.target_auto"), b.target ? "mono" : ""),
      pair(t("sum.family"), family),
    ),
    h("button", { type: "button", class: "btn quiet small", onclick: editBrief }, t("sum.edit")),
  );
}

async function startRun(goalEl, goalError) {
  const b = state.brief;
  const goal = goalEl.value.trim();
  goalError.textContent = "";
  if (!goal) {
    goalError.textContent = t("brief.goal_error");
    goalEl.focus();
    return;
  }
  b.goal = goal;
  clearBanner();
  try {
    const snap = await postJson(`/api/sessions/${state.snap.id}/plan`, {
      goal,
      target: b.target || null,
      model_family: b.family || null,
      lang,
    });
    state.snap = snap;
    state.result = null;
    state.questionSig = "";
    state.questions = null;
    state.qChoices = {};
    clear(sheets.outcome);
    renderBriefSheet(snap);
    buildAgentSheet();
    showSheets("profile", "brief", "agent");
    applySnapshot(snap);
    scrollToEl(sheets.agent, false);
    startPolling();
  } catch (error) {
    showBanner({ title: t("b.cannot_start"), message: errorText(error) });
  }
}

/* "edit": drop the run, keep the file, open the brief again with what was typed. */
async function editBrief() {
  stopPolling();
  stopClock();
  state.generation += 1;
  try {
    state.snap = await api(`/api/sessions/${state.snap.id}/reset`, { method: "POST" });
  } catch (error) {
    showBanner({ title: t("b.edit_failed"), message: errorText(error) });
    return;
  }
  Object.assign(state, { result: null, questionSig: "", questions: null, qChoices: {} });
  clearBanner();
  clear(sheets.agent);
  clear(sheets.outcome);
  renderBriefSheet(state.snap);
  showSheets("profile", "brief");
  setStage("brief");
  scrollToEl(sheets.brief, false);
  const goal = document.getElementById("goal");
  if (goal) goal.focus({ preventScroll: true });
}

/* ================================================================ 03 agent */

const PHASES = ["planning", "validating", "revising", "baseline", "saving"];

let agentParts = null;

function buildAgentSheet() {
  const sheet = sheets.agent;
  clear(sheet);
  const phase = h("p", { class: "phase", id: "phase" });
  const detail = h("p", { class: "lede" });
  const calls = h("span", {});
  const clock = h("span", {});
  const running = h(
    "div",
    { class: "agent-running" },
    phase,
    detail,
    h("div", { class: "loader", "aria-hidden": "true" }),
    h("div", { class: "runmeta" }, h("span", {}, t("agent.elapsed"), " ", clock), h("span", {}, t("agent.calls"), " ", calls)),
  );
  const history = h("div", {});
  const questions = h("div", {});
  const failure = h("div", {});
  put(sheet, sheetHead("03", t("agent.title"), null, "h-agent"), history, running, questions, failure);
  agentParts = { running, phase, detail, calls, clock, history, questions, failure };
}

function renderHistory(qa) {
  clear(agentParts.history);
  if (!qa.length) return;
  put(
    agentParts.history,
    h("p", { class: "label" }, t("q.answered")),
    h("ul", { class: "history" }, qa.map((r) => h("li", {}, inlineMarkdown(r.question.question), " → ", h("span", { class: "a" }, r.answer)))),
  );
}

function applySnapshot(snap, rerender = false) {
  const previous = state.snap && state.snap.status;
  state.snap = snap;
  if (!agentParts) return;
  renderHistory(snap.qa);
  agentParts.calls.textContent = snap.llm_calls;

  if (snap.status === "running") {
    const phase = PHASES.includes(snap.phase) ? snap.phase : null;
    const title = phase ? t(`phase.${phase}`) : t("phase.working");
    agentParts.phase.textContent = title;
    agentParts.phase.classList.toggle("revising", phase === "revising");
    agentParts.detail.textContent = phase ? t(`phase.${phase}_detail`) : "";
    agentParts.running.hidden = false;
    clear(agentParts.questions);
    state.questionSig = "";
    state.questions = null;
    state.clock = { base: snap.elapsed_seconds || 0, at: performance.now() };
    startClock();
    setStage("running");
    if (!rerender && (previous !== "running" || phase === "revising")) announce(title);
  } else if (snap.status === "waiting") {
    agentParts.running.hidden = true;
    stopClock();
    renderQuestions(snap, rerender);
    setStage("question");
  } else if (snap.status === "error") {
    agentParts.running.hidden = true;
    stopClock();
    renderAgentError(snap.error, rerender);
  } else {
    stopClock();
  }
}

function startClock() {
  if (state.clockTimer) return;
  const tick = () => {
    if (state.clock && agentParts) agentParts.clock.textContent = fmtClock(state.clock.base + (performance.now() - state.clock.at) / 1000);
  };
  tick();
  state.clockTimer = window.setInterval(tick, 1000);
}

function stopClock() {
  window.clearInterval(state.clockTimer);
  state.clockTimer = null;
}

/* ---------------------------------------------------------------- questions */

const OTHER = "__other__";

function renderQuestions(snap, force) {
  const sig = `${snap.questions.map((q) => q.id).join("|")}:${snap.qa.length}`;
  if (sig === state.questionSig && !force) return;
  if (sig !== state.questionSig) state.qChoices = {};
  state.questionSig = sig;
  const wrap = agentParts.questions;
  clear(wrap);

  const n = snap.questions.length;
  const entries = snap.questions.map((q, i) => buildQuestion(q, i, n));
  state.questions = entries;
  state.activeQ = 0;
  entries[0].card.classList.add("active");

  put(
    wrap,
    h("p", { class: "lede" }, tn("q.lede", n)),
    entries.map((e) => e.card),
    h(
      "div",
      { class: "qfoot" },
      h("button", { type: "button", class: "btn primary", onclick: submitAnswers }, tn("q.send", n)),
      h("span", { class: "keyhints" }, h("kbd", {}, "1"), "–", h("kbd", {}, "9"), ` ${t("q.hint_choose")} · `, h("kbd", {}, "Enter"), ` ${t("q.hint_send")}`),
    ),
  );
  if (!force) {
    announce(tn("q.announce", n));
    entries[0].card.focus({ preventScroll: true });
    scrollToEl(entries[0].card, false);
  }
}

function buildQuestion(q, i, total) {
  const name = `q-${i}`;
  const rows = [];
  const radios = [];
  const saved = state.qChoices[q.id] || { value: q.recommended, other: "" };

  const otherText = h("input", {
    type: "text",
    class: "input other-text",
    "aria-label": t("q.other_aria", { question: q.question }),
    placeholder: t("q.other_placeholder"),
    hidden: true,
    oninput: (event) => {
      state.qChoices[q.id].other = event.target.value;
    },
  });
  otherText.value = saved.other;

  const select = (value) => {
    for (const r of radios) {
      r.input.checked = r.value === value;
      r.li.classList.toggle("checked", r.input.checked);
    }
    otherText.hidden = value !== OTHER;
    state.qChoices[q.id] = { value, other: otherText.value };
  };

  const addRow = (value, key, content, extra) => {
    const input = h("input", { type: "radio", name, value, onchange: () => select(value), onfocus: () => setActive(i) });
    const li = h("li", { class: "opt" }, h("label", {}, h("span", { class: "key", "aria-hidden": "true" }, key), h("span", { class: "opt-text" }, content), extra, input));
    radios.push({ value, input, li });
    rows.push(li);
  };

  q.options.forEach((value, k) => addRow(value, k + 1, value, value === q.recommended ? h("span", { class: "tag" }, t("q.recommended")) : null));
  if (q.allow_custom) addRow(OTHER, q.options.length + 1, t("q.other"), null);

  const card = h(
    "section",
    { class: "qcard", tabindex: "-1", "aria-labelledby": `qt-${i}`, onfocusin: () => setActive(i), onclick: () => setActive(i) },
    h("p", { class: "q-meta" }, t("q.meta", { i: i + 1, n: total })),
    h("h3", { class: "q-text", id: `qt-${i}` }, inlineMarkdown(q.question)),
    h("p", { class: "q-why" }, h("b", {}, `${t("q.why")} `), inlineMarkdown(q.reason)),
    h("fieldset", {}, h("legend", { class: "sr-only" }, q.question), h("ul", { class: "opts" }, rows)),
    q.allow_custom ? h("div", { class: "other-input" }, otherText) : null,
  );
  select(radios.some((r) => r.value === saved.value) ? saved.value : q.recommended);
  otherText.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submitAnswers();
    }
  });
  return { q, card, radios, otherText, select };
}

function setActive(i) {
  if (!state.questions || state.activeQ === i) return;
  state.activeQ = i;
  state.questions.forEach((e, k) => e.card.classList.toggle("active", k === i));
}

function chooseByKey(number) {
  const entry = state.questions[state.activeQ];
  const radio = entry.radios[number - 1];
  if (!radio) return;
  entry.select(radio.value);
  if (radio.value === OTHER) {
    entry.otherText.focus();
    return;
  }
  radio.input.focus({ preventScroll: true });
  announce(t("q.selected", { value: radio.value }));
  if (state.activeQ < state.questions.length - 1) {
    const next = state.questions[state.activeQ + 1];
    setActive(state.activeQ + 1);
    next.card.focus({ preventScroll: true });
    next.card.scrollIntoView({ block: "nearest", behavior: reducedMotion.matches ? "auto" : "smooth" });
  }
}

async function submitAnswers() {
  if (!state.questions) return;
  const answers = {};
  for (const entry of state.questions) {
    const chosen = entry.radios.find((r) => r.input.checked).value;
    if (chosen === OTHER) {
      const text = entry.otherText.value.trim();
      if (!text) {
        entry.otherText.focus();
        showBanner({ title: t("q.empty_title"), message: t("q.empty_body", { question: entry.q.question }) });
        return;
      }
      answers[entry.q.id] = text;
    } else {
      answers[entry.q.id] = chosen;
    }
  }
  clearBanner();
  try {
    const snap = await postJson(`/api/sessions/${state.snap.id}/answers`, { answers });
    state.questions = null;
    applySnapshot(snap);
    startPolling();
  } catch (error) {
    showBanner({ title: t("q.rejected"), message: errorText(error) });
  }
}

document.addEventListener("keydown", (event) => {
  if (!state.questions || event.ctrlKey || event.metaKey || event.altKey) return;
  const target = event.target;
  const typing = target.matches && target.matches("input[type=text], textarea, select");
  if (typing) {
    if (event.key === "Escape") target.blur();
    return;
  }
  if (/^[1-9]$/.test(event.key)) {
    event.preventDefault();
    chooseByKey(Number(event.key));
  } else if (event.key === "Enter" && !(target.matches && target.matches("button, a, summary"))) {
    event.preventDefault();
    submitAnswers();
  }
});

/* ---------------------------------------------------------------- polling */

function startPolling() {
  stopPolling();
  const generation = state.generation;
  const poll = async () => {
    if (generation !== state.generation) return;
    let snap;
    try {
      snap = await api(`/api/sessions/${state.snap.id}`);
    } catch (error) {
      if (generation !== state.generation) return;
      if (error.status === 404) {
        showBanner({
          title: t("b.gone_title"),
          message: t("b.gone"),
          action: h("button", { type: "button", class: "btn", onclick: startOver }, t("b.start_over")),
        });
        return;
      }
      showBanner({ title: t("b.lost_title"), message: t("b.lost", { message: errorText(error) }), kind: "info" });
      state.pollTimer = window.setTimeout(poll, 2500);
      return;
    }
    if (generation !== state.generation) return;
    if (document.querySelector(".banner.info")) clearBanner();
    if (snap.status === "running" || snap.status === "waiting") {
      applySnapshot(snap);
      state.pollTimer = window.setTimeout(poll, snap.status === "waiting" ? 1500 : 800);
    } else {
      await finishRun(snap, generation);
    }
  };
  state.pollTimer = window.setTimeout(poll, 400);
}

function stopPolling() {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

async function finishRun(snap, generation) {
  applySnapshot(snap);
  agentParts.running.hidden = true;
  clear(agentParts.questions);
  state.questions = null;
  if (snap.status === "error") {
    setStage("error");
    announce(t("err.announce"));
    return;
  }
  try {
    const result = await api(`/api/sessions/${snap.id}/result`);
    if (generation !== state.generation) return;
    state.result = result;
    renderOutcome(snap, result);
    showSheets("profile", "brief", "outcome");
    setStage(result.status === "ok" ? "outcome" : "failed");
    announce(t(result.status === "ok" ? "plan_ready.announce" : "failed.announce"));
    scrollToEl(sheets.outcome, true);
  } catch (error) {
    showBanner({ title: t("b.result_failed"), message: errorText(error) });
  }
}

function renderAgentError(message, rerender) {
  clear(agentParts.failure);
  put(
    agentParts.failure,
    h("p", { class: "verdict failed small" }, t("err.stopped")),
    h("p", { class: "verdict-sub" }, message),
    h("p", { class: "hint" }, t("err.nothing_saved")),
  );
  if (!rerender) scrollToEl(sheets.agent, false);
}

/* ================================================================ 04 outcome */

function renderOutcome(snap, result) {
  const sheet = sheets.outcome;
  clear(sheet);
  const ok = result.status === "ok";

  const nav = [];
  const blocks = [];
  const addBlock = (anchor, label, node) => {
    if (!node) return;
    nav.push(h("a", { href: `#${anchor}` }, label));
    blocks.push(node);
  };

  addBlock("b-plan", t(ok ? "nav.plan" : "nav.last"), result.plan ? planBlock(result, ok) : null);
  addBlock("b-fix", t("nav.fix"), result.revisions.length ? correctionBlock(result) : null);
  addBlock("b-drop", t("nav.drop"), result.plan ? droppedBlock(result) : null);
  addBlock("b-base", t("nav.base"), ok ? baselineBlock(result) : null);
  addBlock("b-use", t("nav.use"), usageBlock(result, snap.id));
  addBlock("b-trace", t("nav.trace"), traceBlock(result, !ok));
  addBlock("b-report", t("nav.report"), reportBlock(result));

  put(
    sheet,
    sheetHead("04", t(ok ? "out.title" : "out.failed_title"), null, "h-outcome"),
    outcomeStrip(snap, result, ok),
    h("nav", { class: "subnav", "aria-label": t("nav.aria") }, nav),
    blocks,
  );
}

function outcomeStrip(snap, result, ok) {
  const run = result.run || {};
  const rows = run.rows || {};
  const cols = run.columns || {};
  const revisions = result.revisions.length;

  let sub;
  if (!ok) sub = result.error || t("out.fail_default");
  else {
    sub = t("out.sub", { train: tn("rows", rows.train), test: tn("rows", rows.test), columns: tn("columns", cols.train) });
    sub += " " + (revisions ? tn("out.sub_rev", revisions) : t("out.sub_clean"));
  }

  const actions = h(
    "div",
    { class: "strip-actions" },
    h("a", { class: "btn primary", href: `/api/sessions/${snap.id}/download`, download: "" }, t("out.download")),
    h("button", { type: "button", class: "btn", onclick: editBrief }, t("out.edit")),
  );

  const usage = result.llm.usage;
  const facts = h(
    "dl",
    { class: "facts" },
    fact(t("fact.input"), rows.before != null ? `${fmtInt(rows.before)} × ${cols.before}` : "—", t("fact.rowscols")),
    ok ? fact(t("fact.prepared"), `${fmtInt(rows.train + rows.test)} × ${cols.train}`, t("fact.rowscols")) : fact(t("fact.prepared"), t("common.none"), t("fact.no_sets")),
    fact(t("fact.calls"), String(result.llm.calls.length), result.llm.model || t("fact.no_model")),
    fact(t("fact.tokens"), usage ? fmtInt(usage.input_tokens + usage.output_tokens) : noUsage(), tokenNote(usage)),
  );

  const parts = [
    h("div", { class: "strip" }, h("div", {}, h("p", { class: `verdict ${ok ? "" : "failed"}` }, t(ok ? "out.ok" : "out.fail")), h("p", { class: "verdict-sub" }, sub)), actions),
    facts,
  ];

  if (ok && rows.train != null) {
    const total = rows.train + rows.test;
    parts.push(
      h(
        "div",
        { role: "img", "aria-label": t("split.aria", { train: tn("rows", rows.train), test: tn("rows", rows.test) }) },
        h(
          "div",
          { class: "splitbar" },
          h("div", { class: "train", style: `width:${(rows.train / total) * 100}%` }, t("split.train_label", { n: fmtInt(rows.train) })),
          h("div", { class: "test", style: `width:${(rows.test / total) * 100}%` }, fmtInt(rows.test)),
        ),
      ),
      h("div", { class: "splitcap" }, h("span", { class: "cap-train" }, t("split.cap_train")), h("span", { class: "cap-test" }, t("split.cap_test"))),
    );
  }
  return h("div", {}, parts);
}

// A scripted planner uses no tokens: the demo says so instead of showing a number.
const noUsage = () => t(state.config.fake ? "fact.demo" : "common.na");

// What a run used, as one line: input, read from the cache (when the provider says), output.
function tokenNote(usage) {
  if (!usage) return state.config.fake ? "" : t("fact.not_reported");
  const vars = { in: fmtInt(usage.input_tokens), out: fmtInt(usage.output_tokens) };
  if (usage.cached_input_tokens == null) return t("fact.in_out", vars);
  return t("fact.in_cached_out", { ...vars, cached: fmtInt(usage.cached_input_tokens) });
}

const fact = (label, value, note) => h("div", { class: "fact" }, h("dt", {}, label), h("dd", {}, value, note ? h("small", {}, note) : null));

function blockHead(title, meta, id) {
  return h("div", { class: "block-head" }, h("h3", { id }, title), meta ? h("span", { class: "sheet-meta" }, meta) : null);
}

/* ---------------------------------------------------------------- plan */

function planBlock(result, ok) {
  const plan = result.plan;
  const issues = result.issues;
  const errors = issues.filter((i) => i.severity === "error").length;
  const warnings = issues.length - errors;
  const meta = t("plan.meta", { target: plan.target, task: plan.task_type, family: plan.model_family });

  const loose = issues.filter((i) => i.step_index == null);
  const parts = [
    blockHead(t(ok ? "plan.title" : "plan.title_failed"), meta, "b-plan-h"),
    h("p", { class: "lede" }, t("plan.lede")),
    lang === "tr" && hasText("plan.en_note") ? h("p", { class: "hint note-line" }, t("plan.en_note")) : null,
    planFlow(plan, issues),
  ];
  if (loose.length) parts.push(h("p", { class: "findings-title" }, t("findings.title")), findingsList(loose));
  if (!issues.length) {
    parts.push(h("p", { class: "clean" }, t("plan.clean")));
  } else if (!loose.length) {
    parts.push(h("p", { class: "hint after-flow" }, t("plan.counts", { errors: tn("errors", errors), warnings: tn("warnings", warnings) })));
  }
  return h("section", { class: "block", id: "b-plan" }, parts);
}

function findingsList(issues) {
  return h(
    "ul",
    { class: "findings" },
    issues.map((i) =>
      h(
        "li",
        { class: i.severity },
        h("span", { class: "note-code" }, i.code, i.column ? [" · ", h("span", { class: "chip" }, i.column)] : null, h("span", { class: "sr-only" }, ` (${t(`sev.${i.severity}_sr`)})`)),
        h("span", {}, i.message),
      ),
    ),
  );
}

function sideOf(plan, index) {
  const split = plan.steps.findIndex((s) => s.kind === "split");
  if (split === -1) return "before";
  return index === split ? "split" : index < split ? "before" : "after";
}

function planFlow(plan, issues) {
  const byStep = new Map();
  for (const issue of issues) {
    if (issue.step_index == null) continue;
    if (!byStep.has(issue.step_index)) byStep.set(issue.step_index, []);
    byStep.get(issue.step_index).push(issue);
  }
  const split = plan.steps.findIndex((s) => s.kind === "split");
  const above = split === -1 ? plan.steps : plan.steps.slice(0, split);
  const below = split === -1 ? [] : plan.steps.slice(split + 1);
  let order = 0;

  const items = (steps, zone) =>
    h(
      "ol",
      {},
      steps.map((step) => {
        const leak = (zone === "all" && step.kind === "stateful") || (zone === "train" && step.kind === "stateless");
        return stepItem(step, plan, byStep.get(step.index) || [], order++, leak);
      }),
    );

  const flow = h("div", { class: "flow" });
  if (above.length || split === -1) {
    put(flow, h("div", { class: "zone zone-all" }, h("div", { class: "zone-cap" }, h("b", {}, t("zone.all")), t("zone.all_text")), items(above, "all")));
  }
  if (split === -1) {
    put(flow, h("p", { class: "nosplit" }, h("b", {}, t("nosplit.title")), " ", t("nosplit.body")));
    return flow;
  }
  put(flow, splitLine(plan.steps[split], byStep.get(split) || [], order++));
  put(
    flow,
    h(
      "div",
      { class: "zone zone-train" },
      h(
        "div",
        { class: "zone-cap" },
        h("b", {}, t("zone.train")),
        t("zone.train_text"),
        h("span", { class: "legend" }, h("span", {}, h("i", { class: "l-train" }), t("legend.train")), h("span", {}, h("i", { class: "l-test" }), t("legend.test"))),
      ),
      below.length ? items(below, "train") : h("p", { class: "hint zone-empty" }, t("zone.empty")),
    ),
  );
  return flow;
}

function issueNote(issue) {
  return h(
    "div",
    { class: `note ${issue.severity}` },
    h("span", { class: "note-code" }, `${t(`sev.${issue.severity}`)} `, issue.code, issue.column ? [" · ", h("span", { class: "chip" }, issue.column)] : null),
    issue.message,
  );
}

function stepItem(step, plan, issues, order, leak) {
  const params = Object.entries(step.parameters);
  const kindTag = h("span", { class: `tag ${step.kind === "stateful" ? "train" : ""}` }, t(step.kind === "stateful" ? "kind.stateful" : "kind.stateless"));
  return h(
    "li",
    { class: `step ${leak ? "leak" : ""}`, "--i": order },
    h("div", { class: "rail", "aria-hidden": "true" }, h("span", { class: "node" }, pad2(step.index))),
    h(
      "div",
      { class: "step-body" },
      h(
        "div",
        { class: "step-head" },
        h("h4", { class: "step-title" }, lookup(`step.${step.op}`) || step.op),
        h("span", { class: "op-code" }, step.op),
        kindTag,
        leak ? h("span", { class: "tag alert" }, t(step.kind === "stateful" ? "kind.leak_up" : "kind.leak_down")) : null,
      ),
      step.all_columns || step.columns.length
        ? h("div", { class: "cols" }, step.all_columns ? h("span", { class: "all" }, t("step.all_columns")) : step.columns.map((c) => h("span", { class: `chip ${c === plan.target ? "target" : ""}` }, c)))
        : null,
      params.length ? h("div", { class: "params" }, params.map(([k, v]) => h("span", {}, h("b", {}, k), String(v)))) : null,
      h("p", { class: "reason" }, step.reason),
    ),
    h("div", { class: "margin" }, issues.map(issueNote)),
  );
}

function splitLine(step, issues, order) {
  const p = step.parameters;
  const params = t("split.params", {
    step: pad2(step.index),
    pct: Math.round(p.test_size * 100),
    stratify: t(p.stratify ? "split.stratified" : "split.unstratified"),
    seed: p.random_state,
  });
  return h(
    "div",
    { class: "splitline", "--i": order },
    h("div", { class: "split-upper" }, h("div", { class: "split-above" }, h("b", {}, t("split.above")), t("split.above_text"))),
    h("div", { class: "split-rule" }, h("span", { class: "split-chip" }, t("split.chip")), h("span", { class: "split-params" }, params)),
    h(
      "div",
      { class: "split-lower" },
      h("div", { class: "split-params-m" }, params),
      h("div", { class: "split-below" }, h("b", {}, t("split.below")), t("split.below_text")),
      h("div", { class: "split-note" }, h("p", { class: "reason" }, step.reason), h("div", { class: "margin" }, issues.map(issueNote))),
    ),
  );
}

/* ---------------------------------------------------------------- correction */

function describeChange(from, to) {
  const notes = [];
  const gone = from.columns.filter((c) => !to.columns.includes(c));
  const added = to.columns.filter((c) => !from.columns.includes(c));
  if (gone.length) notes.push(`− ${gone.join(", ")}`);
  if (added.length) notes.push(`+ ${added.join(", ")}`);
  for (const key of new Set([...Object.keys(from.parameters), ...Object.keys(to.parameters)])) {
    if (JSON.stringify(from.parameters[key]) !== JSON.stringify(to.parameters[key])) {
      const none = t("change.none");
      notes.push(`${key}: ${from.parameters[key] ?? none} → ${to.parameters[key] ?? none}`);
    }
  }
  return notes.join(" · ") || t("change.rewritten");
}

function ledgerColumn(kind, title, sub, steps, annotate) {
  const column = h("div", { class: `lcol ${kind}` }, h("h4", {}, title, h("span", { class: "sub" }, sub)));
  let target = column;
  for (const step of steps) {
    if (step.kind === "split") {
      put(column, h("div", { class: "lsplit" }, h("span", {}, t("split.chip"))));
      target = h("div", { class: "lzone below" });
      put(column, target);
      continue;
    }
    put(target, ledgerRow(step, annotate(step)));
  }
  return column;
}

function ledgerRow(step, { cls = "", tag = null, issues = [] }) {
  return h(
    "div",
    { class: `lrow ${cls}` },
    h("span", { class: "n" }, pad2(step.index)),
    h("div", {}, h("span", { class: "op" }, step.op), step.columns.length ? h("div", { class: "cols" }, step.columns.map((c) => h("span", { class: "chip" }, c))) : null),
    tag ? h("span", { class: "tag fate" }, tag) : null,
    issues.map((i) => h("div", { class: "lrow-issue" }, h("span", { class: "note-code" }, i.code), i.message)),
  );
}

function correctionBlock(result) {
  const revs = result.revisions;
  const parts = [blockHead(t("fix.title"), tn("revisions", revs.length), "b-fix-h"), h("p", { class: "lede" }, t("fix.lede"))];
  revs.forEach((rev, index) => {
    const rejectedPlan = { steps: rev.rejected_steps };
    const left = ledgerColumn("rejected", t("ledger.rejected"), t("ledger.from_call", { n: rev.from_call }), rev.rejected_steps, (step) => {
      const side = sideOf(rejectedPlan, step.index);
      const to = rev.corrected_steps[step.to_index];
      let tag = null;
      if (step.fate === "moved") tag = t(side === "before" ? "fate.down" : "fate.up");
      else if (step.fate === "changed") tag = t("fate.changed", { what: describeChange(step, to) });
      else if (step.fate === "removed") tag = t("fate.removed");
      return { cls: step.issues.length ? "bad" : "", tag: step.issues.length ? tag : null, issues: step.issues };
    });
    const right = ledgerColumn("corrected", t("ledger.corrected"), t("ledger.from_call", { n: rev.to_call }), rev.corrected_steps, (step) => {
      let tag = null;
      if (step.origin === "moved") tag = t("origin.moved", { n: pad2(step.from_index) });
      else if (step.origin === "changed") tag = t("origin.changed", { n: pad2(step.from_index) });
      else if (step.origin === "new") tag = t("origin.new");
      const flagged = step.origin === "moved" || step.origin === "changed";
      return { cls: flagged ? "fixed" : "", tag: flagged || step.origin === "new" ? tag : null };
    });
    const wrap = h(
      "div",
      { class: "rev" },
      h("div", { class: "rev-head" }, h("b", {}, tn("errors", rev.rejected_issues.length)), t("fix.found", { i: index + 1 })),
      h("div", { class: "rev-cols" }, left, right),
    );
    if (rev.plan_level_issues.length) {
      put(wrap, h("p", { class: "findings-title tight" }, t("findings.title")), findingsList(rev.plan_level_issues));
    }
    if (rev.other_codes.length) put(wrap, h("p", { class: "revwarn" }, t("fix.other_codes", { codes: rev.other_codes.join(", ") })));
    if (rev.warnings.length) {
      put(
        wrap,
        h(
          "details",
          { class: "revwarn" },
          h("summary", {}, tn("fix.warnings", rev.warnings.length)),
          h("ul", {}, rev.warnings.map((w) => h("li", {}, h("span", { class: "mono" }, w.code), ": ", w.message))),
        ),
      );
    }
    parts.push(wrap);
  });
  return h("section", { class: "block", id: "b-fix" }, parts);
}

/* ---------------------------------------------------------------- dropped columns */

// pii: from the profile. leakage: only a claim of the agent, so it looks different.
const DROP_TAGS = {
  pii: "alert",
  leakage: "claim",
  identifier: "",
  constant: "",
  high_missing: "cand",
  unsupported_type: "",
  asked: "ink",
};

function droppedBlock(result) {
  const groups = result.dropped;
  const columns = groups.flatMap((g) => g.columns);
  const flagged = columns.filter((c) => c.kinds.includes("leakage") || c.kinds.includes("pii")).length;
  const parts = [
    h(
      "div",
      { class: "block-head" },
      h("h3", { id: "b-drop-h" }, t("drop.title")),
      columns.length ? h("span", { class: "sheet-meta" }, tn("columnsn", columns.length), flagged ? [" · ", h("span", { class: "flagcount" }, t("drop.flagged", { n: flagged }))] : null) : null,
    ),
  ];
  if (!columns.length) {
    parts.push(h("p", { class: "clean" }, t("drop.none")));
    return h("section", { class: "block", id: "b-drop" }, parts);
  }
  parts.push(h("p", { class: "lede" }, t("drop.lede")));
  for (const group of groups) {
    parts.push(
      h(
        "div",
        { class: "dgroup" },
        h("p", { class: "reason" }, h("span", { class: "mono reason-step" }, `${t("drop.step", { i: pad2(group.step_index) })} · `), group.reason),
        group.columns.map(dropRow),
      ),
    );
  }
  return h("section", { class: "block", id: "b-drop" }, parts);
}

function dropRow(column) {
  const pii = column.kinds.includes("pii");
  const claim = column.kinds.includes("leakage");
  const facts = [
    column.inferred_type,
    column.missing_ratio != null ? t("drop.missing", { pct: pct(column.missing_ratio) }) : null,
    column.n_unique != null ? t("drop.unique", { n: fmtInt(column.n_unique) }) : null,
  ].filter(Boolean);
  return h(
    "div",
    { class: `drow ${pii ? "flag" : ""} ${claim ? "claim" : ""}` },
    h(
      "div",
      {},
      h("div", { class: "dname" }, column.column),
      h(
        "div",
        { class: "dtags" },
        column.kinds.map((k) => h("span", { class: `tag ${DROP_TAGS[k]}`, title: lookup(`dt.${k}_title`) || null }, t(`dt.${k}`))),
      ),
    ),
    h(
      "div",
      {},
      h("div", { class: "dfacts" }, facts.join(" · ")),
      column.asked.map((qa) => h("div", { class: "dask" }, t("drop.asked_pre"), h("q", {}, inlineMarkdown(qa.question.question)), t("drop.asked_mid"), h("span", { class: "a" }, qa.answer), ".")),
    ),
  );
}

/* ---------------------------------------------------------------- baseline */

function baselineBlock(result) {
  const base = result.baseline;
  const meta = base && !base.skipped_reason ? t("base.meta", { folds: base.n_folds, rows: tn("trainrows", base.n_rows) }) : null;
  const parts = [blockHead(t("base.title"), meta, "b-base-h")];
  const done = () => h("section", { class: "block", id: "b-base" }, parts);
  if (result.baseline_error) {
    parts.push(h("p", { class: "smalldata" }, t("base.error", { error: result.baseline_error })));
    return done();
  }
  if (!base) {
    parts.push(h("p", { class: "clean" }, t("base.none")));
    return done();
  }
  if (base.skipped_reason) {
    parts.push(h("p", { class: "smalldata" }, t("base.skipped", { reason: base.skipped_reason })));
    return done();
  }
  parts.push(h("p", { class: "baseline-note" }, t("base.note", { dummy: t(base.task_type === "classification" ? "base.dummy_cls" : "base.dummy_reg") })));
  if (base.n_rows < 100) {
    parts.push(h("p", { class: "smalldata" }, h("b", {}, t("base.small")), t("base.small_body", { n: base.n_rows })));
  }
  parts.push(h("div", { class: "metrics" }, base.metrics.map((m) => metricChart(m, base.models))));
  if (base.models.some((m) => m.not_converged)) {
    parts.push(h("p", { class: "baseline-note" }, t("base.not_converged")));
  }
  const skipped = base.models.filter((m) => m.skipped_reason);
  if (skipped.length) {
    parts.push(h("ul", { class: "skipped" }, skipped.map((m) => h("li", {}, h("span", { class: "mono" }, m.name), t("model.skipped"), m.skipped_reason))));
  }
  return done();
}

/* The verdict against the dummy follows the benchmark summary: a difference smaller than the
   larger of the two fold standard deviations is noise, and counts as neither better nor worse. */
function versusDummy(metric, score, reference) {
  const delta = score.mean - reference.mean;
  const spread = Math.max(score.std, reference.std);
  const text = `${delta > 0 ? "+" : "−"}${fmtScore(Math.abs(delta))}`;
  if (Math.abs(delta) < spread || delta === 0) {
    return h("span", { class: "noise", title: t("vs.noise_title", { delta: text }) }, ` · ${t("vs.noise")}`);
  }
  const better = metric.higher_is_better ? delta > 0 : delta < 0;
  return better ? h("span", {}, ` · ${t("vs.better", { delta: text })}`) : h("span", { class: "worse" }, ` · ${t("vs.worse", { delta: text })}`);
}

function metricChart(metric, models) {
  const scored = models.filter((m) => m.scores[metric.name]);
  const reference = scored.find((m) => m.is_reference);
  const values = scored.flatMap((m) => [m.scores[metric.name].mean - m.scores[metric.name].std, m.scores[metric.name].mean + m.scores[metric.name].std]);
  const [domLo, domHi] = metric.domain;
  const lo = domLo != null ? domLo : Math.min(0, ...values);
  let hi = domHi != null ? domHi : Math.max(...values) * 1.08;
  if (hi <= lo) hi = lo + 1;
  const pos = (v) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * 100;
  const at = (v) => `${pos(v).toFixed(2)}%`;
  const span = (a, b) => `${(pos(b) - pos(a)).toFixed(2)}%`;
  const zero = Math.min(Math.max(0, lo), hi);

  const rows = scored.map((model) => {
    const score = model.scores[metric.name];
    const left = Math.min(zero, score.mean);
    const right = Math.max(zero, score.mean);
    const track = h(
      "div",
      { class: "track", "aria-hidden": "true" },
      h("div", { class: `bar ${model.is_reference ? "ref" : ""}`, style: `left:${at(left)};width:${span(left, right)}` }),
      h("div", { class: "whisker", style: `left:${at(score.mean - score.std)};width:${span(score.mean - score.std, score.mean + score.std)}` }),
      reference && !model.is_reference ? h("div", { class: "refline", style: `left:${at(reference.scores[metric.name].mean)}` }) : null,
    );
    const verdict = reference && !model.is_reference ? versusDummy(metric, score, reference.scores[metric.name]) : null;
    return [
      h("div", { class: "mname" }, model.name, model.is_reference ? h("small", {}, t("model.reference")) : null, model.not_converged ? h("small", {}, t("model.not_converged")) : null),
      h("div", {}, track, h("div", { class: "mval" }, h("b", {}, fmtScore(score.mean)), ` ± ${fmtScore(score.std)}`, verdict)),
    ];
  });

  return h(
    "figure",
    { class: "metric" },
    h("figcaption", {}, h("span", { class: "metric-name" }, metric.name), h("span", { class: "metric-dir" }, t(metric.higher_is_better ? "dir.up" : "dir.down"))),
    rows.map(([name, body]) => h("div", { class: "mrow" }, name, body)),
  );
}

/* ---------------------------------------------------------------- usage, files, trace */

function usageBlock(result, id) {
  const llm = result.llm;
  const usage = llm.usage;
  const none = t(state.config.fake ? "fact.demo" : "fact.not_reported");
  const rows = [
    [t("use.model"), llm.model || t("common.none")],
    [t("use.calls"), String(llm.calls.length)],
    [t("use.input"), usage ? tn("tokens", usage.input_tokens) : none],
  ];
  // The cache count is only there when the provider reports one.
  if (usage && usage.cached_input_tokens != null) rows.push([t("use.cache_read"), tn("tokens", usage.cached_input_tokens)]);
  rows.push([t("use.output"), usage ? tn("tokens", usage.output_tokens) : none]);
  const list = h("dl", { class: "deflist" }, rows.map(([label, value]) => [h("dt", {}, label), h("dd", {}, value)]));
  const files = h(
    "ul",
    { class: "files" },
    result.files.map((f) => h("li", {}, h("a", { href: `/api/sessions/${id}/files/${encodeURIComponent(f.name)}`, download: f.name }, f.name), h("span", { class: "size" }, fmtBytes(f.size)))),
  );
  return h(
    "section",
    { class: "block", id: "b-use" },
    blockHead(t("use.title"), null, "b-use-h"),
    h("div", { class: "pair" }, h("div", {}, h("p", { class: "label" }, t("use.planner")), list), h("div", {}, h("p", { class: "label" }, t("use.folder")), files)),
  );
}

function traceBlock(result, open) {
  const calls = result.llm.calls;
  if (!calls.length) return null;
  const rows = calls.map((c) =>
    h(
      "tr",
      {},
      h("td", {}, String(c.index)),
      h("td", {}, t(`trace.${c.purpose}`)),
      h("td", {}, c.response_kind === "invalid" ? h("span", { class: "tag alert-o" }, t("trace.kind_invalid")) : t(`trace.kind_${c.response_kind}`)),
      h("td", {}, h("div", { class: "codes" }, c.issue_codes.map((code) => h("span", { class: "tag alert-o" }, code)), c.warning_codes.map((code) => h("span", { class: "tag" }, code)))),
      h("td", {}, c.usage ? `${fmtInt(c.usage.input_tokens)} / ${fmtInt(c.usage.output_tokens)}` : noUsage()),
    ),
  );
  return h(
    "section",
    { class: "block", id: "b-trace" },
    blockHead(t("trace.title"), t("trace.meta"), "b-trace-h"),
    h(
      "details",
      { class: "disclose", open },
      h("summary", {}, tn("calls", calls.length)),
      h(
        "div",
        { class: "trace-wrap tablewrap tight-top" },
        h("table", { class: "trace" }, h("thead", {}, h("tr", {}, ["#", t("trace.purpose"), t("trace.answered"), t("trace.sent_back"), t("trace.tokens")].map((x) => h("th", { scope: "col" }, x)))), h("tbody", {}, rows)),
      ),
    ),
  );
}

/* ---------------------------------------------------------------- report.md */

function reportBlock(result) {
  if (!result.report_md) return null;
  return h(
    "section",
    { class: "block", id: "b-report" },
    blockHead(t("report.title"), "report.md", "b-report-h"),
    lang === "tr" && hasText("report.en_note") ? h("p", { class: "hint note-line" }, t("report.en_note")) : null,
    renderMarkdown(result.report_md),
  );
}

function inlineMarkdown(text) {
  const out = [];
  const pattern = /`([^`]+)`|\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*/g;
  let last = 0;
  let m;
  while ((m = pattern.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) out.push(h("code", {}, m[1]));
    else if (m[2] !== undefined) out.push(h("strong", {}, inlineMarkdown(m[2])));
    else out.push(h("em", {}, m[3]));
    last = pattern.lastIndex;
  }
  out.push(text.slice(last));
  return out;
}

// A table cell this long may wrap; everything shorter (names, codes, numbers) stays whole.
const LONG_CELL = 48;

function renderMarkdown(source) {
  const root = h("article", { class: "report" });
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const isSeparator = (s) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(s);
  const cells = (row) =>
    row
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split(/(?<!\\)\|/)
      .map((c) => c.replace(/\\\|/g, "|").trim());
  const startsBlock = (s, next) =>
    /^```/.test(s) ||
    /^#{1,6}\s/.test(s) ||
    /^>\s?/.test(s) ||
    /^\s*[-*]\s+/.test(s) ||
    /^\s*\d+\.\s+/.test(s) ||
    (s.trim().startsWith("|") && next !== undefined && isSeparator(next));
  const cell = (tag, text) => h(tag, { class: text.length > LONG_CELL ? "long" : "", scope: tag === "th" ? "col" : null }, inlineMarkdown(text));

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    let m;
    if (/^```/.test(line)) {
      const code = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
      i += 1;
      put(root, h("pre", {}, h("code", {}, code.join("\n"))));
    } else if ((m = /^(#{1,6})\s+(.*)$/.exec(line))) {
      put(root, h(`h${Math.min(m[1].length + 2, 6)}`, {}, inlineMarkdown(m[2])));
      i += 1;
    } else if (line.trim().startsWith("|") && i + 1 < lines.length && isSeparator(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const body = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) body.push(cells(lines[i++]));
      put(
        root,
        h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, h("tr", {}, head.map((c) => cell("th", c)))), h("tbody", {}, body.map((r) => h("tr", {}, r.map((c) => cell("td", c))))))),
      );
    } else if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ""));
      put(root, h("blockquote", {}, h("p", {}, inlineMarkdown(quote.join(" ").trim()))));
    } else if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*]\s+/, ""));
      put(root, h("ul", {}, items.map((x) => h("li", {}, inlineMarkdown(x)))));
    } else if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+\.\s+/, ""));
      put(root, h("ol", {}, items.map((x) => h("li", {}, inlineMarkdown(x)))));
    } else {
      const para = [line];
      i += 1;
      while (i < lines.length && lines[i].trim() && !startsBlock(lines[i], lines[i + 1])) para.push(lines[i++]);
      put(root, h("p", {}, inlineMarkdown(para.join(" "))));
    }
  }
  return root;
}

/* ================================================================ start */

async function init() {
  initTheme();
  initLang();
  const host = document.getElementById("sheets");
  for (const name of SHEET_ORDER) {
    sheets[name] = h("section", { class: "sheet", id: `sheet-${name}`, hidden: true });
    put(host, sheets[name]);
  }
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  try {
    state.config = await api("/api/config");
  } catch (error) {
    applyStatic();
    showBanner({ title: t("b.silent"), message: errorText(error) });
    return;
  }
  renderAll();
}

init();
