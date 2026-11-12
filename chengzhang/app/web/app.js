/* 成章 Chengzhang UI. Talks to Python through window.pywebview.api; falls back to a
   mock backend when opened in a plain browser (for design work). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid);
  return el;
};
const fmt = (n, d = 0) => Number(n || 0).toLocaleString("zh-CN", { minimumFractionDigits: d, maximumFractionDigits: d });

/* ======================================================================= painting */

const SVGNS = "http://www.w3.org/2000/svg";
function rng(seed) { return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646; }
function svg(tag, attrs, parent) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  parent && parent.append(el);
  return el;
}

const HORIZON = 262, SUN_X = 760, SUN_Y = 150;

const PALETTES = {
  day: {
    sky: ["#7E949D", "#A3B2B1", "#C4B8AA", "#A2B1AE"], sea: ["#93A8AA", "#5A7580"],
    skyCols: ["#B9C2BE", "#8EA2A9", "#C8B6A6", "#9C96AB", "#D3BFAE", "#A7B4B3"], skyWarm: "#D9AE93",
    mast: "#5F737C", chimney: "#6A7D86", smoke: "#8C9BA0", quay: "#71848C",
    seaCols: ["#A9B9B7", "#7F989E", "#5E7882", "#B7C1BA", "#8FA6AE", "#6D8790"],
    refl: ["#EE5A24", "#F28A55"], boat: "#22333C",
  },
  night: {
    sky: ["#0E1730", "#1B2746", "#2D3A5E", "#28355A"], sea: ["#24345A", "#0C1426"],
    skyCols: ["#1F2C4C", "#28375A", "#1A2644", "#313F66", "#243256", "#3A4670"], skyWarm: "#6E7FAE",
    mast: "#0C1424", chimney: "#101A2C", smoke: "#2C3A58", quay: "#141E33",
    seaCols: ["#2E4068", "#1C2A48", "#14203A", "#3A4C72", "#24345A", "#182640"],
    refl: ["#D8E3F6", "#AFC3E6"], boat: "#060B16",
  },
};

function paint(p = PALETTES.day) {
  const r = rng(1872);
  const pick = (a) => a[Math.floor(r() * a.length)];
  for (const id of ["#sky-strokes", "#far-harbor", "#sea-strokes", "#reflection", "#boats"]) $(id).replaceChildren();
  $$("#sky stop").forEach((st, i) => st.setAttribute("stop-color", p.sky[i]));
  $$("#sea stop").forEach((st, i) => st.setAttribute("stop-color", p.sea[i]));

  // sky: long, soft horizontal strokes
  const sky = $("#sky-strokes");
  const skyCols = p.skyCols;
  for (let i = 0; i < 46; i++) {
    const y = 8 + r() * 245, w = 90 + r() * 320;
    svg("rect", { x: r() * 1200 - 60, y, width: w, height: 4 + r() * 9, rx: 4,
      fill: y > 120 && Math.abs(r() * 1200 - SUN_X) < 300 ? p.skyWarm : pick(skyCols),
      opacity: (0.25 + r() * 0.45).toFixed(2) }, sky);
  }

  // distant harbor: masts, chimneys and smoke, dissolved in mist — kept to the
  // right of the headline and very faint, the way Monet lets them melt away
  const far = $("#far-harbor");
  for (let i = 0; i < 18; i++) {
    const x = 520 + r() * 640, ht = 24 + r() * 96;
    svg("rect", { x, y: HORIZON - ht, width: 1 + r() * 2, height: ht, fill: p.mast, opacity: (0.4 + r() * 0.6).toFixed(2) }, far);
    if (r() < 0.3) svg("rect", { x: x - 9, y: HORIZON - ht * 0.72, width: 19, height: 1.2, fill: p.mast }, far);
  }
  for (let i = 0; i < 4; i++) {
    const x = 900 + r() * 260, ht = 40 + r() * 50;
    svg("rect", { x, y: HORIZON - ht, width: 7 + r() * 6, height: ht, fill: p.chimney }, far);
    svg("ellipse", { cx: x - 30, cy: HORIZON - ht - 16, rx: 50 + r() * 30, ry: 11 + r() * 7, fill: p.smoke, opacity: 0.55 }, far);
  }
  svg("rect", { x: 480, y: HORIZON - 12, width: 720, height: 14, fill: p.quay, opacity: 0.6 }, far);

  // water: rows of broken strokes
  const sea = $("#sea-strokes");
  const seaCols = p.seaCols;
  for (let y = HORIZON + 4; y < 440; y += 4 + r() * 4) {
    let x = -40 + r() * 40;
    while (x < 1220) {
      const w = 18 + r() * (60 + (y - HORIZON) * 0.6);
      svg("rect", { x, y, width: w, height: 1.6 + r() * (2 + (y - HORIZON) / 60), rx: 1.5,
        fill: pick(seaCols), opacity: (0.35 + r() * 0.5).toFixed(2) }, sea);
      x += w + 6 + r() * 30;
    }
  }

  // the sun's reflection: the one orange thing on the water
  const refl = $("#reflection");
  for (let i = 0; i < 12; i++) {
    const y = HORIZON + 10 + i * 11 + r() * 4;
    const w = 66 - i * 3.4 + r() * 22;
    const x = SUN_X - w / 2 + (r() - 0.5) * 22;
    svg("rect", { x, y, width: w, height: 3 + r() * 2.5, rx: 2,
      fill: i % 3 === 1 ? p.refl[1] : p.refl[0], opacity: (0.95 - i * 0.055).toFixed(2) }, refl);
  }

  // boats: dark silhouettes with a standing figure, and their own dark reflection
  const boats = $("#boats");
  const boat = (cx, cy, s, op) => {
    const g = svg("g", { transform: `translate(${cx} ${cy}) scale(${s})`, opacity: op, fill: p.boat }, boats);
    svg("path", { d: "M-46 0 Q0 9 46 -2 L38 7 Q0 15 -36 8 Z" }, g);
    svg("rect", { x: -9, y: -26, width: 6, height: 26, rx: 2 }, g);
    svg("circle", { cx: -6, cy: -30, r: 4 }, g);
    svg("path", { d: "M-4 -18 L30 14", stroke: p.boat, "stroke-width": 2.2, fill: "none" }, g);
    for (let k = 0; k < 4; k++)
      svg("rect", { x: -40 + r() * 20, y: 12 + k * 6, width: 50 + r() * 30 - k * 10, height: 2.2, rx: 1, opacity: 0.35 - k * 0.07 }, g);
  };
  boat(640, 352, 1.35, 0.92);
  boat(905, 312, 0.85, 0.72);
  boat(1080, 292, 0.5, 0.42);
}

/* The painting is drawn once as SVG (above), then baked into three bitmaps:
     base = sky, mist, sea          refl = the sun's reflection (fades with the sun)
     top  = boats + canvas grain    (a nearer plane: it pans a little faster)
   SVG filters are far too slow to re-render every frame; bitmaps + CSS transforms
   run on the compositor. The sun and its glow are plain divs between the layers. */

const W = 1200, H = 440;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

async function rasterize(keep) {
  const sprite = $("#sprite");
  const defs = sprite.querySelector("defs").cloneNode(true);
  const scene = defs.querySelector("#scene");
  scene.removeAttribute("id");
  for (const el of [...scene.children]) if (!keep.includes(el.id)) el.remove();
  const doc = document.createElementNS(SVGNS, "svg");
  doc.setAttribute("xmlns", SVGNS);
  doc.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const scale = Math.min(2.4, (window.devicePixelRatio || 1) * 1.15);
  doc.setAttribute("width", W * scale);
  doc.setAttribute("height", H * scale);
  doc.append(defs);
  const use = document.createElementNS(SVGNS, "g");
  use.append(...scene.childNodes);
  doc.append(use);
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(doc)], { type: "image/svg+xml" }));
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = W * scale;
  c.height = H * scale;
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  URL.revokeObjectURL(url);
  return new Promise((r) => c.toBlob((bl) => r(URL.createObjectURL(bl)), "image/png"));
}

async function bakeSet(p) {
  paint(p);
  return Promise.all([
    rasterize(["sky-base", "sky-strokes", "far-harbor", "sea-base", "sea-strokes"]),
    rasterize(["reflection"]),
    rasterize(["boats", "grain-layer"]),
  ]);
}

async function bake() {
  // two paintings of the same harbour: dawn and night
  const night = await bakeSet(PALETTES.night);
  const day = await bakeSet(PALETTES.day);
  const ids = ["#layer-base", "#layer-refl", "#layer-top"];
  ids.forEach((q, i) => { $(q).src = day[i]; $(q + "-n").src = night[i]; });
  // decode() can stay pending for images that aren't painted yet (hidden window);
  // the load event is enough to reveal them
  await Promise.all([...ids, ...ids.map((q) => q + "-n")].map((q) => {
    const img = $(q);
    return img.complete ? null : new Promise((r) => { img.onload = img.onerror = r; });
  }));
  $("#vista").classList.add("baked");
}

/* ---- sun: GPU-only (transform/opacity), eased by CSS */

const SUN_TRAVEL = HORIZON - SUN_Y + 40;   // painting units below its risen spot
function setSun(p, { instant = false } = {}) {
  p = Math.max(0, Math.min(1, p));
  const v = $("#vista");
  v.classList.toggle("instant", instant || reduceMotion.matches);
  v.style.setProperty("--sun-drop", ((1 - p) * SUN_TRAVEL).toFixed(1));
  v.style.setProperty("--sun-p", p.toFixed(3));
  if (instant) requestAnimationFrame(() => v.classList.remove("instant"));
}

/* ---- camera: every page is a different view of the same harbor.
   x, y = top-left of the view in painting units, w = view width, h = banner px */

const SHOTS = {
  home:       { x: 0,   y: 0,   w: 1200, h: 372 },  // the whole painting
  history:    { x: 330, y: 248, w: 820,  h: 148 },  // down to the boats on the water
  dictionary: { x: 300, y: 0,   w: 900,  h: 148 },  // up into the brushed sky
  models:     { x: 230, y: 118, w: 970,  h: 148 },  // the sun on the horizon
  settings:   { x: 0,   y: 262, w: 860,  h: 148 },  // quiet water on the left
};
let shot = "home";

function frame(name = shot, { instant = false } = {}) {
  shot = name;
  const v = $("#vista");
  const s = SHOTS[name];
  const vw = v.clientWidth;
  if (!vw) return;
  // fill the banner: scale so the view's width covers the banner, and its height too
  const k = Math.max(vw / s.w, s.h / (H - s.y));
  const cover = Math.max(k, s.h / H, vw / W);
  v.classList.toggle("instant", instant || reduceMotion.matches);
  v.style.height = s.h + "px";
  v.style.setProperty("--k", cover.toFixed(4));
  v.style.setProperty("--tx", (-s.x * cover).toFixed(1) + "px");
  v.style.setProperty("--ty", (-s.y * cover).toFixed(1) + "px");
  // the near plane (boats) drifts a bit further than the far one: parallax
  const near = name === "home" ? 0 : (s.x / W - 0.3) * 60;
  v.style.setProperty("--near", (-near).toFixed(1) + "px");
  v.classList.toggle("tall", name === "home");
  if (instant) requestAnimationFrame(() => v.classList.remove("instant"));
}

/* ======================================================================= backend */

let api = null;
const STATE = { status: {}, settings: {}, stats: {}, hotkeys: {}, hotkey_label: "右 ⌥" };

// README screenshots: ?theme=light|dark picks the look and hides the permission banner (mock only)
const SHOT = new URLSearchParams(location.search);

function mockApi() {
  const now = Date.now() / 1000;
  const hist = [
    { id: "a1", ts: now - 300, app: "微信", text: "明天的会议改到下午三点，地点还是在三楼会议室。", raw: "嗯那个，我想跟你说一下，就是明天的会议改到下午三点了，然后地点还是在三楼会议室。", audio_s: 7.0, done_s: 0.44 },
    { id: "a2", ts: now - 1900, app: "Slack", text: "今天要做三件事：\n1. 修复登录的 bug\n2. 更新 README 文档\n3. 跟设计师对一下新的 UI 稿", raw: "今天要做三件事第一是修复登录的bug第二是更新README文档第三是跟设计师对一下新的UI搞", audio_s: 8.7, done_s: 0.85 },
    { id: "a3", ts: now - 4000, app: "Mail", mode: "英文", text: "We will release a new version next week. Feel free to try it and share your feedback.", raw: "我们下周会发布新版本，欢迎大家试用并提供反馈。", audio_s: 6.5, done_s: 0.61 },
    { id: "a4", ts: now - 90000, app: "Cursor", text: "这个 API 的 latency 有点高，可以先加一个 cache，再看看是不是 database 的 query 太慢了。", raw: "这个API的latency有点高我觉得可以先加一个cache然后再看看是不是day to day的query太慢了", audio_s: 8.4, done_s: 0.73 },
  ];
  let dict = ["成章", "database", "README", "UI稿", "fail to pass", "pass to pass"];
  const settings = { asr_model: "mlx-community/Qwen3-ASR-0.6B-8bit", llm_model: "mlx-community/Qwen3.5-4B-MLX-4bit",
    hotkey: { kind: "modifier", keycode: 61 }, hotkey_mode: "hold", translate_lang: "英文", language: "Chinese", post_roll_ms: 300, fast_path: true, auto_paste: true, appearance: SHOT.get("theme") || "auto" };
  const status = { state: "loading", stage: "vad", progress: 0.02, error: null };
  const stages = [["vad", 0.05], ["asr", 0.15], ["llm", 0.45], ["warmup", 0.85]];
  stages.forEach(([s, p], i) => setTimeout(() => { Object.assign(status, { stage: s, progress: p }); otlEmit({ kind: "status", payload: { ...status } }); }, 500 + i * 900));
  setTimeout(() => { Object.assign(status, { state: "ready", stage: null, progress: 1 }); otlEmit({ kind: "status", payload: { ...status } }); }, 4300);
  const stats = () => ({ today: { count: 3, chars: 1240, audio_min: 9.2, cpm: 134, avg_done_s: 0.42, saved_min: 15.6 }, week: {}, all: {} });
  const catalog = () => ({
    asr: [
      { id: "mlx-community/Qwen3-ASR-0.6B-8bit", name: "Qwen3-ASR 0.6B", blurb: "反应快，日常口述足够准。", facts: { "大小": "1.0 GB", "内存": "约 2 GB" }, size_gb: 1.01, recommended: true, downloaded: true },
      { id: "mlx-community/Qwen3-ASR-1.7B-8bit", name: "Qwen3-ASR 1.7B", blurb: "更准一些，口音重、环境吵、专业词多时更稳；稍慢。", facts: { "大小": "2.5 GB", "内存": "约 4 GB" }, size_gb: 2.47, downloaded: true },
    ],
    llm: [
      { id: "mlx-community/Qwen3.5-4B-MLX-4bit", name: "Qwen3.5 4B", blurb: "整理得最干净：去口头禅、理解改口、自动分条。", facts: { "大小": "3.1 GB", "内存": "约 3.5 GB" }, size_gb: 3.06, recommended: true, downloaded: true },
      { id: "mlx-community/Qwen3.5-2B-MLX-4bit", name: "Qwen3.5 2B", blurb: "更快、更省内存，口头禅偶尔删不干净。", facts: { "大小": "1.8 GB", "内存": "约 2 GB" }, size_gb: 1.75, downloaded: true },
      { id: "mlx-community/Qwen3.5-9B-MLX-4bit", name: "Qwen3.5 9B", blurb: "理解力更强，适合长段口述；生成更慢，内存要求更高。", facts: { "大小": "6.0 GB", "内存": "约 7 GB" }, size_gb: 5.98, downloaded: false, download: { status: "downloading", progress: 0.42 } },
      { id: "", name: "不整理", blurb: "只做语音识别，原样输出，会保留口头禅和改口。", facts: {}, size_gb: 0, downloaded: true },
    ].map((m) => ({ ...m, selected: m.id === settings.llm_model })),
  });
  const ok = (v) => Promise.resolve(v);
  return {
    state: () => ok({ status: { ...status }, settings: { ...settings }, hotkey_label: settings.hotkey.kind === "modifier" ? ({ 61: "右 ⌥", 54: "右 ⌘", 60: "右 ⇧", 62: "右 ⌃", 63: "fn" }[settings.hotkey.keycode] || "?") : "⌃ Space",
      hotkey_presets: [[61, "右 ⌥"], [54, "右 ⌘"], [60, "右 ⇧"], [62, "右 ⌃"], [63, "fn"]].map(([k, l]) => ({ spec: { kind: "modifier", keycode: k }, label: l })), accessibility: SHOT.has("theme"), stats: stats() }),
    models: () => { const c = catalog(); c.asr.forEach((m) => (m.selected = m.id === settings.asr_model)); c.llm.forEach((m) => (m.selected = m.id === settings.llm_model)); return ok(c); },
    history: (limit = 50, offset = 0, q = "") => ok(hist.filter((e) => !q || e.text.includes(q) || e.raw.includes(q)).slice(offset, offset + limit)),
    delete_entry: (id) => { hist.splice(hist.findIndex((e) => e.id === id), 1); return ok(stats()); },
    clear_history: () => { hist.length = 0; return ok(stats()); },
    copy: () => ok(true),
    dictionary: () => ok([...dict]),
    save_dictionary: (w) => ok((dict = [...new Set(w)])),
    set_setting: (k, v) => { settings[k] = v; return api.state(); },
    select_model: (kind, id) => { settings[kind === "asr" ? "asr_model" : "llm_model"] = id; return api.state(); },
    download: () => ok(true),
    request_accessibility: () => ok(false),
    set_hotkey: (spec) => { settings.hotkey = spec; return api.state(); },
    pause_hotkey: () => ok(true),
    mics: () => ok({ mics: [{ name: "MacBook Pro麦克风", builtin: true }, { name: "AirPods", default: true, bluetooth: true }], selected: settings.mic || "auto", resolved: "MacBook Pro麦克风" }),
  };
}

/* ======================================================================= status */

const STAGE_TEXT = { vad: "正在加载声音检测", asr: "正在加载语音识别", llm: "正在加载文字整理", warmup: "正在预热", download: "第一次使用，正在下载模型" };
const WAKE_TEXT = { asr: "正在唤醒语音识别", llm: "正在唤醒文字整理", warmup: "正在唤醒文字整理" };

function renderStatus(st) {
  STATE.status = st;
  const el = $("#hero-status");
  el.className = "status";
  const pct = Math.round((st.progress || 0) * 100);
  if (st.state === "ready") {
    el.classList.add("ready");
    el.textContent = STATE.settings.llm_model === "" ? "就绪，不整理文字" : "就绪";
    setSun(1);
  } else if (st.state === "error") {
    el.classList.add("error");
    el.textContent = st.error || "模型加载失败";
    setSun(0.05);
  } else if (st.state === "asleep") {
    // models freed: the sun sits low over the water until the next utterance
    el.textContent = st.stage === "all" ? "已释放模型，按快捷键唤醒" : "已释放文字整理，说话时自动唤醒";
    setSun(st.stage === "all" ? 0.12 : 0.45);
  } else if (st.state === "waking") {
    el.textContent = WAKE_TEXT[st.stage] || "正在唤醒";
    setSun(0.7);
  } else if (st.state === "loading" || st.state === "downloading") {
    el.innerHTML = "";
    el.textContent = `${STAGE_TEXT[st.stage] || "正在加载"}  ${pct}%`;
    setSun(st.state === "downloading" ? 0.04 + 0.3 * st.progress : 0.25 + 0.75 * st.progress);
  } else {
    el.textContent = "正在加载";
    setSun(0);
  }
  if ($("#page-models:not([hidden])")) loadModels();
  if ($("#page-settings:not([hidden])")) renderMemory();
}

function renderLabels() {
  $$(".hk").forEach((k) => (k.textContent = STATE.hotkey_label));
  $$(".tl").forEach((k) => (k.textContent = STATE.settings.translate_lang || "英文"));
}

function renderLedger(stats) {
  STATE.stats = stats;
  const el = $("#ledger");
  const t = stats.today;
  el.textContent = t && t.count ? `今天 ${fmt(t.chars)} 字` : "";
}

/* ======================================================================= entries */

function dayLabel(ts) {
  const d = new Date(ts * 1000), today = new Date();
  const d0 = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  if (d.getTime() >= d0) return "今天";
  if (d.getTime() >= d0 - 86400000) return "昨天";
  return `${d.getMonth() + 1} 月 ${d.getDate()} 日`;
}
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });

function entryEl(e, { onDelete } = {}) {
  const raw = h("p", { class: "entry-raw", hidden: true }, e.raw || "");
  const toggle = h("button", { type: "button", onclick: () => { raw.hidden = !raw.hidden; toggle.textContent = raw.hidden ? "原话" : "收起原话"; } }, "原话");
  return h("li", { class: "entry" },
    h("p", { class: "entry-text" }, e.text),
    raw,
    h("div", { class: "entry-meta" },
      h("span", { class: "num" }, clock(e.ts)),
      e.app ? h("span", {}, e.app) : null,
      e.mode ? h("span", { class: "tag" }, `译成${e.mode}`) : null,
      h("span", { class: "entry-actions" },
        h("button", { type: "button", onclick: () => api.copy(e.text).then(() => toast("已复制")) }, "复制"),
        e.raw && e.raw !== e.text ? toggle : null,
        onDelete ? h("button", { type: "button", onclick: onDelete }, "删除") : null)));
}

async function loadRecent() {
  const items = await api.history(3, 0, "");
  const ol = $("#recent-list");
  ol.innerHTML = "";
  if (!items.length) {
    $(".recent").hidden = true;
    return;
  }
  $(".recent").hidden = false;
  items.forEach((e) => ol.append(entryEl(e)));
  $("#recent-more").hidden = items.length < 3;
}

let historyQ = "";
async function loadHistory() {
  const items = await api.history(500, 0, historyQ);
  const box = $("#history-list");
  box.innerHTML = "";
  $("#history-clear").hidden = !items.length || !!historyQ;
  if (!items.length) {
    box.append(h("p", { class: "empty-state" }, historyQ ? "没有找到" : "还没有记录"));
    return;
  }
  let day = null, ol = null;
  for (const e of items) {
    const d = dayLabel(e.ts);
    if (d !== day) {
      day = d;
      box.append(h("h2", { class: "day" }, d));
      ol = h("ol", { class: "entries" });
      box.append(ol);
    }
    const li = entryEl(e, { onDelete: async () => { renderLedger(await api.delete_entry(e.id)); li.remove(); toast("已删除"); loadRecent(); } });
    ol.append(li);
  }
}

/* ======================================================================= dictionary */

let words = [];
async function loadDictionary() {
  words = await api.dictionary();
  renderWords();
}
function renderWords(fresh) {
  const ul = $("#words");
  ul.innerHTML = "";
  if (!words.length) ul.append(h("li", { class: "empty-state", style: "border:0;background:none;padding:0" }, "还没有词"));
  words.forEach((w) => ul.append(h("li", { class: w === fresh ? "fresh" : "" }, w,
    h("button", { type: "button", "aria-label": `删除 ${w}`, onclick: () => saveWords(words.filter((x) => x !== w)) }, "×"))));
}
async function saveWords(next, fresh) {
  words = await api.save_dictionary(next);
  renderWords(fresh);
  toast("词典已更新");
}

/* ======================================================================= models */

let modelPoll = null;
function modelState(m) {
  const st = STATE.status;
  if (m.download && m.download.status === "downloading") {
    const p = Math.round(m.download.progress * 100);
    return h("span", { class: "state" }, `下载中 ${p}%`, h("span", { class: "bar" }, h("i", { style: `width:${p}%` })));
  }
  if (m.download && m.download.status === "error") return h("span", { class: "state" }, "下载失败，点击重试");
  if (m.selected) {
    if (st.state === "ready") return h("span", { class: "state in-use" }, "使用中");
    if (st.state === "error") return h("span", { class: "state" }, "加载失败");
    return h("span", { class: "state in-use" }, `加载中 ${Math.round((st.progress || 0) * 100)}%`);
  }
  if (!m.downloaded) return h("span", { class: "state" }, m.size_gb ? `需下载 ${fmt(m.size_gb, 1)} GB` : "需下载");
  return h("span", { class: "state" }, m.id === "" ? "" : "已下载");
}
function optionEl(kind, m) {
  return h("button", { type: "button", class: "option", role: "radio", "aria-checked": m.selected ? "true" : "false",
      onclick: async () => {
        if (m.selected) return;
        await api.select_model(kind, m.id);
        toast(m.downloaded ? `正在切换到 ${m.name}` : `开始下载 ${m.name}`);
        loadModels();
      } },
    h("span", { class: "dot" }),
    h("span", {},
      h("span", { class: "name" }, m.name, m.recommended ? h("span", { class: "rec" }, "推荐") : null),
      h("p", { class: "blurb" }, m.blurb),
      Object.keys(m.facts || {}).length ? h("span", { class: "facts" }, Object.entries(m.facts).map(([k, v]) => h("span", {}, k, h("b", {}, v)))) : null),
    modelState(m));
}
async function loadModels() {
  const c = await api.models();
  const a = $("#models-asr"), l = $("#models-llm");
  a.innerHTML = ""; l.innerHTML = "";
  c.asr.forEach((m) => a.append(optionEl("asr", m)));
  c.llm.forEach((m) => l.append(optionEl("llm", m)));
  const busy = [...c.asr, ...c.llm].some((m) => m.download && m.download.status === "downloading") || !["ready", "error"].includes(STATE.status.state);
  clearTimeout(modelPoll);
  if (busy && $("#page-models:not([hidden])")) modelPoll = setTimeout(loadModels, 1000);
}

/* ======================================================================= settings */

function seg(el, options, value, onPick) {
  el.innerHTML = "";
  for (const [v, label] of options)
    el.append(h("button", { type: "button", role: "radio", "data-v": v, "aria-checked": v === value ? "true" : "false",
      onclick: () => { $$("button", el).forEach((b) => b.setAttribute("aria-checked", b.dataset.v === v ? "true" : "false")); onPick(v); } }, label));
}
/* ---- hotkey recorder: the page listens to the keyboard while the global hotkey is paused */

const MAC_KEYCODES = {
  KeyA: 0, KeyS: 1, KeyD: 2, KeyF: 3, KeyH: 4, KeyG: 5, KeyZ: 6, KeyX: 7, KeyC: 8, KeyV: 9, KeyB: 11, KeyQ: 12,
  KeyW: 13, KeyE: 14, KeyR: 15, KeyY: 16, KeyT: 17, Digit1: 18, Digit2: 19, Digit3: 20, Digit4: 21, Digit6: 22,
  Digit5: 23, Equal: 24, Digit9: 25, Digit7: 26, Minus: 27, Digit8: 28, Digit0: 29, BracketRight: 30, KeyO: 31,
  KeyU: 32, BracketLeft: 33, KeyI: 34, KeyP: 35, Enter: 36, KeyL: 37, KeyJ: 38, Quote: 39, KeyK: 40, Semicolon: 41,
  Backslash: 42, Comma: 43, Slash: 44, KeyN: 45, KeyM: 46, Period: 47, Tab: 48, Space: 49, Backquote: 50,
  Backspace: 51, F1: 122, F2: 120, F3: 99, F4: 118, F5: 96, F6: 97, F7: 98, F8: 100, F9: 101, F10: 109, F11: 103,
  F12: 111, F13: 105, F14: 107, F15: 113, F16: 106, F17: 64, F18: 79, F19: 80, F20: 90,
  ArrowLeft: 123, ArrowRight: 124, ArrowDown: 125, ArrowUp: 126,
};
const MODIFIER_CODES = { AltRight: 61, AltLeft: 58, MetaRight: 54, MetaLeft: 55, ShiftRight: 60, ShiftLeft: 56, ControlRight: 62, ControlLeft: 59 };
const KEY_LABEL = (code) => code.replace(/^Key|^Digit/, "").replace("Arrow", "").replace("Space", "Space");

function sameHotkey(a, b) {
  if (!a || !b) return false;
  return a.kind === b.kind && a.keycode === b.keycode && [...(a.mods || [])].sort().join() === [...(b.mods || [])].sort().join();
}

function renderHotkey() {
  $("#hotkey-now").textContent = STATE.hotkey_label;
  const box = $("#hotkey-presets");
  box.innerHTML = "";
  for (const p of STATE.hotkey_presets || []) {
    box.append(h("button", { type: "button", "aria-pressed": sameHotkey(p.spec, STATE.settings.hotkey) ? "true" : "false",
      onclick: () => saveHotkey(p.spec) }, p.label));
  }
}

async function saveHotkey(spec) {
  const st = await api.set_hotkey(spec);
  Object.assign(STATE, { settings: st.settings, hotkey_label: st.hotkey_label });
  renderLabels();
  renderHotkey();
  toast(`快捷键改为 ${st.hotkey_label}`);
}

let recording = null;
function stopRecording() {
  if (!recording) return;
  window.removeEventListener("keydown", recording.down, true);
  window.removeEventListener("keyup", recording.up, true);
  $(".hotkey").classList.remove("recording");
  $("#hotkey-record").textContent = "更换";
  $("#hotkey-now").textContent = STATE.hotkey_label;
  api.pause_hotkey(false);
  recording = null;
}
function startRecording() {
  if (recording) return stopRecording();
  api.pause_hotkey(true);
  $(".hotkey").classList.add("recording");
  $("#hotkey-record").textContent = "取消";
  $("#hotkey-now").textContent = "请按键…";
  let lone = null; // a modifier pressed on its own, waiting to see whether a key follows
  const mods = (e) => ["control", "option", "shift", "command"].filter((m, i) => [e.ctrlKey, e.altKey, e.shiftKey, e.metaKey][i]);
  recording = {
    down: (e) => {
      e.preventDefault(); e.stopPropagation();
      if (e.code === "Escape" && !e.ctrlKey && !e.altKey && !e.metaKey) return stopRecording();
      if (MODIFIER_CODES[e.code] !== undefined) { lone = e.code; $("#hotkey-now").textContent = "…"; return; }
      lone = null;
      const keycode = MAC_KEYCODES[e.code];
      if (keycode === undefined) return toast("这个键不支持，换一个试试");
      const m = mods(e);
      const plain = /^(Key|Digit)|^(Space|Enter|Tab|Backspace|Comma|Period|Slash|Semicolon|Quote|Minus|Equal|Backquote|Backslash|Bracket)/.test(e.code);
      if (plain && m.length === 0) return toast("单独的字母、数字或空格会影响打字，请加上修饰键");
      stopRecording();
      saveHotkey({ kind: "key", keycode, mods: m, name: KEY_LABEL(e.code) });
    },
    up: (e) => {
      e.preventDefault(); e.stopPropagation();
      if (lone && e.code === lone) { stopRecording(); saveHotkey({ kind: "modifier", keycode: MODIFIER_CODES[lone] }); }
    },
  };
  window.addEventListener("keydown", recording.down, true);
  window.addEventListener("keyup", recording.up, true);
}

async function setSetting(k, v, msg = "已保存") {
  const st = await api.set_setting(k, v);
  Object.assign(STATE, { settings: st.settings, hotkey_label: st.hotkey_label });
  renderLabels();
  toast(msg);
}
function renderSettings() {
  const s = STATE.settings;
  seg($("#set-hotkey-mode"), [["hold", "按住说话"], ["toggle", "按一下开始，再按一下结束"]], s.hotkey_mode || "hold",
    (v) => setSetting("hotkey_mode", v, v === "toggle" ? "按一下开始，再按一下结束" : "按住说话"));
  renderHotkey();
  seg($("#set-appearance"), [["auto", "跟随系统"], ["light", "白天"], ["dark", "夜间"]], s.appearance || "auto",
    async (v) => { await setSetting("appearance", v, v === "dark" ? "日落" : v === "light" ? "日出" : "跟随系统"); applyTheme(); });
  seg($("#set-language"), [["Chinese", "中文"], ["auto", "自动"]], s.language, (v) => setSetting("language", v));
  $("#set-translate").value = s.translate_lang;
  $("#set-postroll").value = s.post_roll_ms;
  $("#postroll-out").textContent = s.post_roll_ms ? `${s.post_roll_ms} 毫秒` : "关闭";
  $("#set-fast").checked = s.fast_path;
  $("#set-keepmic").checked = !!s.keep_mic_open;
  $("#set-saveaudio").checked = !!s.save_audio;
  $("#set-unload-after").value = String(s.unload_after_min || 0);
  seg($("#set-unload-what"), [["llm", "只释放文字整理"], ["all", "全部"]], s.unload_what || "llm",
    (v) => setSetting("unload_what", v).then(renderMemory));
  renderMemory();
  renderMics();
  $("#set-paste").checked = s.auto_paste;
}

function renderMemory() {
  const s = STATE.settings, st = STATE.status;
  const what = s.unload_what || "llm";
  $("#unload-what-note").textContent = what === "llm"
    ? "语音识别留在内存里，说话立刻开始；文字整理在你说话时唤醒，约 3 秒"
    : "最省内存；唤醒要 3–7 秒，这期间不能说话";
  $("#set-unload-what").closest(".setting").hidden = !+s.unload_after_min && st.state !== "asleep";
  const btn = $("#mem-action"), now = $("#mem-now");
  if (st.state === "asleep") {
    now.textContent = what === "all" || st.stage === "all" ? "模型已释放" : "文字整理已释放";
    btn.textContent = "唤醒"; btn.disabled = false; btn.dataset.act = "wake";
  } else if (st.state === "ready") {
    now.textContent = st.memory_gb ? `模型占用约 ${fmt(st.memory_gb, 1)} GB` : "模型已加载";
    btn.textContent = "立即释放"; btn.disabled = false; btn.dataset.act = "unload";
  } else {
    now.textContent = st.state === "waking" ? "正在唤醒" : "正在加载";
    btn.textContent = "立即释放"; btn.disabled = true; btn.dataset.act = "";
  }
}

async function renderMics() {
  if (!api.mics) return;
  const { mics, selected, resolved } = await api.mics();
  const sel = $("#set-mic");
  sel.innerHTML = "";
  sel.append(h("option", { value: "auto" }, resolved ? `自动（${resolved}）` : "自动（系统默认）"));
  for (const m of mics) sel.append(h("option", { value: m.name }, m.name + (m.default ? "（系统默认）" : "")));
  sel.value = mics.some((m) => m.name === selected) ? selected : "auto";
  const chosen = sel.value === "auto" ? mics.find((m) => m.name === resolved) : mics.find((m) => m.name === sel.value);
  $("#mic-note").classList.toggle("warn", !!(chosen && chosen.bluetooth));
}

/* ======================================================================= shell */

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1600);
}

const PAGES = { home: loadRecent, history: loadHistory, dictionary: loadDictionary, models: loadModels, settings: renderSettings };
let routed = false;
function route() {
  const page = (location.hash || "#home").slice(1);
  const name = PAGES[page] ? page : "home";
  const next = $(`#page-${name}`);
  const prev = $(".page:not([hidden])");
  $$("nav a").forEach((a) => (a.dataset.page === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  moveMark(name, !routed);
  frame(name, { instant: !routed });
  if (prev !== next) {
    if (prev) prev.hidden = true;
    next.hidden = false;
    if (routed && !reduceMotion.matches) {
      next.classList.remove("enter");
      void next.offsetWidth; // restart the animation
      next.classList.add("enter");
    }
  }
  $("main").scrollTop = 0;
  routed = true;
  if (api) PAGES[name]();
}

function moveMark(name, instant) {
  const mark = $("#nav-mark"), link = $(`nav a[data-page="${name}"]`);
  mark.classList.toggle("instant", instant);
  mark.style.transform = `translateY(${link.offsetTop + link.offsetHeight / 2 - 1}px)`;
}

// events pushed from Python
window.otlEmit = ({ kind, payload }) => {
  if (kind === "accessibility") $("#ax-banner").hidden = !!payload;
  if (kind === "status") renderStatus(payload);
  else if (kind === "stats") renderLedger(payload);
  else if (kind === "models" && $("#page-models:not([hidden])")) loadModels();
  else if (kind === "entry") {
    loadRecent();
    if ($("#page-history:not([hidden])")) loadHistory();
  }
};

let baking = Promise.resolve();
async function start(backend) {
  api = backend;
  const st = await api.state();
  Object.assign(STATE, st);
  applyTheme({ instant: true });
  renderLabels();
  renderStatus(st.status);
  renderLedger(st.stats);
  $("#ax-banner").hidden = !!st.accessibility;
  route();
  baking.then(
    () => api.client_log && api.client_log(`painting baked (${$("#layer-base").naturalWidth}px)`),
    (e) => api.client_log && api.client_log(`painting bake failed: ${e}`));
}

/* ---- day / night */

const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
let themed = false;
function wantDark() {
  const pref = (STATE.settings && STATE.settings.appearance) || "auto";
  return pref === "dark" || (pref === "auto" && darkQuery.matches);
}
function applyTheme({ instant = false } = {}) {
  const dark = wantDark();
  const root = document.documentElement, v = $("#vista");
  if ((root.dataset.theme === "dark") === dark && themed) return;
  const animate = themed && !instant && !reduceMotion.matches;
  if (animate) {
    root.classList.add("theming");
    clearTimeout(applyTheme.t);
    applyTheme.t = setTimeout(() => root.classList.remove("theming"), 3400);
  } else {
    v.classList.add("instant");
    requestAnimationFrame(() => requestAnimationFrame(() => v.classList.remove("instant")));
  }
  root.dataset.theme = dark ? "dark" : "light";
  v.classList.toggle("night", dark);
  themed = true;
}
darkQuery.addEventListener("change", () => applyTheme());

function wire() {
  paint();
  setSun(0, { instant: true });
  frame('home', { instant: true });
  baking = bake();
  let lastW = 0;
  new ResizeObserver((entries) => {
    const w = entries[0].contentRect.width;
    if (Math.abs(w - lastW) < 0.5) return; // height changes come from our own transition
    lastW = w;
    frame(shot, { instant: true });
  }).observe($("#vista"));
  window.addEventListener("hashchange", route);
  $("#ax-open").addEventListener("click", async () => {
    const ok = await api.request_accessibility();
    if (ok) $("#ax-banner").hidden = true;
    else toast("在列表里打开你启动成章所用的终端的开关，然后重新启动");
  });
  let qTimer;
  $("#history-q").addEventListener("input", (e) => { clearTimeout(qTimer); qTimer = setTimeout(() => { historyQ = e.target.value.trim(); loadHistory(); }, 200); });
  $("#history-clear").addEventListener("click", async () => {
    if (!confirm("清空全部记录？这一步不能撤销。")) return;
    renderLedger(await api.clear_history());
    loadHistory(); loadRecent();
    toast("记录已清空");
  });
  $("#add-word").addEventListener("submit", (e) => {
    e.preventDefault();
    const w = $("#word-input").value.trim();
    if (!w) return;
    $("#word-input").value = "";
    if (words.includes(w)) return toast("词典里已经有这个词了");
    saveWords([...words, w], w);
  });
  $("#custom-model").addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = $("#custom-id").value.trim();
    if (!/^[\w.-]+\/[\w.-]+$/.test(id)) return toast("模型 ID 的格式是 作者/名字");
    await api.select_model($("#custom-kind").value, id);
    $("#custom-id").value = "";
    toast(`正在准备 ${id}`);
    loadModels();
  });
  $("#hotkey-record").addEventListener("click", startRecording);
  window.addEventListener("hashchange", stopRecording);
  $("#set-translate").addEventListener("change", (e) => setSetting("translate_lang", e.target.value));
  $("#set-postroll").addEventListener("input", (e) => { const v = +e.target.value; $("#postroll-out").textContent = v ? `${v} 毫秒` : "关闭"; });
  $("#set-postroll").addEventListener("change", (e) => setSetting("post_roll_ms", +e.target.value));
  $("#set-fast").addEventListener("change", (e) => setSetting("fast_path", e.target.checked));
  $("#set-paste").addEventListener("change", (e) => setSetting("auto_paste", e.target.checked));
  $("#set-mic").addEventListener("change", async (e) => { await setSetting("mic", e.target.value, "麦克风已切换"); renderMics(); });
  $("#set-keepmic").addEventListener("change", (e) => setSetting("keep_mic_open", e.target.checked));
  $("#set-saveaudio").addEventListener("change", (e) => setSetting("save_audio", e.target.checked));
  $("#set-unload-after").addEventListener("change", async (e) => {
    await setSetting("unload_after_min", +e.target.value, +e.target.value ? `闲置 ${e.target.selectedOptions[0].text}后释放` : "不自动释放");
    renderMemory();
  });
  $("#mem-action").addEventListener("click", async (e) => {
    const act = e.currentTarget.dataset.act;
    if (act === "unload") { await api.unload_models(STATE.settings.unload_what || "llm"); toast("正在释放"); }
    else if (act === "wake") { await api.wake_models(); toast("正在唤醒"); }
  });

  if (window.pywebview && window.pywebview.api) start(window.pywebview.api);
  else {
    let started = false;
    window.addEventListener("pywebviewready", () => { if (!started) { started = true; start(window.pywebview.api); } });
    // plain browser: no Python backend shows up — use the mock
    setTimeout(() => { if (!started && !window.pywebview) { started = true; start(mockApi()); } }, 400);
  }
}

wire();
