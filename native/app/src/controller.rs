//! App controller (port of controller.py): owns the engine, microphone, hotkey, HUD
//! and storage. Hotkey and engine events are handled on one worker thread; AppKit
//! calls are hopped onto the main thread; JS pushes never block.
use crate::mic::{self, Mic};
use crate::models::{self, Downloads};
use crate::store::Store;
use cz_core::backend::Backend;
use cz_core::engine::{Command, Config, Engine, Event, LiveSettings, Outcome};
use cz_core::learn;
use cz_platform::{Hotkey, HotkeyEvent, HotkeyHandle};
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

pub const REPO_URL: &str = "https://github.com/izumedonabe/chengzhang";

pub enum Work {
    HotkeyDown(bool),
    HotkeyUp,
    HotkeyCancel,
    Learn,
    Engine(u64, Event),
    ForceFlush(u64),
    Slow(u64),
}

/// After this long in "整理中" the HUD says it's slow and that esc gives up on it.
const SLOW_AFTER: Duration = Duration::from_secs(6);

pub struct Ctl {
    pub app: AppHandle,
    pub store: Store,
    pub downloads: Downloads,
    mic: Mic,
    engine: Mutex<Option<Engine>>,
    engine_tx: Arc<Mutex<Option<Sender<Command>>>>,
    engine_gen: AtomicU64,
    status: Mutex<Map<String, Value>>,
    hotkey: Mutex<Option<HotkeyHandle>>,
    recording: (Mutex<bool>, Condvar),
    session: Mutex<Value>,
    loaded: Mutex<(bool, bool)>, // asr, llm in memory
    last_used: Mutex<Instant>,
    hud_gen: AtomicU64,
    capture_gen: AtomicU64,
    /// The dictation that was released and is still being transcribed / polished.
    pending: Mutex<Option<u64>>,
    dictation_gen: AtomicU64,
    /// Cancels the engine hasn't acknowledged yet: until then, what it emits belongs to
    /// a dictation the user gave up on and must not be pasted.
    discarding: AtomicU64,
    polish: AtomicBool,
    learning: AtomicBool,
    work: Sender<Work>,
}

fn s_str(s: &Map<String, Value>, k: &str) -> String {
    s.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string()
}
fn s_bool(s: &Map<String, Value>, k: &str) -> bool {
    s.get(k).and_then(|v| v.as_bool()).unwrap_or(false)
}
fn s_num(s: &Map<String, Value>, k: &str) -> f64 {
    s.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

/// macOS: MLX (Metal). Elsewhere: llama.cpp (Vulkan / CPU).
fn backend() -> Arc<dyn Backend> {
    #[cfg(target_os = "macos")]
    return Arc::new(cz_mlx::backend::MlxBackend);
    #[cfg(not(target_os = "macos"))]
    return Arc::new(cz_llama::LlamaBackend);
}

/// The correction hotkey: a key with at least one modifier; anything else = the default.
pub fn learn_hotkey_from(v: Option<&Value>) -> Hotkey {
    let ok = |h: &Hotkey| h.kind == "key" && !h.mods.is_empty();
    v.and_then(|v| serde_json::from_value::<Hotkey>(v.clone()).ok())
        .map(|h| h.sanitized())
        .filter(ok)
        .unwrap_or_else(|| serde_json::from_value(crate::store::defaults()["learn_hotkey"].clone()).unwrap())
}

pub fn hotkey_from(v: Option<&Value>) -> Hotkey {
    match v {
        Some(Value::String(s)) => Hotkey::from_legacy(s),
        Some(v @ Value::Object(_)) => serde_json::from_value::<Hotkey>(v.clone()).map(|h| h.sanitized()).unwrap_or_default(),
        _ => Hotkey::default(),
    }
}

impl Ctl {
    pub fn new(app: AppHandle) -> Arc<Ctl> {
        let store = Store::new();
        let (work_tx, work_rx) = channel::<Work>();
        let engine_tx: Arc<Mutex<Option<Sender<Command>>>> = Arc::default();
        let etx = engine_tx.clone();
        let mic = Mic::new(Arc::new(move |x| {
            if let Some(tx) = etx.lock().unwrap().as_ref() {
                let _ = tx.send(Command::Audio(x));
            }
        }));
        let dl_app = app.clone();
        let downloads = Downloads::new(store.models_dir(), Arc::new(move |id: &str| {
            if let Some(c) = dl_app.try_state::<Arc<Ctl>>() {
                c.on_download_change(id);
            }
        }));
        downloads.set_source(store.settings().get("download_source").and_then(|v| v.as_str()).unwrap_or("auto"));
        let status = json!({"state": "starting", "stage": null, "progress": 0.0, "error": null});
        let ctl = Arc::new(Ctl {
            app,
            store,
            downloads,
            mic,
            engine: Mutex::new(None),
            engine_tx,
            engine_gen: AtomicU64::new(0),
            status: Mutex::new(status.as_object().unwrap().clone()),
            hotkey: Mutex::new(None),
            recording: (Mutex::new(false), Condvar::new()),
            session: Mutex::new(json!({})),
            loaded: Mutex::new((false, false)),
            last_used: Mutex::new(Instant::now()),
            hud_gen: AtomicU64::new(0),
            capture_gen: AtomicU64::new(0),
            pending: Mutex::new(None),
            dictation_gen: AtomicU64::new(0),
            discarding: AtomicU64::new(0),
            polish: AtomicBool::new(false),
            learning: AtomicBool::new(false),
            work: work_tx,
        });
        let c = ctl.clone();
        std::thread::Builder::new().name("cz-work".into()).spawn(move || {
            for w in work_rx {
                match w {
                    Work::HotkeyDown(shift) => c.hotkey_down(shift),
                    Work::HotkeyUp => c.hotkey_up(),
                    Work::HotkeyCancel => c.hotkey_cancel(),
                    Work::Learn => c.learn(),
                    Work::Engine(gen, e) => c.on_engine(gen, e),
                    Work::ForceFlush(gen) => c.force_flush(gen),
                    Work::Slow(gen) => c.slow(gen),
                }
            }
        }).unwrap();
        ctl
    }

    // ------------------------------------------------------------------ push to UI

    fn eval(&self, label: &str, js: String) {
        if let Some(w) = self.app.get_webview_window(label) {
            let _ = w.eval(&js);
        }
    }

    pub fn ui(&self, kind: &str, payload: Value) {
        let p = json!({"kind": kind, "payload": payload});
        self.eval("main", format!("window.otlEmit && window.otlEmit({p})"));
    }

    fn hud(&self, state: &str, payload: Value) {
        if state != "hidden" {
            self.hud_gen.fetch_add(1, Ordering::SeqCst);
        }
        let p = json!({"state": state, "payload": payload});
        self.eval("hud", format!("window.hudState && window.hudState({p})"));
        let hidden = state == "hidden";
        let app = self.app.clone();
        let _ = self.app.run_on_main_thread(move || {
            if let Some(w) = app.get_webview_window("hud") {
                if let Some(h) = crate::native_handle(&w) {
                    if hidden { cz_platform::hide_window(h) } else { cz_platform::show_without_focus(h) }
                }
            }
        });
    }

    /// Hide the HUD after `delay` — unless something newer was shown since.
    fn hide_hud_later(self: &Arc<Self>, secs: f32) {
        let gen = self.hud_gen.load(Ordering::SeqCst);
        let c = self.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f32(secs));
            if c.hud_gen.load(Ordering::SeqCst) == gen {
                c.hud("hidden", Value::Null);
            }
        });
    }

    fn set_status(&self, kw: Value) {
        let snapshot = {
            let mut st = self.status.lock().unwrap();
            if let Value::Object(o) = kw {
                for (k, v) in o {
                    st.insert(k, v);
                }
            }
            st.clone()
        };
        eprintln!("[cz] status {} {} {:.2} {}", snapshot["state"], snapshot["stage"], snapshot["progress"].as_f64().unwrap_or(0.0), snapshot["error"]);
        self.ui("status", Value::Object(snapshot));
    }

    fn state_is(&self, s: &str) -> bool {
        self.status.lock().unwrap().get("state").and_then(|v| v.as_str()) == Some(s)
    }

    // ------------------------------------------------------------------ lifecycle

    pub fn boot(self: &Arc<Self>) {
        self.install_hotkey();
        self.place_hud();
        let c = self.clone();
        std::thread::Builder::new().name("cz-level".into()).spawn(move || c.level_ticker()).unwrap();
        let c = self.clone();
        std::thread::Builder::new().name("cz-idle".into()).spawn(move || c.idle_watch()).unwrap();
        let c = self.clone();
        std::thread::Builder::new().name("cz-ax".into()).spawn(move || c.accessibility_watch()).unwrap();
        self.configure_mic();
        self.load_engine();
    }

    fn place_hud(&self) {
        let (x, top, w, h) = cz_platform::main_screen_frame();
        if let Some(win) = self.app.get_webview_window("hud") {
            let (hw, hh) = (320.0, 96.0);
            let _ = win.set_position(tauri::LogicalPosition::new(x + (w - hw) / 2.0, top + h - hh - 36.0));
        }
    }

    /// (Re)create the global hotkey tap on the main thread.
    pub fn install_hotkey(self: &Arc<Self>) {
        let c = self.clone();
        let _ = self.app.run_on_main_thread(move || {
            *c.hotkey.lock().unwrap() = None; // drop the old tap first
            let s = c.store.settings();
            let spec = hotkey_from(s.get("hotkey"));
            let learn = learn_hotkey_from(s.get("learn_hotkey"));
            let label = format!("{} / 纠错 {}", spec.label(), learn.label());
            let work = c.work.clone();
            // the tap runs on the main thread: only enqueue there, never block
            let cb = Arc::new(move |e: HotkeyEvent| {
                let _ = work.send(match e {
                    HotkeyEvent::Down { shift } => Work::HotkeyDown(shift),
                    HotkeyEvent::Up => Work::HotkeyUp,
                    HotkeyEvent::Cancel => Work::HotkeyCancel,
                    HotkeyEvent::Learn => Work::Learn,
                });
            });
            let how = match HotkeyHandle::install(spec, Some(learn), cb) {
                Some((h, how)) => {
                    *c.hotkey.lock().unwrap() = Some(h);
                    how
                }
                None => "none",
            };
            eprintln!("[cz] hotkey {label} via {how}; accessibility={}", cz_platform::accessibility_trusted(false));
        });
    }

    fn make_config(&self) -> Option<Config> {
        let s = self.store.settings();
        let dir = self.store.models_dir();
        let asr = models::asr_spec(&dir, &s_str(&s, "asr_model")).or_else(|| models::asr_spec(&dir, models::DEFAULT_ASR))?;
        let llm_id = s_str(&s, "llm_model");
        let llm = if llm_id.is_empty() { None } else { models::llm_spec(&dir, &llm_id) };
        let mut cfg = Config::new(asr, llm);
        let lang = s_str(&s, "language");
        cfg.language = if lang == "auto" { None } else { Some(lang) };
        cfg.dictionary = self.store.dictionary();
        let tl = s_str(&s, "translate_lang");
        cfg.translate_lang = (!tl.is_empty()).then_some(tl);
        cfg.fast_path = s_bool(&s, "fast_path");
        cfg.post_roll_max_ms = s_num(&s, "post_roll_ms") as u32;
        Some(cfg)
    }

    pub fn load_engine(self: &Arc<Self>) {
        let s = self.store.settings();
        let dir = self.store.models_dir();
        let needed: Vec<String> = [s_str(&s, "asr_model"), s_str(&s, "llm_model")].into_iter().filter(|m| !m.is_empty()).collect();
        let missing: Vec<&String> = needed.iter().filter(|m| !models::is_downloaded(&dir, m)).collect();
        // stop the old engine (dropping it joins its thread)
        let gen = self.engine_gen.fetch_add(1, Ordering::SeqCst) + 1;
        *self.engine_tx.lock().unwrap() = None;
        self.discarding.store(0, Ordering::SeqCst);
        if self.pending.lock().unwrap().take().is_some() {
            self.hud("hidden", Value::Null);
        }
        let old = self.engine.lock().unwrap().take();
        drop(old);
        if !missing.is_empty() {
            self.set_status(json!({"state": "downloading", "stage": "download", "progress": 0.0, "error": null}));
            for m in missing {
                self.downloads.start(m);
            }
            return;
        }
        let Some(cfg) = self.make_config() else {
            self.set_status(json!({"state": "error", "error": "没有可用的语音识别模型"}));
            return;
        };
        self.polish.store(cfg.llm.is_some(), Ordering::SeqCst);
        self.set_status(json!({"state": "loading", "stage": "asr", "progress": 0.02, "error": null}));
        let work = self.work.clone();
        let eng = Engine::spawn(cfg, backend(), Box::new(move |e| {
            let _ = work.send(Work::Engine(gen, e));
        }));
        *self.engine_tx.lock().unwrap() = Some(eng.sender());
        *self.engine.lock().unwrap() = Some(eng);
    }

    fn send(&self, cmd: Command) {
        if let Some(tx) = self.engine_tx.lock().unwrap().as_ref() {
            let _ = tx.send(cmd);
        }
    }

    fn has_engine(&self) -> bool {
        self.engine_tx.lock().unwrap().is_some()
    }

    pub fn on_download_change(self: &Arc<Self>, _id: &str) {
        self.ui("models", self.catalog());
        if !self.state_is("downloading") {
            return;
        }
        let s = self.store.settings();
        let needed: Vec<String> = [s_str(&s, "asr_model"), s_str(&s, "llm_model")].into_iter().filter(|m| !m.is_empty()).collect();
        let errors: Vec<String> = needed.iter()
            .filter_map(|m| self.downloads.state(m))
            .filter(|st| st["status"] == "error")
            .map(|st| st["error"].as_str().unwrap_or("").to_string())
            .collect();
        if let Some(e) = errors.first() {
            self.set_status(json!({"state": "error", "error": format!("模型下载失败：{e}")}));
        } else if needed.iter().all(|m| models::is_downloaded(&self.store.models_dir(), m)) {
            self.load_engine();
        }
    }

    fn download_progress(&self) -> f64 {
        let s = self.store.settings();
        let needed: Vec<String> = [s_str(&s, "asr_model"), s_str(&s, "llm_model")].into_iter().filter(|m| !m.is_empty()).collect();
        if needed.is_empty() {
            return 1.0;
        }
        needed.iter().map(|m| self.downloads.progress(m)).sum::<f64>() / needed.len() as f64
    }

    // ------------------------------------------------------------------ engine events

    fn on_engine(self: &Arc<Self>, gen: u64, e: Event) {
        if gen != self.engine_gen.load(Ordering::SeqCst) {
            return; // events from an engine we already replaced
        }
        if self.discarding.load(Ordering::SeqCst) > 0 {
            match e {
                Event::Canceled => {
                    let _ = self.discarding.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
                    return;
                }
                // what's left of the dictation the user gave up on
                Event::Segment(_) | Event::CaptureDone { .. } | Event::Raw(_) | Event::Text(_) | Event::Done(_) => return,
                Event::Error(msg) => {
                    eprintln!("[cz] engine error (cancelled dictation): {msg}");
                    if !self.state_is("ready") {
                        self.set_status(json!({"state": "error", "error": msg}));
                    }
                    return;
                }
                _ => {}
            }
        }
        match e {
            Event::Loading { stage, progress, reload } => {
                let state = if reload { "waking" } else { "loading" };
                self.set_status(json!({"state": state, "stage": stage, "progress": progress}));
            }
            Event::Ready { memory_gb, .. } => {
                *self.loaded.lock().unwrap() = (true, self.polish.load(Ordering::SeqCst));
                *self.last_used.lock().unwrap() = Instant::now();
                self.set_status(json!({"state": "ready", "stage": null, "progress": 1.0,
                                       "memory_gb": (memory_gb * 10.0).round() / 10.0}));
            }
            Event::Unloaded { all, memory_gb } => {
                *self.loaded.lock().unwrap() = (!all, false);
                let what = if all { "all" } else { "llm" };
                self.set_status(json!({"state": "asleep", "stage": what, "progress": 0.0,
                                       "memory_gb": (memory_gb * 10.0).round() / 10.0}));
            }
            Event::Error(msg) => {
                eprintln!("[cz] engine error: {msg}");
                // a dictation in flight always gets an answer, whatever state the app is in
                // (e.g. the LLM failing to come back while waking up)
                let in_flight = self.pending.lock().unwrap().take().is_some() || self.recording();
                if in_flight {
                    self.set_recording(false);
                    self.end_dictation();
                    let short: String = msg.chars().take(60).collect();
                    self.hud("error", json!(format!("出错了：{short}")));
                    self.hide_hud_later(2.6);
                }
                if !self.state_is("ready") {
                    self.set_status(json!({"state": "error", "error": msg}));
                }
                self.stop_capture();
            }
            Event::CaptureDone { .. } => {
                self.stop_capture();
                self.send(Command::Flush);
            }
            Event::Segment(info) => self.hud("partial", json!(info.text)),
            Event::Done(o) => {
                self.pending.lock().unwrap().take();
                self.end_dictation();
                self.finish(*o)
            }
            _ => {}
        }
    }

    fn finish(self: &Arc<Self>, r: Outcome) {
        let text = r.text.clone();
        if text.is_empty() {
            self.hud("empty", Value::Null);
            self.hide_hud_later(1.2);
            return;
        }
        let s = self.store.settings();
        let auto = s_bool(&s, "auto_paste");
        let mut pasted = false;
        let t = text.clone();
        if auto {
            pasted = cz_platform::accessibility_trusted(false);
            let _ = self.app.run_on_main_thread(move || {
                cz_platform::paste_text(&t); // copies only when not trusted
            });
            if !pasted {
                self.ui("accessibility", json!(false));
            }
        } else {
            let _ = self.app.run_on_main_thread(move || cz_platform::copy_text(&t));
        }
        let round3 = |x: f32| ((x * 1000.0).round() / 1000.0) as f64;
        let session = self.session.lock().unwrap().clone();
        let entry = json!({
            "text": text,
            "raw": r.raw,
            "mode": r.mode,
            "app": session.get("app").cloned().unwrap_or(json!("")),
            "audio_s": ((r.audio_s * 100.0).round() / 100.0) as f64,
            "done_s": round3(r.release_to_done_s),
            "first_s": r.release_to_first_char_s.map(round3),
            "polished": r.polished,
            "fallback": r.fallback_to_raw,
            "mic": self.mic.name.lock().unwrap().clone(),
            "engine": "native",
        });
        let entry = self.store.add(entry.as_object().unwrap().clone());
        if s_bool(&s, "save_audio") && !r.audio.is_empty() {
            self.store.save_audio(entry["id"].as_str().unwrap_or("x"), &r.audio);
        }
        if pasted {
            self.hud("done", json!({"text": text, "done_s": r.release_to_done_s}));
            self.hide_hud_later(0.9);
        } else {
            self.hud("copied", json!({"text": text, "auto": auto}));
            self.hide_hud_later(2.6);
        }
        eprintln!("[cz] done: audio {:.1}s, release→first char {}, →done {:.0} ms",
                  r.audio_s, r.release_to_first_char_s.map(|x| format!("{:.0} ms", x * 1e3)).unwrap_or("-".into()),
                  r.release_to_done_s * 1e3);
        self.ui("entry", Value::Object(entry));
        self.ui("stats", self.store.stats());
    }

    // ------------------------------------------------------------------ recording

    fn recording(&self) -> bool {
        *self.recording.0.lock().unwrap()
    }

    fn set_recording(&self, on: bool) {
        *self.recording.0.lock().unwrap() = on;
        self.recording.1.notify_all();
    }

    /// Sleeps on a condvar between dictations: no wakeups at all when idle.
    fn level_ticker(&self) {
        loop {
            {
                let g = self.recording.0.lock().unwrap();
                let _g = self.recording.1.wait_while(g, |r| !*r).unwrap();
            }
            std::thread::sleep(Duration::from_millis(40));
            if self.recording() {
                self.eval("hud", format!("window.hudLevel && window.hudLevel({:.3})", self.mic.level()));
            }
        }
    }

    fn hotkey_down(self: &Arc<Self>, shift: bool) {
        let s = self.store.settings();
        let toggle = s_str(&s, "hotkey_mode") == "toggle";
        if toggle && self.recording() {
            self.stop_recording();
            return;
        }
        let state = self.status.lock().unwrap().get("state").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if (state == "asleep" || state == "waking") && self.has_engine() {
            // wake up. If speech recognition is still loaded, start listening right away —
            // the LLM comes back while the user talks.
            let (asr_loaded, _) = *self.loaded.lock().unwrap();
            if state == "asleep" {
                self.send(Command::Reload);
                self.set_status(json!({"state": "waking", "stage": if asr_loaded { "llm" } else { "asr" }, "progress": 0.0}));
            }
            if !asr_loaded {
                self.hud("waking", Value::Object(self.status.lock().unwrap().clone()));
                self.hide_hud_later(1.6);
                return;
            }
        } else if state != "ready" || !self.has_engine() {
            self.hud("loading", Value::Object(self.status.lock().unwrap().clone()));
            self.hide_hud_later(1.2);
            return;
        }
        *self.last_used.lock().unwrap() = Instant::now();
        if self.recording() || self.mic.capturing() {
            return;
        }
        if self.pending.lock().unwrap().is_some() {
            // the last one isn't out yet. Starting another now would mix the two up.
            let mut p = self.session.lock().unwrap().clone();
            p["note"] = json!("上一段还没好，esc 可放弃");
            self.hud("thinking", p);
            return;
        }
        let tl = s_str(&s, "translate_lang");
        let mode = if shift && !tl.is_empty() && self.polish.load(Ordering::SeqCst) { Some(tl) } else { None };
        *self.session.lock().unwrap() = json!({"app": cz_platform::frontmost_app(), "mode": mode});
        // Some("") = plain polish (not the config default)
        self.send(Command::Start(Some(mode.clone().unwrap_or_default())));
        if let Err(e) = self.mic.start() {
            self.send(Command::Cancel);
            self.hud("error", json!(format!("无法打开麦克风：{e}")));
            self.hide_hud_later(2.0);
            return;
        }
        self.set_recording(true);
        if let Some(h) = self.hotkey.lock().unwrap().as_ref() {
            h.set_active(true);
        }
        let hint = if toggle { "再按一次结束，esc 放弃" } else { "松开结束，esc 放弃" };
        self.hud("listening", json!({"mode": mode, "hint": hint}));
    }

    fn hotkey_up(self: &Arc<Self>) {
        if s_str(&self.store.settings(), "hotkey_mode") == "toggle" {
            return; // in toggle mode releasing the key means nothing
        }
        self.stop_recording();
    }

    fn stop_recording(self: &Arc<Self>) {
        if !self.recording() {
            return;
        }
        *self.last_used.lock().unwrap() = Instant::now();
        // esc stays live until the text is out: it gives up on a slow one
        self.set_recording(false);
        self.send(Command::End(Instant::now()));
        let d = self.dictation_gen.fetch_add(1, Ordering::SeqCst) + 1;
        *self.pending.lock().unwrap() = Some(d);
        self.hud("thinking", self.session.lock().unwrap().clone());
        let work = self.work.clone();
        std::thread::spawn(move || {
            std::thread::sleep(SLOW_AFTER);
            let _ = work.send(Work::Slow(d));
        });
        // safety net: if the engine never reports capture_done, stop anyway
        let gen = self.capture_gen.fetch_add(1, Ordering::SeqCst) + 1;
        let work = self.work.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1500));
            let _ = work.send(Work::ForceFlush(gen));
        });
    }

    fn hotkey_cancel(self: &Arc<Self>) {
        self.end_dictation();
        self.pending.lock().unwrap().take();
        self.set_recording(false);
        self.stop_capture();
        if self.has_engine() {
            // the engine may still be busy with it: drop whatever it says until it has
            // seen the cancel
            self.discarding.fetch_add(1, Ordering::SeqCst);
            self.send(Command::Cancel);
        }
        self.hud("hidden", Value::Null);
    }

    /// esc goes back to the app once a dictation is over.
    fn end_dictation(&self) {
        if let Some(h) = self.hotkey.lock().unwrap().as_ref() {
            h.set_active(false);
        }
    }

    /// Still "整理中" after a while: usually the Mac is busy (another model on the GPU, a
    /// big build, memory swapping). Say so, rather than look stuck.
    fn slow(&self, gen: u64) {
        if *self.pending.lock().unwrap() == Some(gen) {
            let mut p = self.session.lock().unwrap().clone();
            p["note"] = json!("比平时慢，esc 可放弃");
            self.hud("thinking", p);
        }
    }

    fn force_flush(&self, gen: u64) {
        if gen == self.capture_gen.load(Ordering::SeqCst) && self.mic.capturing() {
            self.stop_capture();
            self.send(Command::Flush);
        }
    }

    fn stop_capture(&self) {
        self.capture_gen.fetch_add(1, Ordering::SeqCst);
        self.mic.stop();
    }

    /// Permission is usually granted while we're running. Notice it, rebuild the tap
    /// (a full tap can only be created once trusted) and drop the banner. Polls only
    /// often while untrusted.
    fn accessibility_watch(self: &Arc<Self>) {
        let mut trusted = cz_platform::accessibility_trusted(false);
        loop {
            std::thread::sleep(Duration::from_millis(if trusted { 30_000 } else { 1_500 }));
            let now = cz_platform::accessibility_trusted(false);
            if now != trusted {
                trusted = now;
                eprintln!("[cz] accessibility -> {now}");
                self.install_hotkey();
                self.ui("accessibility", json!(now));
            }
        }
    }

    // ------------------------------------------------------------------ learning from corrections

    /// The correction hotkey: the user fixed the last dictation in the text box. Read it
    /// back, diff it against what we pasted, and add the corrected names and terms to the
    /// dictionary (see `learn`: nothing is ever replaced automatically).
    fn learn(self: &Arc<Self>) {
        if self.recording() || self.mic.capturing() {
            // the combo shares a modifier with the dictation key (⌥): that press wasn't dictation
            self.hotkey_cancel();
        }
        if self.learning.swap(true, Ordering::SeqCst) {
            return;
        }
        let c = self.clone();
        std::thread::spawn(move || {
            c.learn_now();
            c.learning.store(false, Ordering::SeqCst);
        });
    }

    fn learn_note(self: &Arc<Self>, title: &str, detail: String, error: bool) {
        self.hud(if error { "error" } else { "learned" }, json!({"title": title, "detail": detail}));
        self.hide_hud_later(if error { 2.6 } else { 2.2 });
    }

    fn learn_now(self: &Arc<Self>) {
        let Some((edited, how)) = cz_platform::grab_text() else {
            return self.learn_note("没读到文字", "选中改好的那段，再按一次".into(), true);
        };
        // which dictation is this? the most similar of the recent ones (newest wins a tie)
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64()).unwrap_or(0.0) - 2.0 * 86400.0;
        let mut best: Option<(f32, Map<String, Value>)> = None;
        for e in self.store.history(10, 0, "") {
            if e.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0) < since {
                continue;
            }
            let text = e.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let score = learn::similarity(text, &edited);
            if best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, e));
            }
        }
        let Some((score, entry)) = best.filter(|(s, _)| *s >= 0.5) else {
            eprintln!("[cz] learn: no matching dictation for {} chars from {how}", edited.chars().count());
            return self.learn_note("没找到对应的输入", "只能纠正最近几次口述的结果".into(), true);
        };
        let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let text = entry.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let Some(l) = learn::learn(text, &edited) else {
            return self.learn_note("文字太长", "选中改过的那段，再按一次".into(), true);
        };
        eprintln!("[cz] learn: {how}, match {score:.2}, pairs {:?}, terms {:?}", l.pairs, l.terms);
        if !l.changed {
            return self.learn_note("没有发现改动", "先在输入框里改好，再按快捷键".into(), false);
        }
        if let Some(e) = self.store.update_entry(&id, "corrected", json!(l.corrected)) {
            self.ui("entry", Value::Object(e));
        }
        let added = self.store.add_words(&l.terms);
        if !added.is_empty() {
            self.send(Command::SetDictionary(self.store.dictionary()));
            self.ui("lexicon", Value::Null);
            return self.learn_note("已加入词典", added.join("、"), false);
        }
        if !l.terms.is_empty() {
            return self.learn_note("词典里已经有了", l.terms.join("、"), false);
        }
        self.learn_note("已记下改正", "没有发现要加进词典的人名或术语".into(), false);
    }

    // ------------------------------------------------------------------ memory

    fn idle_watch(self: &Arc<Self>) {
        loop {
            std::thread::sleep(Duration::from_secs(15));
            let s = self.store.settings();
            let mins = s_num(&s, "unload_after_min");
            if mins <= 0.0 || !self.has_engine() || self.recording() || self.mic.capturing() || !self.state_is("ready") {
                continue;
            }
            if self.last_used.lock().unwrap().elapsed().as_secs_f64() >= mins * 60.0 {
                self.unload(&s_str(&s, "unload_what"));
            }
        }
    }

    pub fn unload(&self, what: &str) {
        if self.has_engine() && !self.recording() && self.state_is("ready") {
            self.send(Command::Unload { all: what == "all" });
        }
    }

    pub fn wake(&self) {
        if self.has_engine() && self.state_is("asleep") {
            let (asr, _) = *self.loaded.lock().unwrap();
            self.send(Command::Reload);
            self.set_status(json!({"state": "waking", "stage": if asr { "llm" } else { "asr" }, "progress": 0.0}));
        }
    }

    // ------------------------------------------------------------------ queries / commands

    pub fn configure_mic(&self) {
        let s = self.store.settings();
        self.mic.configure(mic::resolve_mic(&s_str(&s, "mic")), s_bool(&s, "keep_mic_open"));
    }

    pub fn catalog(&self) -> Value {
        let s = self.store.settings();
        self.downloads.catalog(&s_str(&s, "asr_model"), &s_str(&s, "llm_model"))
    }

    fn apply_live_settings(&self) {
        let s = self.store.settings();
        let lang = s_str(&s, "language");
        self.send(Command::Live(LiveSettings {
            language: if lang == "auto" { None } else { Some(lang) },
            fast_path: s_bool(&s, "fast_path"),
            post_roll_max_ms: s_num(&s, "post_roll_ms") as u32,
        }));
    }

    // ------------------------------------------------------------------ the web UI's API

    pub fn state(&self) -> Value {
        let s = self.store.settings();
        let mut status = self.status.lock().unwrap().clone();
        if status.get("state").and_then(|v| v.as_str()) == Some("downloading") {
            status.insert("progress".into(), json!(self.download_progress()));
        }
        let hk = hotkey_from(s.get("hotkey"));
        let learn_hk = learn_hotkey_from(s.get("learn_hotkey"));
        let presets: Vec<Value> = Hotkey::presets().into_iter().map(|p| json!({"label": p.label(), "spec": p})).collect();
        json!({
            "status": status,
            "settings": s,
            "hotkey_label": hk.label(),
            "learn_hotkey_label": learn_hk.label(),
            "hotkey_presets": presets,
            "accessibility": cz_platform::accessibility_trusted(false),
            "stats": self.store.stats(),
        })
    }

    pub fn api(self: &Arc<Self>, method: &str, args: &[Value]) -> Result<Value, String> {
        let a = |i: usize| args.get(i).cloned().unwrap_or(Value::Null);
        let a_str = |i: usize| a(i).as_str().unwrap_or("").to_string();
        Ok(match method {
            "state" => self.state(),
            "models" => self.catalog(),
            "unload_models" => {
                let w = a_str(0);
                self.unload(if w.is_empty() { "llm" } else { &w });
                json!(true)
            }
            "wake_models" => {
                self.wake();
                json!(true)
            }
            "set_hotkey" => {
                let spec = hotkey_from(Some(&a(0)));
                self.store.update_setting("hotkey", serde_json::to_value(spec).unwrap());
                self.install_hotkey();
                self.state()
            }
            "set_learn_hotkey" => {
                let spec = learn_hotkey_from(Some(&a(0)));
                if spec == hotkey_from(self.store.settings().get("hotkey")) {
                    return Err("和说话快捷键相同".into());
                }
                self.store.update_setting("learn_hotkey", serde_json::to_value(spec).unwrap());
                self.install_hotkey();
                self.state()
            }
            "pause_hotkey" => {
                if let Some(h) = self.hotkey.lock().unwrap().as_ref() {
                    h.set_paused(a(0).as_bool().unwrap_or(false));
                }
                json!(true)
            }
            "mics" => {
                let s = self.store.settings();
                let mics: Vec<Value> = mic::list_mics().into_iter().map(|m| json!({
                    "name": m.name, "default": m.default, "builtin": m.builtin, "bluetooth": m.bluetooth})).collect();
                json!({"mics": mics, "selected": s["mic"], "resolved": mic::resolve_mic(&s_str(&s, "mic"))})
            }
            "history" => {
                let limit = a(0).as_u64().unwrap_or(50) as usize;
                let offset = a(1).as_u64().unwrap_or(0) as usize;
                json!(self.store.history(limit, offset, &a_str(2)))
            }
            "delete_entry" => {
                self.store.delete(&a_str(0));
                self.store.stats()
            }
            "clear_history" => {
                self.store.clear();
                self.store.stats()
            }
            "copy" => {
                let t = a_str(0);
                let _ = self.app.run_on_main_thread(move || cz_platform::copy_text(&t));
                json!(true)
            }
            "dictionary" => json!(self.store.dictionary()),
            "save_dictionary" => {
                let words: Vec<String> = a(0).as_array().map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
                let words = self.store.set_dictionary(&words);
                self.send(Command::SetDictionary(words.clone()));
                json!(words)
            }
            "set_setting" => self.set_setting(&a_str(0), a(1)),
            "select_model" => {
                let key = if a_str(0) == "asr" { "asr_model" } else { "llm_model" };
                self.set_setting(key, json!(a_str(1).trim()))
            }
            "download" => {
                self.downloads.start(a_str(0).trim());
                json!(true)
            }
            "request_accessibility" => {
                let trusted = cz_platform::accessibility_trusted(true);
                if !trusted {
                    let _ = self.app.run_on_main_thread(|| cz_platform::open_privacy_pane("Accessibility"));
                }
                json!(trusted)
            }
            // the About card behind the seal on the painting
            "about" => {
                let s = self.store.settings();
                let name = |id: &str| models::find(id).map(|m| m.name.to_string())
                    .unwrap_or_else(|| id.rsplit('/').next().unwrap_or("").to_string());
                let llm = name(&s_str(&s, "llm_model"));
                json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "repo": REPO_URL,
                    "asr": name(&s_str(&s, "asr_model")),
                    "llm": if llm.is_empty() { "不整理".to_string() } else { llm },
                })
            }
            "open_url" => {
                // only our own repository, like the Python app
                let url = a_str(0);
                if !url.starts_with(REPO_URL) {
                    return Ok(json!(false));
                }
                let _ = self.app.run_on_main_thread(move || {
                    cz_platform::open_url(&url);
                });
                json!(true)
            }
            "client_log" => {
                eprintln!("[ui] {}", a_str(0));
                json!(true)
            }
            _ => return Err(format!("unknown api method {method}")),
        })
    }

    fn set_setting(self: &Arc<Self>, key: &str, value: Value) -> Value {
        let before = self.store.settings();
        self.store.update_setting(key, value.clone());
        match key {
            "hotkey" => self.install_hotkey(),
            "translate_lang" => {
                let v = value.as_str().unwrap_or("").to_string();
                self.send(Command::SetTranslateLang((!v.is_empty()).then_some(v)));
            }
            "asr_model" | "llm_model" if before.get(key) != Some(&value) => self.load_engine(),
            "mic" | "keep_mic_open" => self.configure_mic(),
            "download_source" => self.downloads.set_source(value.as_str().unwrap_or("auto")),
            _ => self.apply_live_settings(),
        }
        self.state()
    }
}
