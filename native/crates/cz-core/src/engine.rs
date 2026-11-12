//! The dictation pipeline (port of engine.py): VAD segmentation -> Qwen3-ASR per
//! segment while the user talks -> each segment's text is prefilled into the LLM
//! right away -> after release only the last segment + template tail remain.
//!
//! All inference runs on one engine thread. Callers talk to it only through
//! `Engine::send()` and the event callback.

use crate::backend::{Asr, AsrSpec, Backend, Llm, LlmSpec};
use crate::prompts;
use crate::text::{join_segments, needs_llm, suspicious};
use crate::vad::{Segmenter, VadConfig, SR};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Config {
    pub asr: AsrSpec,
    pub llm: Option<LlmSpec>, // None = no polishing
    pub language: Option<String>,
    pub dictionary: Vec<String>,
    pub translate: Option<String>,      // default mode (CLI --translate)
    pub translate_lang: Option<String>, // extra mode available per utterance (Shift+hotkey)
    pub fast_path: bool,
    pub vad: VadConfig,
    pub max_tokens: usize,
    /// People often release the key before the last syllable ends: after release keep
    /// capturing until VAD sees this much silence, at most post_roll_max_ms.
    pub post_roll_max_ms: u32,
    pub post_roll_silence_ms: u32,
    pub incremental_prefill: bool,
}

impl Config {
    pub fn new(asr: AsrSpec, llm: Option<LlmSpec>) -> Self {
        Config {
            asr,
            llm,
            language: Some("Chinese".into()),
            dictionary: vec![],
            translate: None,
            translate_lang: None,
            fast_path: true,
            vad: VadConfig::default(),
            max_tokens: 1024,
            post_roll_max_ms: 300,
            post_roll_silence_ms: 100,
            incremental_prefill: true,
        }
    }
}

/// Settings that can change without reloading models.
#[derive(Clone, Debug)]
pub struct LiveSettings {
    pub language: Option<String>,
    pub fast_path: bool,
    pub post_roll_max_ms: u32,
}

pub enum Command {
    Start(Option<String>), // mode: None = config default, Some("") = plain polish, Some(lang) = translate
    Audio(Vec<f32>),
    End(Instant),
    Flush,
    Cancel,
    Unload { all: bool },
    Reload,
    SetDictionary(Vec<String>),
    SetTranslateLang(Option<String>),
    Live(LiveSettings),
    Quit,
}

#[derive(Clone, Debug, Default)]
pub struct SegmentInfo {
    pub start_s: f32,
    pub end_s: f32,
    pub asr_s: f32,
    pub llm_prefill_s: f32,
    pub text: String,
}

#[derive(Clone, Debug, Default)]
pub struct LlmStats {
    pub ttft_s: Option<f32>,
    pub total_s: f32,
    pub prompt_tokens_after_release: usize,
    pub gen_tokens: usize,
    pub gen_tps: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub raw: String,
    pub text: String,
    pub fallback_to_raw: bool,
    pub polished: bool,
    pub audio_s: f32,
    pub post_roll_s: f32,
    pub mode: Option<String>,
    pub segments: Vec<SegmentInfo>,
    pub release_to_raw_s: f32,
    pub release_to_first_char_s: Option<f32>,
    pub release_to_done_s: f32,
    pub llm: Option<LlmStats>,
    pub audio: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum Event {
    Loading { stage: &'static str, progress: f32, reload: bool },
    Ready { load_s: f32, reload: bool, memory_gb: f32 },
    Unloaded { all: bool, memory_gb: f32 },
    Segment(SegmentInfo),
    CaptureDone { post_roll_s: f32 },
    Raw(String),
    Text(String),
    Done(Box<Outcome>),
    PromptsRebuilt { secs: f32 },
    Error(String),
}

pub type OnEvent = Box<dyn Fn(Event) + Send>;

pub struct Engine {
    tx: Sender<Command>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Engine {
    pub fn spawn(cfg: Config, backend: Arc<dyn Backend>, on_event: OnEvent) -> Engine {
        let (tx, rx) = channel();
        let thread = std::thread::Builder::new()
            .name("cz-engine".into())
            .spawn(move || Worker::new(cfg, backend, on_event).run(rx))
            .expect("spawn engine thread");
        Engine { tx, thread: Some(thread) }
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.tx.send(cmd);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ----------------------------------------------------------------------------- polisher

/// LLM cleanup with one cached prefix per mode (slot). None = polish only,
/// Some(lang) = polish + translate. All modes are prefilled at load.
struct Polisher {
    llm: Box<dyn Llm>,
    modes: Vec<Option<String>>,
    live: Option<usize>, // slot of the working sequence, if one is open
}


impl Polisher {
    fn build(&mut self, modes: &[Option<String>], dictionary: &[String]) -> Result<(), String> {
        self.modes.clear();
        for m in modes {
            if !self.modes.contains(m) && self.modes.len() < self.llm.slots() {
                self.modes.push(m.clone());
            }
        }
        for (i, m) in self.modes.iter().enumerate() {
            self.llm.set_prefix(i, &prompts::prefix(dictionary, m.as_deref()))?;
        }
        self.live = None;
        Ok(())
    }

    fn slot(&self, mode: &Option<String>) -> Option<usize> {
        self.modes.iter().position(|m| m == mode)
    }

    fn begin(&mut self, mode: &Option<String>) -> Result<(), String> {
        let slot = self.slot(mode).ok_or("unknown mode")?;
        self.llm.begin(slot)?;
        self.live = Some(slot);
        Ok(())
    }

    /// Polish. With `incremental`, the raw text was already fed since begin();
    /// only the template tail is prefilled here.
    fn run(&mut self, raw: &str, mode: &Option<String>, incremental: bool, max_tokens: usize,
           on_text: &mut dyn FnMut(&str)) -> Result<(String, LlmStats), String> {
        let t0 = Instant::now();
        let slot = self.slot(mode).ok_or("unknown mode")?;
        let mut after_release = 0;
        if !(incremental && self.live == Some(slot)) {
            self.llm.begin(slot)?;
            after_release += self.llm.feed(raw, false)?;
        }
        after_release += self.llm.feed(prompts::SUFFIX, true)?;
        self.live = None;
        let mut out = String::new();
        let mut ttft = None;
        let t_gen = Instant::now();
        let n = self.llm.generate(max_tokens, &mut |piece| {
            if ttft.is_none() && !piece.trim().is_empty() {
                ttft = Some(t0.elapsed().as_secs_f32());
            }
            out.push_str(piece);
            on_text(piece);
            true
        })?;
        self.llm.end();
        let gen_s = t_gen.elapsed().as_secs_f32();
        let text = match out.rfind("</think>") {
            Some(p) => out[p + 8..].trim().to_string(),
            None => out.trim().to_string(),
        };
        Ok((text, LlmStats {
            ttft_s: ttft,
            total_s: t0.elapsed().as_secs_f32(),
            prompt_tokens_after_release: after_release,
            gen_tokens: n,
            gen_tps: if gen_s > 0.0 { n as f32 / gen_s } else { 0.0 },
        }))
    }
}

// ----------------------------------------------------------------------------- worker

struct Worker {
    cfg: Config,
    backend: Arc<dyn Backend>,
    emit: OnEvent,
    asr: Option<Box<dyn Asr>>,
    polisher: Option<Polisher>,
    seg: Segmenter,
    // utterance state
    active: bool,
    mode: Option<String>,
    release: Option<(Instant, usize)>, // release time, sample count at release
    texts: Vec<String>,
    seg_log: Vec<SegmentInfo>,
}

fn file_gb(p: &std::path::Path) -> f32 {
    std::fs::metadata(p).map(|m| m.len() as f32 / 1e9).unwrap_or(0.0)
}

impl Worker {
    fn new(cfg: Config, backend: Arc<dyn Backend>, emit: OnEvent) -> Self {
        let seg = Segmenter::new(cfg.vad.clone());
        Worker {
            cfg, backend, emit, asr: None, polisher: None, seg,
            active: false, mode: None, release: None, texts: vec![], seg_log: vec![],
        }
    }

    fn run(mut self, rx: Receiver<Command>) {
        if let Err(e) = self.load(false) {
            (self.emit)(Event::Error(e));
            return;
        }
        while let Ok(cmd) = rx.recv() {
            let r = match cmd {
                Command::Quit => return,
                Command::Start(mode) => self.on_start(mode),
                Command::Audio(x) => self.on_audio(&x),
                Command::End(t) => self.on_end(t),
                Command::Flush => {
                    if self.active && self.release.is_some() { self.finalize() } else { Ok(()) }
                }
                Command::Cancel => {
                    self.active = false;
                    self.release = None;
                    if let Some(p) = &mut self.polisher {
                        p.live = None;
                        p.llm.end();
                    }
                    Ok(())
                }
                Command::Unload { all } => {
                    self.unload(all);
                    Ok(())
                }
                Command::Reload => self.load(true),
                Command::SetDictionary(words) => {
                    self.cfg.dictionary = words;
                    self.rebuild_prompts()
                }
                Command::SetTranslateLang(lang) => {
                    self.cfg.translate_lang = lang;
                    self.rebuild_prompts()
                }
                Command::Live(s) => {
                    self.cfg.language = s.language;
                    self.cfg.fast_path = s.fast_path;
                    self.cfg.post_roll_max_ms = s.post_roll_max_ms;
                    Ok(())
                }
            };
            if let Err(e) = r {
                self.active = false;
                (self.emit)(Event::Error(e));
            }
        }
    }

    fn memory_gb(&self) -> f32 {
        let mut gb = 0.0;
        if self.asr.is_some() {
            gb += file_gb(&self.cfg.asr.model) + file_gb(&self.cfg.asr.mmproj);
        }
        if let (Some(_), Some(l)) = (&self.polisher, &self.cfg.llm) {
            gb += file_gb(&l.model) + 0.5; // + KV / recurrent state
        }
        gb
    }

    fn modes(&self) -> Vec<Option<String>> {
        let mut v = vec![self.cfg.translate.clone(), None, self.cfg.translate_lang.clone()];
        v.retain(|m| m.as_deref() != Some(""));
        v
    }

    /// Load whatever isn't in memory yet (ASR and/or LLM), then warm up.
    fn load(&mut self, reload: bool) -> Result<(), String> {
        let t0 = Instant::now();
        if reload && self.asr.is_some() && (self.polisher.is_some() || self.cfg.llm.is_none()) {
            (self.emit)(Event::Ready { load_s: 0.0, reload, memory_gb: self.memory_gb() });
            return Ok(());
        }
        let noise: Vec<f32> = (0..SR * 2).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.02).collect();
        if self.asr.is_none() {
            (self.emit)(Event::Loading { stage: "asr", progress: 0.15, reload });
            let mut asr = self.backend.load_asr(&self.cfg.asr)?;
            asr.transcribe(&noise, "", self.cfg.language.as_deref())?; // compile GPU kernels now
            self.asr = Some(asr);
        }
        if self.polisher.is_none() {
            if let Some(spec) = self.cfg.llm.clone() {
                (self.emit)(Event::Loading { stage: "llm", progress: 0.45, reload });
                // one slot per mode: polish + Shift-translate (+ a default translate mode from the CLI)
                let slots = if self.cfg.translate.is_some() { 3 } else { 2 };
                let llm = self.backend.load_llm(&spec, slots)?;
                let mut p = Polisher { llm, modes: vec![], live: None };
                p.build(&self.modes(), &self.cfg.dictionary)?;
                (self.emit)(Event::Loading { stage: "warmup", progress: 0.85, reload });
                p.run("嗯那个我们明天开会吧", &None, false, 16, &mut |_| {})?;
                self.polisher = Some(p);
            }
        }
        if !reload {
            self.seg.push(&noise); // first VAD call allocates; do it now
            self.seg.reset();
        }
        (self.emit)(Event::Ready { load_s: t0.elapsed().as_secs_f32(), reload, memory_gb: self.memory_gb() });
        Ok(())
    }

    /// Free model weights to give memory back. ("Reload") brings them back.
    fn unload(&mut self, all: bool) {
        if self.active {
            return; // never in the middle of an utterance
        }
        self.polisher = None;
        if all {
            self.asr = None;
        }
        (self.emit)(Event::Unloaded { all, memory_gb: self.memory_gb() });
    }

    fn rebuild_prompts(&mut self) -> Result<(), String> {
        let modes = self.modes();
        if let Some(p) = &mut self.polisher {
            let t = Instant::now();
            p.build(&modes, &self.cfg.dictionary)?;
            (self.emit)(Event::PromptsRebuilt { secs: t.elapsed().as_secs_f32() });
        }
        Ok(())
    }

    fn on_start(&mut self, mode: Option<String>) -> Result<(), String> {
        if self.asr.is_none() {
            self.load(true)?;
        }
        self.mode = match mode {
            None => self.cfg.translate.clone(),
            Some(m) if m.is_empty() => None,
            Some(m) => Some(m),
        };
        if let Some(p) = &self.polisher {
            if p.slot(&self.mode).is_none() {
                self.mode = None; // e.g. translate requested but not prepared
            }
        }
        self.active = true;
        self.release = None;
        self.seg.reset();
        self.texts.clear();
        self.seg_log.clear();
        if self.cfg.incremental_prefill {
            let mode = self.mode.clone();
            if let Some(p) = &mut self.polisher {
                p.begin(&mode)?;
            }
        }
        Ok(())
    }

    fn transcribe(&mut self, start: usize, end: usize) -> Result<(), String> {
        let t = Instant::now();
        let ctx = prompts::asr_context(&self.cfg.dictionary);
        let asr = self.asr.as_mut().ok_or("speech model not loaded")?;
        let text = asr.transcribe(self.seg.audio(start, end), &ctx, self.cfg.language.as_deref())?;
        let asr_s = t.elapsed().as_secs_f32();
        let t = Instant::now();
        if !text.is_empty() {
            let before = join_segments(&self.texts);
            self.texts.push(text.clone());
            let after = join_segments(&self.texts);
            if let Some(p) = &mut self.polisher {
                if p.live.is_some() && p.llm.feed(&after[before.len()..], false).is_err() {
                    p.live = None; // e.g. context full: finalize will retry in one go, then fall back to raw
                }
            }
        }
        let info = SegmentInfo {
            start_s: start as f32 / SR as f32,
            end_s: end as f32 / SR as f32,
            asr_s,
            llm_prefill_s: t.elapsed().as_secs_f32(),
            text,
        };
        self.seg_log.push(info.clone());
        (self.emit)(Event::Segment(info));
        Ok(())
    }

    fn on_audio(&mut self, x: &[f32]) -> Result<(), String> {
        if !self.active {
            return Ok(()); // audio arriving after finalization
        }
        for (s, e) in self.seg.push(x) {
            self.transcribe(s, e)?;
        }
        if self.release.is_some() {
            self.maybe_finalize()?;
        }
        Ok(())
    }

    fn on_end(&mut self, t: Instant) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }
        self.release = Some((t, self.seg.total()));
        self.maybe_finalize()
    }

    fn maybe_finalize(&mut self) -> Result<(), String> {
        let (_, at) = self.release.unwrap();
        let after_ms = (self.seg.total() - at) as f32 * 1000.0 / SR as f32;
        if !self.seg.pending_speech()
            || self.seg.silence_ms() >= self.cfg.post_roll_silence_ms as f32
            || after_ms >= self.cfg.post_roll_max_ms as f32
        {
            self.finalize()?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), String> {
        self.active = false;
        let (t_release, at) = self.release.take().unwrap();
        let post_roll_s = (self.seg.total() - at) as f32 / SR as f32;
        (self.emit)(Event::CaptureDone { post_roll_s });
        if let Some((s, e)) = self.seg.finish() {
            self.transcribe(s, e)?;
        }
        let raw = join_segments(&self.texts);
        let release_to_raw_s = t_release.elapsed().as_secs_f32();
        (self.emit)(Event::Raw(raw.clone()));

        let mut first: Option<Instant> = None;
        let emit = &self.emit;
        let mut on_text = |chunk: &str| {
            if first.is_none() && !chunk.trim().is_empty() {
                first = Some(Instant::now());
            }
            emit(Event::Text(chunk.to_string()));
        };
        let mut llm = None;
        let (mut text, mut fallback) = (raw.clone(), false);
        let mode = self.mode.clone();
        match &mut self.polisher {
            Some(p) if !raw.is_empty() && needs_llm(&raw, mode.as_deref(), self.cfg.fast_path) => {
                match p.run(&raw, &mode, self.cfg.incremental_prefill, self.cfg.max_tokens, &mut on_text) {
                    Ok((out, stats)) => {
                        if suspicious(&raw, &out, mode.as_deref()) {
                            fallback = true;
                        } else {
                            text = out;
                        }
                        llm = Some(stats);
                    }
                    Err(e) => {
                        eprintln!("[cz] polish failed, using raw transcript: {e}");
                        p.live = None;
                        p.llm.end();
                        fallback = true;
                    }
                }
            }
            p => {
                if let Some(p) = p {
                    p.live = None;
                    p.llm.end();
                }
                if !raw.is_empty() {
                    on_text(&raw);
                }
            }
        }
        let outcome = Outcome {
            raw,
            text,
            fallback_to_raw: fallback,
            polished: llm.is_some(),
            audio_s: self.seg.total() as f32 / SR as f32,
            post_roll_s,
            mode,
            segments: std::mem::take(&mut self.seg_log),
            release_to_raw_s,
            release_to_first_char_s: first.map(|f| (f - t_release).as_secs_f32()),
            release_to_done_s: t_release.elapsed().as_secs_f32(),
            llm,
            audio: self.seg.all_audio().to_vec(),
        };
        (self.emit)(Event::Done(Box::new(outcome)));
        Ok(())
    }
}
