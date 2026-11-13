//! 16 kHz mono capture from a chosen device (port of the Python Mic class).
//! The cpal stream lives on its own thread. keep_open: the stream keeps running and
//! the last 0.5 s before the hotkey are kept, so the first syllable is never lost.
//!
//! cpal 0.15 on macOS never stops a dropped input stream (its disconnect listener keeps
//! the stream alive), so a dropped stream goes on feeding us audio. Streams are therefore
//! paused, never just dropped: between dictations the one stream is paused (mic off) and
//! resumed for the next one.
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

const BUILTIN_HINTS: &[&str] = &["MacBook", "Built-in", "内建", "内置", "iMac", "Studio Display"];
const BLUETOOTH_HINTS: &[&str] = &["AirPods", "Beats", "Bluetooth", "蓝牙", "Buds", "WH-", "WF-"];

pub struct MicInfo {
    pub name: String,
    pub default: bool,
    pub builtin: bool,
    pub bluetooth: bool,
}

pub fn list_mics() -> Vec<MicInfo> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok()).unwrap_or_default();
    let mut out = vec![];
    if let Ok(devs) = host.input_devices() {
        for d in devs {
            if let Ok(n) = d.name() {
                out.push(MicInfo {
                    default: n == default,
                    builtin: BUILTIN_HINTS.iter().any(|h| n.contains(h)),
                    bluetooth: BLUETOOTH_HINTS.iter().any(|h| n.contains(h)),
                    name: n,
                });
            }
        }
    }
    out
}

/// "auto" = the built-in mic when there is one (AirPods-style headsets need ~0.5–1 s to
/// switch into call mode, which eats the first words). Otherwise a device name.
pub fn resolve_mic(pref: &str) -> Option<String> {
    let mics = list_mics();
    if !pref.is_empty() && pref != "auto" {
        if let Some(m) = mics.iter().find(|m| m.name == pref) {
            return Some(m.name.clone());
        }
    }
    mics.iter().find(|m| m.builtin).map(|m| m.name.clone())
}

enum Cmd {
    Configure { device: Option<String>, keep_open: bool },
    Start(Sender<Result<(), String>>),
    Stop,
}

pub type OnAudio = Arc<dyn Fn(Vec<f32>) + Send + Sync>;

pub struct Mic {
    tx: Sender<Cmd>,
    level: Arc<AtomicU32>,
    capturing: Arc<AtomicBool>,
    pub name: Arc<Mutex<String>>,
}

struct Shared {
    capturing: Arc<AtomicBool>,
    keep_open: AtomicBool,
    level: Arc<AtomicU32>,
    ring: Mutex<VecDeque<Vec<f32>>>,
    on_audio: OnAudio,
}

impl Mic {
    pub fn new(on_audio: OnAudio) -> Mic {
        let (tx, rx) = channel::<Cmd>();
        let level = Arc::new(AtomicU32::new(0));
        let capturing = Arc::new(AtomicBool::new(false));
        let name = Arc::new(Mutex::new(String::new()));
        let shared = Arc::new(Shared {
            capturing: capturing.clone(),
            keep_open: AtomicBool::new(false),
            level: level.clone(),
            ring: Mutex::new(VecDeque::new()),
            on_audio,
        });
        let name2 = name.clone();
        std::thread::Builder::new().name("cz-mic".into()).spawn(move || {
            let mut stream: Option<cpal::Stream> = None;
            let mut paused = false;
            let mut device: Option<String> = None;
            for cmd in rx {
                match cmd {
                    Cmd::Configure { device: d, keep_open } => {
                        let changed = d != device || keep_open != shared.keep_open.load(Ordering::Relaxed);
                        device = d;
                        shared.keep_open.store(keep_open, Ordering::Relaxed);
                        if changed && !shared.capturing.load(Ordering::Relaxed) {
                            close(&mut stream);
                            shared.ring.lock().unwrap().clear();
                            if keep_open {
                                stream = open(&device, &shared, &name2).ok();
                                paused = false;
                            }
                        }
                    }
                    Cmd::Start(reply) => {
                        if paused {
                            paused = false;
                            if stream.as_ref().is_some_and(|s| s.play().is_err()) {
                                close(&mut stream);
                            }
                        }
                        if stream.is_none() {
                            match open(&device, &shared, &name2) {
                                Ok(s) => stream = Some(s),
                                Err(e) => {
                                    let _ = reply.send(Err(e));
                                    continue;
                                }
                            }
                        } else {
                            // audio from just before the key went down
                            let pre: Vec<Vec<f32>> = shared.ring.lock().unwrap().drain(..).collect();
                            for x in pre {
                                (shared.on_audio)(x);
                            }
                        }
                        shared.capturing.store(true, Ordering::Relaxed);
                        let _ = reply.send(Ok(()));
                    }
                    Cmd::Stop => {
                        shared.capturing.store(false, Ordering::Relaxed);
                        shared.level.store(0, Ordering::Relaxed);
                        if !shared.keep_open.load(Ordering::Relaxed) {
                            // pausing stops the device (the mic indicator goes off); the same
                            // stream is resumed for the next dictation
                            if let Some(s) = &stream {
                                if s.pause().is_err() {
                                    close(&mut stream);
                                }
                            }
                            paused = stream.is_some();
                        }
                    }
                }
            }
        }).expect("spawn mic thread");
        Mic { tx, level, capturing, name }
    }

    pub fn configure(&self, device: Option<String>, keep_open: bool) {
        let _ = self.tx.send(Cmd::Configure { device, keep_open });
    }

    pub fn start(&self) -> Result<(), String> {
        let (tx, rx) = channel();
        let _ = self.tx.send(Cmd::Start(tx));
        rx.recv().unwrap_or(Err("麦克风线程已退出".into()))
    }

    pub fn stop(&self) {
        let _ = self.tx.send(Cmd::Stop);
    }

    pub fn capturing(&self) -> bool {
        self.capturing.load(Ordering::Relaxed)
    }

    pub fn level(&self) -> f32 {
        f32::from_bits(self.level.load(Ordering::Relaxed))
    }
}

/// Stop a stream for good. Dropping alone isn't enough (see the top of this file).
fn close(stream: &mut Option<cpal::Stream>) {
    if let Some(s) = stream.take() {
        let _ = s.pause();
    }
}

fn open(device: &Option<String>, sh: &Arc<Shared>, name: &Arc<Mutex<String>>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let dev = match device {
        Some(n) => host
            .input_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().map(|x| &x == n).unwrap_or(false))
            .or_else(|| host.default_input_device()),
        None => host.default_input_device(),
    }
    .ok_or("没有可用的麦克风")?;
    *name.lock().unwrap() = dev.name().unwrap_or_default();
    let cfg = dev.default_input_config().map_err(|e| e.to_string())?;
    let rate = cfg.sample_rate().0;
    let channels = cfg.channels() as usize;
    let mut resampler = (rate != 16000).then(|| cz_core::resample::Resampler::new(rate, 16000));
    let sh2 = sh.clone();
    let preroll_blocks = 16usize; // ≈ 0.5 s of 32 ms blocks
    let mut deliver = move |mono: &[f32]| {
        let mut x = Vec::with_capacity(mono.len() / 2);
        match &mut resampler {
            Some(r) => r.process(mono, &mut x),
            None => x.extend_from_slice(mono),
        }
        if x.is_empty() {
            return;
        }
        let rms = (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt() + 1e-9;
        // map roughly -60..-10 dBFS to 0..1, smoothed
        let lvl = ((20.0 * rms.log10() + 60.0) / 50.0).clamp(0.0, 1.0);
        let prev = f32::from_bits(sh2.level.load(Ordering::Relaxed));
        sh2.level.store((0.6 * prev + 0.4 * lvl).to_bits(), Ordering::Relaxed);
        if sh2.capturing.load(Ordering::Relaxed) {
            (sh2.on_audio)(x);
        } else if sh2.keep_open.load(Ordering::Relaxed) {
            let mut ring = sh2.ring.lock().unwrap();
            ring.push_back(x);
            while ring.len() > preroll_blocks {
                ring.pop_front();
            }
        }
    };
    let err = |e| eprintln!("[cz] mic error: {e}");
    let sc: cpal::StreamConfig = cfg.clone().into();
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(&sc, move |d: &[f32], _: &_| {
            let mono: Vec<f32> = d.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
            deliver(&mono);
        }, err, None),
        cpal::SampleFormat::I16 => dev.build_input_stream(&sc, move |d: &[i16], _: &_| {
            let mono: Vec<f32> = d.chunks(channels)
                .map(|f| f.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / channels as f32).collect();
            deliver(&mono);
        }, err, None),
        f => return Err(format!("不支持的音频格式 {f:?}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// Uses the real microphone (only counts samples). Run with
    /// `cargo test -p chengzhang -- --ignored one_stream`.
    /// Each dictation must get about as many samples as it lasted, however many came
    /// before it, and nothing may arrive between dictations.
    #[test]
    #[ignore]
    fn one_stream_however_many_dictations() {
        let got = Arc::new(AtomicUsize::new(0));
        let g = got.clone();
        let mic = Mic::new(Arc::new(move |x: Vec<f32>| {
            g.fetch_add(x.len(), Ordering::SeqCst);
        }));
        mic.configure(resolve_mic("auto"), false);
        for i in 0..5 {
            got.store(0, Ordering::SeqCst);
            mic.start().unwrap();
            std::thread::sleep(Duration::from_millis(1000));
            mic.stop();
            std::thread::sleep(Duration::from_millis(300));
            let n = got.swap(0, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(700));
            let idle = got.load(Ordering::SeqCst);
            eprintln!("dictation {i}: {:.2} s of audio for 1.0 s, {idle} samples while stopped", n as f32 / 16000.0);
            assert!((12000..=20000).contains(&n), "dictation {i}: {n} samples for 1 s");
            assert_eq!(idle, 0, "audio kept coming after stop");
        }
    }
}
