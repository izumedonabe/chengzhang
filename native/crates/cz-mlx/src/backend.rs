//! cz-core backend traits on MLX.
//!   - prefix slots are cache snapshots (MLX arrays are immutable and refcounted, so a
//!     "copy" is free — no GPU work at all, unlike llama.cpp's seq_cp)
//!   - feed() prefills asynchronously: the GPU works while the user keeps talking
use crate::asr::Qwen3Asr;
use crate::qwen35::{self, Cache, Qwen35};
use crate::tokenizer::Tokenizer;
use cz_core::backend::{Asr, AsrSpec, Backend, Llm, LlmSpec, Result};
use mlx_rs::Array;

pub struct MlxAsr(Qwen3Asr);

impl Asr for MlxAsr {
    fn transcribe(&mut self, pcm: &[f32], context: &str, language: Option<&str>) -> Result<String> {
        // ~13 tokens per second of audio is plenty of headroom for text
        let max_tokens = (pcm.len() as f32 / 16000.0 * 13.0) as usize + 32;
        wired(|| self.0.transcribe(pcm, context, language, max_tokens))
    }
}

pub struct MlxLlm {
    model: Qwen35,
    tok: Tokenizer,
    slots: Vec<Option<Cache>>,
    work: Option<Cache>,
    last: Option<Array>, // final hidden state of the last fed token
}

// MLX arrays are reference-counted handles; the engine uses this from one thread.
unsafe impl Send for MlxLlm {}
unsafe impl Send for MlxAsr {}

fn async_eval_cache(c: &Cache) -> Result<()> {
    let mut arrs: Vec<&Array> = vec![];
    for l in c {
        match l {
            qwen35::LayerCache::Gdn { conv, state } => arrs.extend(conv.iter().chain(state.iter())),
            qwen35::LayerCache::Kv { k, v, .. } => arrs.extend(k.iter().chain(v.iter())),
        }
    }
    mlx_rs::transforms::async_eval(arrs).map_err(|e| e.to_string())
}

impl MlxLlm {
    fn prefill(&self, toks: &[u32], cache: &mut Cache) -> Result<Array> {
        let mut last = None;
        for chunk in toks.chunks(512) {
            let h = self.model.forward(chunk, cache)?;
            async_eval_cache(cache)?;
            last = Some(h);
        }
        last.ok_or_else(|| "empty prefill".into())
    }
}

impl Llm for MlxLlm {
    fn slots(&self) -> usize {
        self.slots.len()
    }

    fn set_prefix(&mut self, slot: usize, text: &str) -> Result<usize> {
        let toks = self.tok.encode(text, true);
        let mut cache = self.model.make_cache();
        self.prefill(&toks, &mut cache)?;
        qwen35::eval_cache(&cache)?;
        *self.slots.get_mut(slot).ok_or("bad slot")? = Some(cache);
        Ok(toks.len())
    }

    fn begin(&mut self, slot: usize) -> Result<()> {
        let c = self.slots.get(slot).and_then(|c| c.clone()).ok_or("prefix slot not prepared")?;
        self.work = Some(c);
        self.last = None;
        Ok(())
    }

    fn feed(&mut self, text: &str, special: bool) -> Result<usize> {
        let toks = self.tok.encode(text, special);
        if toks.is_empty() {
            return Ok(0);
        }
        let mut cache = self.work.take().ok_or("no working sequence")?;
        let h = self.prefill(&toks, &mut cache);
        self.work = Some(cache);
        self.last = Some(h?);
        Ok(toks.len())
    }

    fn generate(&mut self, max_tokens: usize, on_text: &mut dyn FnMut(&str) -> bool) -> Result<usize> {
        wired(|| self.generate_inner(max_tokens, on_text))
    }

    fn end(&mut self) {
        self.work = None;
        self.last = None;
    }
}

impl MlxLlm {
    fn generate_inner(&mut self, max_tokens: usize, on_text: &mut dyn FnMut(&str) -> bool) -> Result<usize> {
        let h = self.last.take().ok_or("nothing to generate from")?;
        let mut cache = self.work.take().ok_or("no working sequence")?;
        let m = &self.model;
        let e = |x: mlx_rs::error::Exception| x.to_string();
        let mut y = m.argmax_last(&h)?;
        mlx_rs::transforms::async_eval([&y]).map_err(e)?;
        let mut pending: Vec<u8> = vec![];
        let mut n = 0;
        while n < max_tokens {
            // queue the next step before reading this token, so the GPU never idles
            let hh = m.forward_ids(&y.reshape(&[1, 1]).map_err(e)?, &mut cache)?;
            let ny = m.argmax_last(&hh)?;
            mlx_rs::transforms::async_eval([&ny]).map_err(e)?;
            let id = y.item_cast::<u32>();
            if m.cfg.eos.contains(&id) {
                break;
            }
            n += 1;
            self.tok.token_bytes(id, &mut pending);
            let valid = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(err) if err.error_len().is_none() => err.valid_up_to(),
                Err(_) => pending.len(),
            };
            if valid > 0 {
                let chunk: Vec<u8> = pending.drain(..valid).collect();
                if !on_text(&String::from_utf8_lossy(&chunk)) {
                    break;
                }
            }
            y = ny;
        }
        if !pending.is_empty() {
            on_text(&String::from_utf8_lossy(&pending));
        }
        Ok(n)
    }
}

fn wired_cap() -> usize {
    std::env::var("CZ_WIRED_GB").ok().and_then(|v| v.parse::<f64>().ok())
        .map(|gb| (gb * 1e9) as usize)
        .unwrap_or_else(|| (physical_memory() / 3).min(8_000_000_000))
}

/// Keep buffers resident while `f` runs (what mlx-lm does around generation), then let
/// macOS reclaim them again. CZ_WIRED_MODE=always|never overrides.
fn wired<T>(f: impl FnOnce() -> T) -> T {
    let mode = std::env::var("CZ_WIRED_MODE").unwrap_or_default();
    if mode == "always" || mode == "never" {
        return f();
    }
    let old = mlx_rs::memory::set_wired_limit(wired_cap()).unwrap_or(0);
    let r = f();
    let _ = mlx_rs::transforms::eval(std::iter::empty::<&Array>()); // nothing queued is left mid-flight
    let _ = mlx_rs::memory::set_wired_limit(old);
    r
}

fn physical_memory() -> usize {
    extern "C" {
        fn sysctlbyname(name: *const i8, oldp: *mut std::ffi::c_void, oldlenp: *mut usize, newp: *mut std::ffi::c_void, newlen: usize) -> i32;
    }
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    unsafe { sysctlbyname(c"hw.memsize".as_ptr(), &mut v as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0) };
    v as usize
}

pub struct MlxBackend;

impl MlxBackend {
    /// Point MLX at the kernel library shipped next to the executable / in Resources.
    pub fn init() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // Keep model buffers resident (like mlx-lm does while generating): under memory
            // pressure, unwired weights get compressed/swapped and the next dictation pays
            // for it (measured: LLM prefill 30 ms wired vs 260 ms unwired). The limit is a
            // cap, not a reservation; unloading the models releases it.
            // MLX keeps freed buffers for reuse; without a cap that cache holds on to every
            // temporary from prefill/warmup (measured +2 GB idle footprint).
            let _ = mlx_rs::memory::set_cache_limit(256 << 20);
            if std::env::var("CZ_WIRED_MODE").as_deref() == Ok("always") {
                let _ = mlx_rs::memory::set_wired_limit(wired_cap());
            }
            if let Ok(exe) = std::env::current_exe() {
                let dir = exe.parent().map(|p| p.to_path_buf()).unwrap_or_default();
                for p in [dir.join("../Resources/mlx.metallib"), dir.join("mlx.metallib")] {
                    if p.exists() {
                        let _ = mlx_rs::metal::set_metallib_path(p.to_string_lossy());
                        break;
                    }
                }
            }
        });
    }
}

impl Backend for MlxBackend {
    fn load_asr(&self, spec: &AsrSpec) -> Result<Box<dyn Asr>> {
        Self::init();
        Ok(Box::new(MlxAsr(Qwen3Asr::load(&spec.model)?)))
    }

    fn load_llm(&self, spec: &LlmSpec, slots: usize) -> Result<Box<dyn Llm>> {
        Self::init();
        let model = Qwen35::load(&spec.model)?;
        let tok = Tokenizer::load(&spec.model)?;
        Ok(Box::new(MlxLlm { model, tok, slots: vec![None; slots], work: None, last: None }))
    }
}
