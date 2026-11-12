//! Safe Rust wrapper over the C shim (shim/cz_llama.cpp) implementing cz-core's backend traits.
use cz_core::backend::{Asr, AsrSpec, Backend, Llm, LlmSpec, Result};
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;
use std::sync::Once;

#[repr(C)]
struct CzAsr {
    _p: [u8; 0],
}
#[repr(C)]
struct CzLlm {
    _p: [u8; 0],
}
type PieceCb = extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int;

extern "C" {
    fn cz_init(verbose: c_int);
    fn cz_asr_load(model: *const c_char, mmproj: *const c_char) -> *mut CzAsr;
    fn cz_asr_free(a: *mut CzAsr);
    fn cz_asr_transcribe(
        a: *mut CzAsr, pcm: *const f32, n: usize, context: *const c_char, language: *const c_char,
        max_tokens: c_int, out: *mut c_char, out_cap: c_int,
    ) -> c_int;
    fn cz_llm_load(model: *const c_char, n_ctx: c_int, n_slots: c_int) -> *mut CzLlm;
    fn cz_llm_free(m: *mut CzLlm);
    fn cz_llm_set_prefix(m: *mut CzLlm, slot: c_int, text: *const c_char) -> c_int;
    fn cz_llm_begin(m: *mut CzLlm, slot: c_int) -> c_int;
    fn cz_llm_feed(m: *mut CzLlm, text: *const c_char, special: c_int) -> c_int;
    fn cz_llm_generate(m: *mut CzLlm, max_tokens: c_int, cb: PieceCb, user: *mut c_void) -> c_int;
    fn cz_llm_end(m: *mut CzLlm);
    fn cz_resampler_new(rate_in: c_int, rate_out: c_int) -> *mut c_void;
    fn cz_resampler_free(r: *mut c_void);
    fn cz_resampler_process(r: *mut c_void, input: *const f32, n: usize, out: *mut *const f32) -> i64;
}

static INIT: Once = Once::new();
fn init() {
    INIT.call_once(|| unsafe { cz_init(std::env::var("CZ_DEBUG").is_ok() as c_int) });
}

fn cpath(p: &Path) -> Result<CString> {
    CString::new(p.to_string_lossy().as_bytes()).map_err(|e| e.to_string())
}
fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap()
}

// ----------------------------------------------------------------------------- ASR

pub struct LlamaAsr(*mut CzAsr);
unsafe impl Send for LlamaAsr {}

impl Drop for LlamaAsr {
    fn drop(&mut self) {
        unsafe { cz_asr_free(self.0) }
    }
}

impl Asr for LlamaAsr {
    fn transcribe(&mut self, pcm: &[f32], context: &str, language: Option<&str>) -> Result<String> {
        let mut out = vec![0u8; 16 * 1024];
        let (c, l) = (cstr(context), cstr(language.unwrap_or("")));
        // ~13 tokens per second of audio is plenty of headroom for text
        let max_tokens = (pcm.len() as f32 / 16000.0 * 13.0) as c_int + 32;
        let n = unsafe {
            cz_asr_transcribe(self.0, pcm.as_ptr(), pcm.len(), c.as_ptr(), l.as_ptr(), max_tokens,
                              out.as_mut_ptr() as *mut c_char, out.len() as c_int)
        };
        if n < 0 {
            return Err(format!("语音识别失败（{n}）"));
        }
        out.truncate(n as usize);
        Ok(String::from_utf8_lossy(&out).trim().to_string())
    }
}

// ----------------------------------------------------------------------------- LLM

pub struct LlamaLlm {
    ptr: *mut CzLlm,
    slots: usize,
}
unsafe impl Send for LlamaLlm {}

impl Drop for LlamaLlm {
    fn drop(&mut self) {
        unsafe { cz_llm_free(self.ptr) }
    }
}

struct GenState<'a> {
    pending: Vec<u8>, // bytes of an unfinished UTF-8 sequence
    on_text: &'a mut dyn FnMut(&str) -> bool,
}

extern "C" fn on_piece(bytes: *const c_char, len: c_int, user: *mut c_void) -> c_int {
    let st = unsafe { &mut *(user as *mut GenState) };
    st.pending.extend_from_slice(unsafe { std::slice::from_raw_parts(bytes as *const u8, len as usize) });
    let valid = match std::str::from_utf8(&st.pending) {
        Ok(_) => st.pending.len(),
        Err(e) if e.error_len().is_none() => e.valid_up_to(), // incomplete tail: wait for more
        Err(_) => st.pending.len(),                            // invalid: emit lossily
    };
    if valid == 0 {
        return 0;
    }
    let chunk: Vec<u8> = st.pending.drain(..valid).collect();
    let s = String::from_utf8_lossy(&chunk);
    if (st.on_text)(&s) { 0 } else { 1 }
}

impl Llm for LlamaLlm {
    fn slots(&self) -> usize {
        self.slots
    }
    fn set_prefix(&mut self, slot: usize, text: &str) -> Result<usize> {
        let n = unsafe { cz_llm_set_prefix(self.ptr, slot as c_int, cstr(text).as_ptr()) };
        if n < 0 { Err(format!("预填充失败（{n}）")) } else { Ok(n as usize) }
    }
    fn begin(&mut self, slot: usize) -> Result<()> {
        let rc = unsafe { cz_llm_begin(self.ptr, slot as c_int) };
        if rc < 0 { Err(format!("无效的模式槽位 {slot}")) } else { Ok(()) }
    }
    fn feed(&mut self, text: &str, special: bool) -> Result<usize> {
        let n = unsafe { cz_llm_feed(self.ptr, cstr(text).as_ptr(), special as c_int) };
        if n < 0 { Err(format!("预填充失败（{n}）")) } else { Ok(n as usize) }
    }
    fn generate(&mut self, max_tokens: usize, on_text: &mut dyn FnMut(&str) -> bool) -> Result<usize> {
        let mut st = GenState { pending: Vec::new(), on_text };
        let n = unsafe {
            cz_llm_generate(self.ptr, max_tokens as c_int, on_piece, &mut st as *mut GenState as *mut c_void)
        };
        if !st.pending.is_empty() {
            let s = String::from_utf8_lossy(&st.pending).into_owned();
            (st.on_text)(&s);
        }
        if n < 0 { Err(format!("生成失败（{n}）")) } else { Ok(n as usize) }
    }
    fn end(&mut self) {
        unsafe { cz_llm_end(self.ptr) }
    }
}

// ----------------------------------------------------------------------------- backend

pub struct LlamaBackend;

impl Backend for LlamaBackend {
    fn load_asr(&self, spec: &AsrSpec) -> Result<Box<dyn Asr>> {
        init();
        let (m, p) = (cpath(&spec.model)?, cpath(&spec.mmproj)?);
        let ptr = unsafe { cz_asr_load(m.as_ptr(), p.as_ptr()) };
        if ptr.is_null() {
            return Err(format!("无法加载语音识别模型：{}", spec.model.display()));
        }
        Ok(Box::new(LlamaAsr(ptr)))
    }
    fn load_llm(&self, spec: &LlmSpec, slots: usize) -> Result<Box<dyn Llm>> {
        init();
        let m = cpath(&spec.model)?;
        let ptr = unsafe { cz_llm_load(m.as_ptr(), spec.n_ctx as c_int, slots as c_int) };
        if ptr.is_null() {
            return Err(format!("无法加载整理模型：{}", spec.model.display()));
        }
        Ok(Box::new(LlamaLlm { ptr, slots }))
    }
}

// ----------------------------------------------------------------------------- resampler

/// Streaming resampler (microphone rate -> 16 kHz).
pub struct Resampler(*mut c_void);
unsafe impl Send for Resampler {}

impl Resampler {
    pub fn new(rate_in: u32, rate_out: u32) -> Option<Self> {
        let p = unsafe { cz_resampler_new(rate_in as c_int, rate_out as c_int) };
        (!p.is_null()).then(|| Resampler(p))
    }
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let mut ptr: *const f32 = std::ptr::null();
        let n = unsafe { cz_resampler_process(self.0, input.as_ptr(), input.len(), &mut ptr) };
        if n > 0 && !ptr.is_null() {
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, n as usize) });
        }
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        unsafe { cz_resampler_free(self.0) }
    }
}
