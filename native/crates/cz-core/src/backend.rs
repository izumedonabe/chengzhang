//! Inference backend interface. The engine only talks to these traits, so a
//! Windows build (llama.cpp with Vulkan/CPU) or another runtime can be swapped in.

pub type Result<T> = std::result::Result<T, String>;

/// Speech recognition on 16 kHz mono f32 PCM, one call per VAD segment.
pub trait Asr: Send {
    /// `context`: biasing text (personal dictionary). `language`: force a language
    /// ("Chinese", "English"…) or None to auto-detect.
    fn transcribe(&mut self, pcm: &[f32], context: &str, language: Option<&str>) -> Result<String>;
}

/// The polishing LLM with cached prompt prefixes and one incremental working sequence.
pub trait Llm: Send {
    /// Number of prefix slots (one per mode).
    fn slots(&self) -> usize;
    /// Prefill `text` (chat template already rendered, special tokens allowed) into a slot.
    fn set_prefix(&mut self, slot: usize, text: &str) -> Result<usize>;
    /// Start a working sequence as a cheap copy of a slot.
    fn begin(&mut self, slot: usize) -> Result<()>;
    /// Append text to the working sequence (prefill). Returns tokens fed.
    fn feed(&mut self, text: &str, special: bool) -> Result<usize>;
    /// Greedy generation; `on_text` gets UTF-8 pieces and returns false to stop.
    fn generate(&mut self, max_tokens: usize, on_text: &mut dyn FnMut(&str) -> bool) -> Result<usize>;
    /// Drop the working sequence.
    fn end(&mut self);
}

/// Where the model files are and how to load them.
pub trait Backend: Send + Sync {
    fn load_asr(&self, spec: &AsrSpec) -> Result<Box<dyn Asr>>;
    fn load_llm(&self, spec: &LlmSpec, slots: usize) -> Result<Box<dyn Llm>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct AsrSpec {
    pub model: std::path::PathBuf,
    pub mmproj: std::path::PathBuf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LlmSpec {
    pub model: std::path::PathBuf,
    pub n_ctx: usize,
}
