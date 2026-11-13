//! Model catalog shown in the app and downloads (GGUF files from Hugging Face).
//! Downloads use the system `curl` (present on macOS and Windows 10+), so the app
//! carries no HTTP/TLS stack of its own.
use cz_core::backend::{AsrSpec, LlmSpec};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

pub struct Model {
    pub id: &'static str, // Hugging Face repo
    pub name: &'static str,
    pub blurb: &'static str,
    pub facts: &'static [(&'static str, &'static str)],
    pub files: &'static [(&'static str, u64)], // file name, bytes (ASR: model, mmproj)
    pub recommended: bool,
}

pub const ASR: &[Model] = &[
    Model {
        id: "ggml-org/Qwen3-ASR-0.6B-GGUF",
        name: "Qwen3-ASR 0.6B",
        blurb: "反应快，日常口述足够准。",
        facts: &[("大小", "1.0 GB"), ("内存", "约 1.2 GB")],
        files: &[("Qwen3-ASR-0.6B-Q8_0.gguf", 804_749_248), ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", 214_392_480)],
        recommended: true,
    },
    Model {
        id: "ggml-org/Qwen3-ASR-1.7B-GGUF",
        name: "Qwen3-ASR 1.7B",
        blurb: "更准一些，口音重、环境吵、专业词多时更稳；稍慢。",
        facts: &[("大小", "2.5 GB"), ("内存", "约 2.8 GB")],
        files: &[("Qwen3-ASR-1.7B-Q8_0.gguf", 2_165_034_944), ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", 355_709_344)],
        recommended: false,
    },
];

pub const LLM: &[Model] = &[
    Model {
        id: "unsloth/Qwen3.5-4B-GGUF",
        name: "Qwen3.5 4B",
        blurb: "整理得最干净：去口头禅、理解改口、自动分条。",
        facts: &[("大小", "2.7 GB"), ("内存", "约 3.2 GB")],
        files: &[("Qwen3.5-4B-Q4_K_M.gguf", 2_740_937_888)],
        recommended: true,
    },
    Model {
        id: "unsloth/Qwen3.5-2B-GGUF",
        name: "Qwen3.5 2B",
        blurb: "更快、更省内存，口头禅偶尔删不干净。",
        facts: &[("大小", "1.3 GB"), ("内存", "约 1.7 GB")],
        files: &[("Qwen3.5-2B-Q4_K_M.gguf", 1_280_835_840)],
        recommended: false,
    },
    Model {
        id: "",
        name: "不整理",
        blurb: "只做语音识别，原样输出，会保留口头禅和改口。",
        facts: &[],
        files: &[],
        recommended: false,
    },
];

pub const DEFAULT_ASR: &str = "ggml-org/Qwen3-ASR-0.6B-GGUF";
pub const DEFAULT_LLM: &str = "unsloth/Qwen3.5-4B-GGUF";

pub fn default_hub() -> PathBuf {
    crate::store::app_dir().join("gguf")
}

pub fn asr_spec(dir: &Path, id: &str) -> Option<AsrSpec> {
    let m = find(id).filter(|m| m.files.len() == 2 && is_downloaded(dir, id))?;
    let p = paths(dir, m);
    Some(AsrSpec { model: p[0].clone(), mmproj: p[1].clone() })
}

pub fn llm_spec(dir: &Path, id: &str) -> Option<LlmSpec> {
    let m = find(id).filter(|m| !m.files.is_empty() && is_downloaded(dir, id))?;
    Some(LlmSpec { model: paths(dir, m)[0].clone(), n_ctx: 2560 })
}

pub fn find(id: &str) -> Option<&'static Model> {
    ASR.iter().chain(LLM.iter()).find(|m| m.id == id)
}

fn size_gb(m: &Model) -> f64 {
    m.files.iter().map(|f| f.1).sum::<u64>() as f64 / 1e9
}

pub fn paths(dir: &Path, m: &Model) -> Vec<PathBuf> {
    m.files.iter().map(|f| dir.join(f.0)).collect()
}

pub fn is_downloaded(dir: &Path, id: &str) -> bool {
    match find(id) {
        Some(m) => m.files.iter().all(|(f, n)| std::fs::metadata(dir.join(f)).map(|md| md.len() == *n).unwrap_or(false)),
        None => id.is_empty(),
    }
}

#[derive(Clone)]
struct Dl {
    status: &'static str, // downloading | done | error
    error: Option<String>,
}

/// Background downloads with progress = bytes on disk / expected bytes.
#[derive(Clone)]
pub struct Downloads {
    dir: PathBuf,
    active: Arc<Mutex<HashMap<String, Dl>>>,
    on_change: Arc<dyn Fn(&str) + Send + Sync>,
}

impl Downloads {
    pub fn new(dir: PathBuf, on_change: Arc<dyn Fn(&str) + Send + Sync>) -> Downloads {
        Downloads { dir, active: Arc::default(), on_change }
    }

    /// The download_source setting. GGUF checkpoints are only fetched from Hugging Face
    /// for now; ModelScope (macOS) mirrors the MLX repos.
    pub fn set_source(&self, _pref: &str) {}

    pub fn state(&self, id: &str) -> Option<Value> {
        let d = self.active.lock().unwrap().get(id).cloned()?;
        let m = find(id)?;
        let progress = if d.status == "done" {
            1.0
        } else {
            let total: u64 = m.files.iter().map(|f| f.1).sum();
            let have: u64 = m.files.iter().map(|(f, n)| {
                let done = std::fs::metadata(self.dir.join(f)).map(|md| md.len()).unwrap_or(0);
                let part = std::fs::metadata(self.dir.join(format!("{f}.part"))).map(|md| md.len()).unwrap_or(0);
                (done.max(part)).min(*n)
            }).sum();
            (have as f64 / total.max(1) as f64).min(0.99)
        };
        Some(json!({"status": d.status, "progress": progress, "error": d.error}))
    }

    pub fn progress(&self, id: &str) -> f64 {
        if is_downloaded(&self.dir, id) {
            return 1.0;
        }
        self.state(id).and_then(|s| s["progress"].as_f64()).unwrap_or(0.0)
    }

    pub fn start(&self, id: &str) {
        let Some(m) = find(id) else { return };
        {
            let mut a = self.active.lock().unwrap();
            if a.get(id).map(|d| d.status == "downloading").unwrap_or(false) {
                return;
            }
            a.insert(id.to_string(), Dl { status: "downloading", error: None });
        }
        let this = self.clone();
        let id = id.to_string();
        std::thread::spawn(move || {
            let _ = std::fs::create_dir_all(&this.dir);
            let mut err = None;
            for (f, n) in m.files {
                let dst = this.dir.join(f);
                if std::fs::metadata(&dst).map(|md| md.len() == *n).unwrap_or(false) {
                    continue;
                }
                let part = this.dir.join(format!("{f}.part"));
                let url = format!("https://huggingface.co/{}/resolve/main/{f}", m.id);
                let ok = Command::new("curl")
                    .args(["-fsSL", "--retry", "3", "-C", "-", "-o"])
                    .arg(&part)
                    .arg(&url)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if ok && std::fs::metadata(&part).map(|md| md.len() == *n).unwrap_or(false) {
                    let _ = std::fs::rename(&part, &dst);
                } else {
                    err = Some(format!("{f} 下载失败"));
                    break;
                }
            }
            this.active.lock().unwrap().insert(
                id.clone(),
                Dl { status: if err.is_some() { "error" } else { "done" }, error: err },
            );
            (this.on_change)(&id);
        });
        (self.on_change)(m.id);
    }

    pub fn catalog(&self, asr_sel: &str, llm_sel: &str) -> Value {
        let decorate = |items: &[Model], current: &str| -> Vec<Value> {
            items.iter().map(|m| {
                let facts: serde_json::Map<String, Value> = m.facts.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
                json!({
                    "id": m.id, "name": m.name, "blurb": m.blurb, "facts": facts,
                    "size_gb": if m.files.is_empty() { json!(0) } else { json!(size_gb(m)) },
                    "recommended": m.recommended,
                    "selected": m.id == current,
                    "downloaded": is_downloaded(&self.dir, m.id),
                    "download": self.state(m.id),
                })
            }).collect()
        };
        json!({"asr": decorate(ASR, asr_sel), "llm": decorate(LLM, llm_sel)})
    }
}
