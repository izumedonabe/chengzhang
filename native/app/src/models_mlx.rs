//! macOS model catalog (MLX checkpoints, same ids as the Python app) and downloads.
//! Models live in the Hugging Face cache layout (~/.cache/huggingface/hub), so the
//! Python and native apps share one copy. Downloads use the system `curl`, from Hugging
//! Face or from ModelScope (modelscope.cn mirrors the same mlx-community repos and is
//! reachable from mainland China); "auto" tries Hugging Face first and falls back.
use cz_core::backend::{AsrSpec, LlmSpec};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

pub const DEFAULT_ASR: &str = "mlx-community/Qwen3-ASR-0.6B-8bit";
pub const DEFAULT_LLM: &str = "mlx-community/Qwen3.5-4B-MLX-4bit";

pub struct Model {
    pub id: &'static str,
    pub name: &'static str,
    pub blurb: &'static str,
    pub facts: &'static [(&'static str, &'static str)],
    pub size_gb: f64,
    pub recommended: bool,
}

pub const ASR: &[Model] = &[
    Model { id: "mlx-community/Qwen3-ASR-0.6B-8bit", name: "Qwen3-ASR 0.6B", blurb: "反应快，日常口述足够准。",
            facts: &[("大小", "1.0 GB"), ("内存", "约 1.2 GB")], size_gb: 1.01, recommended: true },
    Model { id: "mlx-community/Qwen3-ASR-1.7B-8bit", name: "Qwen3-ASR 1.7B", blurb: "更准一些，口音重、环境吵、专业词多时更稳；稍慢。",
            facts: &[("大小", "2.5 GB"), ("内存", "约 2.8 GB")], size_gb: 2.47, recommended: false },
];

pub const LLM: &[Model] = &[
    Model { id: "mlx-community/Qwen3.5-4B-MLX-4bit", name: "Qwen3.5 4B", blurb: "整理得最干净：去口头禅、理解改口、自动分条。",
            facts: &[("大小", "3.1 GB"), ("内存", "约 3.2 GB")], size_gb: 3.06, recommended: true },
    Model { id: "mlx-community/Qwen3.5-2B-MLX-4bit", name: "Qwen3.5 2B", blurb: "更快、更省内存，口头禅偶尔删不干净。",
            facts: &[("大小", "1.8 GB"), ("内存", "约 1.7 GB")], size_gb: 1.75, recommended: false },
    Model { id: "mlx-community/Qwen3.5-9B-MLX-4bit", name: "Qwen3.5 9B", blurb: "理解力更强，适合长段口述；生成更慢，内存要求更高。",
            facts: &[("大小", "6.0 GB"), ("内存", "约 6.5 GB")], size_gb: 5.98, recommended: false },
    Model { id: "", name: "不整理", blurb: "只做语音识别，原样输出，会保留口头禅和改口。",
            facts: &[], size_gb: 0.0, recommended: false },
];

pub fn find(id: &str) -> Option<&'static Model> {
    ASR.iter().chain(LLM.iter()).find(|m| m.id == id)
}

fn repo_dir(hub: &Path, id: &str) -> PathBuf {
    hub.join(format!("models--{}", id.replace('/', "--")))
}

/// The snapshot directory with a complete checkpoint, if there is one.
pub fn local_path(hub: &Path, id: &str) -> Option<PathBuf> {
    if id.is_empty() {
        return None;
    }
    let repo = repo_dir(hub, id);
    let mut candidates = vec![];
    if let Ok(sha) = std::fs::read_to_string(repo.join("refs/main")) {
        candidates.push(repo.join("snapshots").join(sha.trim()));
    }
    if let Ok(rd) = std::fs::read_dir(repo.join("snapshots")) {
        candidates.extend(rd.filter_map(|x| x.ok().map(|x| x.path())));
    }
    candidates.into_iter().find(|d| complete(d))
}

fn complete(d: &Path) -> bool {
    if !d.join("config.json").exists() {
        return false;
    }
    let has_tok = d.join("tokenizer.json").exists() || d.join("vocab.json").exists();
    let weights: Vec<String> = std::fs::read_to_string(d.join("model.safetensors.index.json")).ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["weight_map"].as_object().map(|m| m.values().filter_map(|f| f.as_str().map(String::from)).collect()))
        .unwrap_or_else(|| vec!["model.safetensors".into()]);
    has_tok && weights.iter().all(|f| std::fs::metadata(d.join(f)).map(|m| m.len() > 0).unwrap_or(false))
}

pub fn is_downloaded(hub: &Path, id: &str) -> bool {
    id.is_empty() || local_path(hub, id).is_some()
}

pub fn asr_spec(hub: &Path, id: &str) -> Option<AsrSpec> {
    local_path(hub, id).map(|model| AsrSpec { model, mmproj: PathBuf::new() })
}

pub fn llm_spec(hub: &Path, id: &str) -> Option<LlmSpec> {
    local_path(hub, id).map(|model| LlmSpec { model, n_ctx: 0 })
}

pub fn default_hub() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(h);
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

// ----------------------------------------------------------------------------- downloads

#[derive(Clone)]
struct Dl {
    status: &'static str, // downloading | done | error
    error: Option<String>,
    files: Vec<(String, u64)>,
    dir: Option<PathBuf>,
    source: Option<Source>,
}

#[derive(Clone)]
pub struct Downloads {
    hub: PathBuf,
    active: Arc<Mutex<HashMap<String, Dl>>>,
    on_change: Arc<dyn Fn(&str) + Send + Sync>,
    /// "auto" | "huggingface" | "modelscope" (the download_source setting)
    pref: Arc<Mutex<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    HuggingFace,
    ModelScope,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::HuggingFace => "Hugging Face",
            Source::ModelScope => "ModelScope",
        }
    }

    /// (revision, [(path, size)]) of everything the checkpoint needs.
    fn listing(self, id: &str) -> Result<(String, Vec<(String, u64)>), String> {
        let skip = |p: &str| matches!(p, ".gitattributes" | "README.md" | "configuration.json");
        match self {
            Source::HuggingFace => {
                let info = curl_json(&format!("https://huggingface.co/api/models/{id}"))?;
                let sha = info["sha"].as_str().ok_or("没有找到模型版本")?.to_string();
                let tree = curl_json(&format!("https://huggingface.co/api/models/{id}/tree/{sha}?recursive=true"))?;
                let files = tree.as_array().ok_or("无法读取文件列表")?.iter()
                    .filter(|f| f["type"] == "file")
                    .filter_map(|f| Some((f["path"].as_str()?.to_string(), f["size"].as_u64().unwrap_or(0))))
                    .filter(|(p, _)| !skip(p))
                    .collect();
                Ok((sha, files))
            }
            Source::ModelScope => {
                let v = curl_json(&format!("https://modelscope.cn/api/v1/models/{id}/repo/files?Revision=master&Recursive=true"))?;
                if v["Code"].as_i64() != Some(200) {
                    return Err(v["Message"].as_str().unwrap_or("ModelScope 上没有这个模型").to_string());
                }
                let all = v["Data"]["Files"].as_array().ok_or("无法读取文件列表")?;
                let blobs: Vec<&Value> = all.iter().filter(|f| f["Type"] == "blob").collect();
                let rev = blobs.iter().find_map(|f| f["Revision"].as_str()).unwrap_or("master").to_string();
                let files = blobs.iter()
                    .filter_map(|f| Some((f["Path"].as_str()?.to_string(), f["Size"].as_u64().unwrap_or(0))))
                    .filter(|(p, _)| !skip(p))
                    .collect();
                Ok((rev, files))
            }
        }
    }

    fn url(self, id: &str, rev: &str, file: &str) -> String {
        match self {
            Source::HuggingFace => format!("https://huggingface.co/{id}/resolve/{rev}/{file}"),
            Source::ModelScope => format!("https://modelscope.cn/models/{id}/resolve/{rev}/{file}"),
        }
    }
}

fn curl_json(url: &str) -> Result<Value, String> {
    let out = Command::new("curl").args(["-fsSL", "--retry", "3", "--connect-timeout", "8", url])
        .output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("无法访问 {url}"));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())
}

/// Can we reach Hugging Face right now? A quick probe, so "auto" doesn't hang on a
/// network that blocks it.
fn huggingface_reachable() -> bool {
    Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-m", "5", "-w", "%{http_code}", "https://huggingface.co/api/models/mlx-community/Qwen3-ASR-0.6B-8bit"])
        .output().map(|o| o.stdout == b"200").unwrap_or(false)
}

impl Downloads {
    pub fn new(hub: PathBuf, on_change: Arc<dyn Fn(&str) + Send + Sync>) -> Downloads {
        Downloads { hub, active: Arc::default(), on_change, pref: Arc::new(Mutex::new("auto".into())) }
    }

    /// The download_source setting: "auto", "huggingface" or "modelscope".
    pub fn set_source(&self, pref: &str) {
        *self.pref.lock().unwrap() = pref.to_string();
    }

    /// Where to try, in order.
    fn sources(&self) -> Vec<Source> {
        match self.pref.lock().unwrap().as_str() {
            "huggingface" => vec![Source::HuggingFace],
            "modelscope" => vec![Source::ModelScope],
            _ if huggingface_reachable() => vec![Source::HuggingFace, Source::ModelScope],
            _ => vec![Source::ModelScope, Source::HuggingFace],
        }
    }

    pub fn state(&self, id: &str) -> Option<Value> {
        let d = self.active.lock().unwrap().get(id).cloned()?;
        let progress = if d.status == "done" {
            1.0
        } else {
            let total: u64 = d.files.iter().map(|f| f.1).sum();
            let have: u64 = match &d.dir {
                Some(dir) => d.files.iter().map(|(f, n)| {
                    let done = std::fs::metadata(dir.join(f)).map(|m| m.len()).unwrap_or(0);
                    let part = std::fs::metadata(dir.join(format!("{f}.part"))).map(|m| m.len()).unwrap_or(0);
                    done.max(part).min(*n)
                }).sum(),
                None => 0,
            };
            (have as f64 / total.max(1) as f64).min(0.99)
        };
        Some(json!({"status": d.status, "progress": progress, "error": d.error, "source": d.source.map(Source::label)}))
    }

    pub fn progress(&self, id: &str) -> f64 {
        if is_downloaded(&self.hub, id) {
            return 1.0;
        }
        self.state(id).and_then(|s| s["progress"].as_f64()).unwrap_or(0.0)
    }

    pub fn start(&self, id: &str) {
        if id.is_empty() {
            return;
        }
        {
            let mut a = self.active.lock().unwrap();
            if a.get(id).map(|d| d.status == "downloading").unwrap_or(false) {
                return;
            }
            a.insert(id.to_string(), Dl { status: "downloading", error: None, files: vec![], dir: None, source: None });
        }
        let this = self.clone();
        let id_owned = id.to_string();
        std::thread::spawn(move || {
            let id = id_owned;
            let r = this.fetch(&id);
            let mut a = this.active.lock().unwrap();
            let d = a.get_mut(&id).unwrap();
            match r {
                Ok(()) => d.status = "done",
                Err(e) => {
                    d.status = "error";
                    d.error = Some(e);
                }
            }
            drop(a);
            (this.on_change)(&id);
        });
        (self.on_change)(id);
    }

    fn fetch(&self, id: &str) -> Result<(), String> {
        let sources = self.sources();
        let mut errors = vec![];
        for src in &sources {
            match self.fetch_from(*src, id) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    eprintln!("[cz] download {id} from {}: {e}", src.label());
                    errors.push(format!("{}：{e}", src.label()));
                }
            }
        }
        let hint = if sources == [Source::HuggingFace] { "。在国内可以到「设置」把模型下载源换成 ModelScope" } else { "" };
        Err(format!("{}{hint}", errors.join("；")))
    }

    fn fetch_from(&self, src: Source, id: &str) -> Result<(), String> {
        let (sha, files) = src.listing(id)?;
        let repo = repo_dir(&self.hub, id);
        let dir = repo.join("snapshots").join(&sha);
        {
            let mut a = self.active.lock().unwrap();
            if let Some(d) = a.get_mut(id) {
                d.files = files.clone();
                d.dir = Some(dir.clone());
                d.source = Some(src);
            }
        }
        for (f, n) in &files {
            let dst = dir.join(f);
            if std::fs::metadata(&dst).map(|m| m.len() == *n).unwrap_or(false) {
                continue;
            }
            if let Some(p) = dst.parent() {
                std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
            let part = dir.join(format!("{f}.part"));
            let ok = Command::new("curl")
                .args(["-fsSL", "--retry", "3", "--connect-timeout", "8", "-C", "-", "-o"])
                .arg(&part)
                .arg(src.url(id, &sha, f))
                .status().map(|s| s.success()).unwrap_or(false);
            if !ok || std::fs::metadata(&part).map(|m| m.len() != *n).unwrap_or(true) {
                return Err(format!("{f} 下载失败"));
            }
            std::fs::rename(&part, &dst).map_err(|e| e.to_string())?;
        }
        std::fs::create_dir_all(repo.join("refs")).map_err(|e| e.to_string())?;
        std::fs::write(repo.join("refs/main"), &sha).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn catalog(&self, asr_sel: &str, llm_sel: &str) -> Value {
        let decorate = |items: &[Model], current: &str| -> Vec<Value> {
            items.iter().map(|m| {
                let facts: serde_json::Map<String, Value> = m.facts.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
                json!({
                    "id": m.id, "name": m.name, "blurb": m.blurb, "facts": facts, "size_gb": m.size_gb,
                    "recommended": m.recommended, "selected": m.id == current,
                    "downloaded": is_downloaded(&self.hub, m.id), "download": self.state(m.id),
                })
            }).collect()
        };
        json!({"asr": decorate(ASR, asr_sel), "llm": decorate(LLM, llm_sel)})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Network test: cargo test -p chengzhang -- --ignored download
    #[test]
    #[ignore]
    fn download_into_hf_layout() {
        let hub = std::env::temp_dir().join(format!("cz-hub-{}", std::process::id()));
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let tx = Mutex::new(tx);
        let d = Downloads::new(hub.clone(), Arc::new(move |id: &str| { let _ = tx.lock().unwrap().send(id.to_string()); }));
        let id = "mlx-community/silero-vad";
        d.start(id);
        loop {
            rx.recv_timeout(std::time::Duration::from_secs(120)).unwrap();
            let st = d.state(id).unwrap();
            if st["status"] != "downloading" {
                assert_eq!(st["status"], "done", "{st}");
                break;
            }
        }
        let repo = repo_dir(&hub, id);
        let sha = std::fs::read_to_string(repo.join("refs/main")).unwrap();
        let snap = repo.join("snapshots").join(sha.trim());
        assert!(snap.join("config.json").exists());
        assert!(std::fs::read_dir(&snap).unwrap().count() >= 2);
        std::fs::remove_dir_all(&hub).unwrap();
    }

    /// Network test: cargo test -p chengzhang -- --ignored modelscope
    #[test]
    #[ignore]
    fn download_from_modelscope_into_hf_layout() {
        let hub = std::env::temp_dir().join(format!("cz-hub-ms-{}", std::process::id()));
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let tx = Mutex::new(tx);
        let d = Downloads::new(hub.clone(), Arc::new(move |id: &str| { let _ = tx.lock().unwrap().send(id.to_string()); }));
        d.set_source("modelscope");
        let id = "mlx-community/silero-vad";
        d.start(id);
        let st = loop {
            rx.recv_timeout(std::time::Duration::from_secs(120)).unwrap();
            let st = d.state(id).unwrap();
            if st["status"] != "downloading" {
                break st;
            }
        };
        assert_eq!(st["status"], "done", "{st}");
        assert_eq!(st["source"], "ModelScope");
        let repo = repo_dir(&hub, id);
        let sha = std::fs::read_to_string(repo.join("refs/main")).unwrap();
        let snap = repo.join("snapshots").join(sha.trim());
        assert!(snap.join("config.json").exists());
        assert!(!snap.join("configuration.json").exists(), "ModelScope's own metadata file is skipped");
        let n = std::fs::read_dir(&snap).unwrap().filter(|e| !e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".part")).count();
        assert!(n >= 2, "{n} files");
        std::fs::remove_dir_all(&hub).unwrap();
    }
}
