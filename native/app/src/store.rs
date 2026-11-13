//! Local persistence (port of store.py): ~/Library/Application Support/Chengzhang on
//! macOS, %APPDATA%\\Chengzhang on Windows.
//! History and the dictionary are shared with the Python app (same files, same format);
//! settings live in their own file because model ids differ (GGUF vs MLX).
use cz_core::text::norm;
use serde_json::{json, Map, Value};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::{DEFAULT_ASR, DEFAULT_LLM};
const TYPING_CPM: f64 = 50.0; // rough Chinese typing speed, for "time saved"

pub fn defaults() -> Map<String, Value> {
    let v = json!({
        "asr_model": DEFAULT_ASR,
        "llm_model": DEFAULT_LLM,             // "" = no polishing
        "hotkey": {"kind": "modifier", "keycode": 61},
        "hotkey_mode": "hold",                 // hold | toggle
        "translate_lang": "英文",
        "language": "Chinese",                 // ASR language; "auto" = detect
        "post_roll_ms": 300,
        "fast_path": true,
        "auto_paste": true,
        "appearance": "auto",
        "mic": "auto",
        "keep_mic_open": false,
        "save_audio": false,
        "unload_after_min": 0,
        "unload_what": "llm",
    });
    v.as_object().unwrap().clone()
}

pub fn app_dir() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
        PathBuf::from(base).join("Chengzhang")
    }
    #[cfg(not(windows))]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join("Library/Application Support/Chengzhang")
    }
}

pub struct Store {
    pub root: PathBuf,
    lock: Mutex<()>,
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl Store {
    pub fn new() -> Store {
        let root = app_dir();
        let _ = fs::create_dir_all(&root);
        let s = Store { root, lock: Mutex::new(()) };
        if !s.dict_path().exists() {
            let _ = fs::write(s.dict_path(), "# 个人词典：每行一个词，同时用于语音识别和润色。\n");
        }
        if !s.settings_path().exists() {
            // carry over preferences from the Python app; its model ids only where we know them
            if let Ok(Value::Object(mut old)) = fs::read_to_string(s.root.join("settings.json"))
                .map(|t| serde_json::from_str(&t).unwrap_or(Value::Null))
            {
                for k in ["asr_model", "llm_model"] {
                    let known = old.get(k).and_then(|v| v.as_str()).map(|id| crate::models::find(id).is_some()).unwrap_or(false);
                    if !known {
                        old.remove(k);
                    }
                }
                let _ = s.write_settings(&old);
            }
        }
        s
    }

    fn settings_path(&self) -> PathBuf {
        self.root.join("native-settings.json")
    }
    fn dict_path(&self) -> PathBuf {
        self.root.join("dictionary.txt")
    }
    fn history_path(&self) -> PathBuf {
        self.root.join("history.jsonl")
    }
    pub fn models_dir(&self) -> PathBuf {
        crate::models::default_hub()
    }

    // ------------------------------------------------------------------ settings

    pub fn settings(&self) -> Map<String, Value> {
        let mut s = defaults();
        if let Ok(Value::Object(o)) = fs::read_to_string(self.settings_path())
            .map(|t| serde_json::from_str(&t).unwrap_or(Value::Null))
        {
            for (k, v) in o {
                if s.contains_key(&k) {
                    s.insert(k, v);
                }
            }
        }
        s
    }

    fn write_settings(&self, s: &Map<String, Value>) -> std::io::Result<()> {
        fs::write(self.settings_path(), serde_json::to_string_pretty(s).unwrap())
    }

    pub fn update_setting(&self, key: &str, value: Value) -> Map<String, Value> {
        let _g = self.lock.lock().unwrap();
        let mut s = self.settings();
        if s.contains_key(key) {
            s.insert(key.to_string(), value);
            let _ = self.write_settings(&s);
        }
        s
    }

    // ------------------------------------------------------------------ dictionary

    pub fn dictionary(&self) -> Vec<String> {
        fs::read_to_string(self.dict_path())
            .unwrap_or_default()
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect()
    }

    pub fn set_dictionary(&self, words: &[String]) -> Vec<String> {
        let mut clean: Vec<String> = vec![];
        for w in words.iter().map(|w| w.trim()).filter(|w| !w.is_empty()) {
            if !clean.iter().any(|c| c == w) {
                clean.push(w.to_string());
            }
        }
        let body = format!("# 个人词典：每行一个词，同时用于语音识别和润色。\n{}\n", clean.join("\n"));
        let _ = fs::write(self.dict_path(), body);
        clean
    }

    // ------------------------------------------------------------------ history

    pub fn add(&self, mut entry: Map<String, Value>) -> Map<String, Value> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let id = format!("{:012x}", (nanos ^ ((std::process::id() as u128) << 40)) & 0xffff_ffff_ffff);
        entry.insert("id".into(), json!(id));
        entry.insert("ts".into(), json!(now()));
        let _g = self.lock.lock().unwrap();
        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(self.history_path()) {
            let _ = writeln!(f, "{}", Value::Object(entry.clone()));
        }
        entry
    }

    fn all(&self) -> Vec<Map<String, Value>> {
        fs::read_to_string(self.history_path())
            .unwrap_or_default()
            .lines()
            .filter_map(|l| match serde_json::from_str(l) {
                Ok(Value::Object(o)) => Some(o),
                _ => None,
            })
            .collect()
    }

    pub fn history(&self, limit: usize, offset: usize, query: &str) -> Vec<Map<String, Value>> {
        let q = query.to_lowercase();
        let field = |e: &Map<String, Value>, k: &str| e.get(k).and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
        self.all()
            .into_iter()
            .rev()
            .filter(|e| !e.get("deleted").and_then(|v| v.as_bool()).unwrap_or(false))
            .filter(|e| q.is_empty() || field(e, "text").contains(&q) || field(e, "raw").contains(&q))
            .skip(offset)
            .take(limit)
            .collect()
    }

    pub fn delete(&self, id: &str) {
        let _g = self.lock.lock().unwrap();
        let body: String = self
            .all()
            .into_iter()
            .filter(|e| e.get("id").and_then(|v| v.as_str()) != Some(id))
            .map(|e| Value::Object(e).to_string() + "\n")
            .collect();
        let _ = fs::write(self.history_path(), body);
    }

    pub fn clear(&self) {
        let _g = self.lock.lock().unwrap();
        let _ = fs::write(self.history_path(), "");
    }

    pub fn save_audio(&self, id: &str, audio: &[f32]) {
        let d = self.root.join("recordings");
        let _ = fs::create_dir_all(&d);
        let _ = fs::write(d.join(format!("{id}.wav")), cz_core::wav::write_pcm16(audio, 16000));
    }

    pub fn stats(&self) -> Value {
        let items = self.all();
        let today0 = local_midnight();
        let week0 = today0 - 6.0 * 86400.0;
        let ts = |e: &Map<String, Value>| e.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let agg = |es: Vec<&Map<String, Value>>| {
            // count what was said (the raw transcript), so translations don't count letters
            let chars: usize = es.iter().map(|e| {
                let s = e.get("raw").and_then(|v| v.as_str()).filter(|s| !s.is_empty())
                    .or_else(|| e.get("text").and_then(|v| v.as_str())).unwrap_or("");
                norm(s).chars().count()
            }).sum();
            let audio: f64 = es.iter().map(|e| e.get("audio_s").and_then(|v| v.as_f64()).unwrap_or(0.0)).sum();
            let done: Vec<f64> = es.iter().filter_map(|e| e.get("done_s").and_then(|v| v.as_f64())).collect();
            json!({
                "count": es.len(),
                "chars": chars,
                "audio_min": audio / 60.0,
                "cpm": if audio > 1.0 { chars as f64 / (audio / 60.0) } else { 0.0 },
                "avg_done_s": if done.is_empty() { Value::Null } else { json!(done.iter().sum::<f64>() / done.len() as f64) },
                "saved_min": (chars as f64 / TYPING_CPM - audio / 60.0).max(0.0),
            })
        };
        json!({
            "today": agg(items.iter().filter(|e| ts(e) >= today0).collect()),
            "week": agg(items.iter().filter(|e| ts(e) >= week0).collect()),
            "all": agg(items.iter().collect()),
        })
    }
}

/// Unix time of today's local midnight.
fn local_midnight() -> f64 {
    #[repr(C)]
    struct Tm {
        sec: i32, min: i32, hour: i32, mday: i32, mon: i32, year: i32, wday: i32, yday: i32, isdst: i32,
        gmtoff: i64, zone: *const i8,
    }
    extern "C" {
        fn localtime_r(t: *const i64, out: *mut Tm) -> *mut Tm;
        fn mktime(tm: *mut Tm) -> i64;
    }
    let t = now() as i64;
    unsafe {
        let mut tm: Tm = std::mem::zeroed();
        localtime_r(&t, &mut tm);
        tm.sec = 0;
        tm.min = 0;
        tm.hour = 0;
        tm.isdst = -1;
        mktime(&mut tm) as f64
    }
}
