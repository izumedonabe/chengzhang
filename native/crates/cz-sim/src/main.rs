//! cz-sim a.wav b.wav …  [--no-llm] [--fast] [--release-early-ms N] [--dict 词1,词2] [--translate 英文]
//! Streams each file into the engine at real-time speed (32 ms blocks), "releases the
//! key" at the end of the file, and prints the transcript, the polished text and timings.
use cz_core::backend::{AsrSpec, LlmSpec};
use cz_core::engine::{Command, Config, Engine, Event};
use cz_core::vad::SR;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let mut files = vec![];
    let (mut no_llm, mut fast, mut early_ms, mut dict, mut translate) = (false, false, 0usize, vec![], None);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-llm" => no_llm = true,
            "--fast" => fast = true,
            "--release-early-ms" => early_ms = args.next().unwrap().parse().unwrap(),
            "--dict" => dict = args.next().unwrap().split(',').map(String::from).collect(),
            "--translate" => translate = args.next(),
            _ => files.push(PathBuf::from(a)),
        }
    }
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Application Support/Chengzhang/gguf");
    if let Ok(slots) = std::env::var("CZ_GEN_TEST") {
        gen_test(&dir, slots.parse().unwrap());
        return;
    }
    // CZ_BACKEND=llama: llama.cpp + GGUF (the Windows path); default on macOS: MLX
    let use_llama = !cfg!(target_os = "macos") || std::env::var("CZ_BACKEND").map(|b| b == "llama").unwrap_or(false);
    let hf = |repo: &str| -> PathBuf {
        let d = PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
            .join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
        std::fs::read_dir(&d).ok().and_then(|mut r| r.next()).and_then(|x| x.ok()).map(|x| x.path()).unwrap_or(d)
    };
    let (asr, llm) = if use_llama {
        (AsrSpec { model: dir.join("Qwen3-ASR-0.6B-Q8_0.gguf"), mmproj: dir.join("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf") },
         (!no_llm).then(|| LlmSpec { model: dir.join("Qwen3.5-4B-Q4_K_M.gguf"), n_ctx: 2560 }))
    } else {
        (AsrSpec { model: hf("mlx-community/Qwen3-ASR-0.6B-8bit"), mmproj: PathBuf::new() },
         (!no_llm).then(|| LlmSpec { model: hf("mlx-community/Qwen3.5-4B-MLX-4bit"), n_ctx: 0 }))
    };
    let backend: Arc<dyn cz_core::backend::Backend> = if use_llama {
        Arc::new(cz_llama::LlamaBackend)
    } else {
        #[cfg(target_os = "macos")]
        { Arc::new(cz_mlx::backend::MlxBackend) }
        #[cfg(not(target_os = "macos"))]
        { unreachable!() }
    };
    let mut cfg = Config::new(asr, llm);
    cfg.dictionary = dict;
    cfg.translate_lang = translate.clone();

    let (tx, rx) = channel();
    let t_load = Instant::now();
    let engine = Engine::spawn(cfg, backend, Box::new(move |e| {
        let _ = tx.send(e);
    }));
    loop {
        match rx.recv().unwrap() {
            Event::Ready { load_s, .. } => {
                eprintln!("loaded in {load_s:.2}s (wall {:.2}s)", t_load.elapsed().as_secs_f32());
                break;
            }
            Event::Error(e) => panic!("{e}"),
            _ => {}
        }
    }

    for f in &files {
        let (pcm, rate) = cz_core::wav::read(&std::fs::read(f).unwrap()).unwrap();
        assert_eq!(rate as usize, SR, "{}: expected 16 kHz", f.display());
        let block = SR * 32 / 1000;
        let release_at = pcm.len().saturating_sub(early_ms * SR / 1000);
        engine.send(Command::Start(translate.as_ref().map(|_| translate.clone().unwrap())));
        let t0 = Instant::now();
        let mut sent = 0;
        let mut released = false;
        let mut first_text = None;
        let outcome = loop {
            // keep streaming (including post-roll after release) until the engine is done
            if sent < pcm.len() + SR {
                let end = (sent + block).min(pcm.len() + SR);
                let chunk: Vec<f32> = (sent..end).map(|i| pcm.get(i).copied().unwrap_or(0.0)).collect();
                if !fast {
                    let due = t0 + Duration::from_secs_f64(end as f64 / SR as f64);
                    if let Some(d) = due.checked_duration_since(Instant::now()) {
                        std::thread::sleep(d);
                    }
                }
                engine.send(Command::Audio(chunk));
                sent = end;
                if !released && sent >= release_at {
                    released = true;
                    engine.send(Command::End(Instant::now()));
                }
            }
            let mut done = None;
            while let Ok(e) = rx.try_recv() {
                match e {
                    Event::Text(_) if first_text.is_none() => first_text = Some(Instant::now()),
                    Event::Done(o) => done = Some(o),
                    Event::Error(e) => panic!("{e}"),
                    _ => {}
                }
            }
            if let Some(o) = done {
                break o;
            }
            if sent >= pcm.len() + SR {
                engine.send(Command::Flush);
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let name = f.file_stem().unwrap().to_string_lossy();
        let segs: Vec<String> = outcome.segments.iter().map(|s| format!("{:.2}-{:.2}s asr {:.0}ms prefill {:.0}ms",
            s.start_s, s.end_s, s.asr_s * 1e3, s.llm_prefill_s * 1e3)).collect();
        println!("== {name}  audio {:.1}s  post-roll {:.0}ms  segments {}", outcome.audio_s, outcome.post_roll_s * 1e3, segs.len());
        for s in segs {
            println!("   {s}");
        }
        println!("   raw : {}", outcome.raw);
        println!("   text: {}", outcome.text.replace('\n', "⏎"));
        let llm = outcome.llm.as_ref();
        println!(
            "   release→raw {:.0}ms  →first char {}  →done {:.0}ms  | llm ttft {} prompt-after-release {} tok, {} tok @ {:.1} tok/s{}",
            outcome.release_to_raw_s * 1e3,
            outcome.release_to_first_char_s.map(|s| format!("{:.0}ms", s * 1e3)).unwrap_or("-".into()),
            outcome.release_to_done_s * 1e3,
            llm.and_then(|l| l.ttft_s).map(|s| format!("{:.0}ms", s * 1e3)).unwrap_or("-".into()),
            llm.map(|l| l.prompt_tokens_after_release).unwrap_or(0),
            llm.map(|l| l.gen_tokens).unwrap_or(0),
            llm.map(|l| l.gen_tps).unwrap_or(0.0),
            if outcome.fallback_to_raw { "  (fallback to raw)" } else { "" },
        );
    }
}

/// Raw LLM decode speed with `slots` prefix slots (CZ_GEN_TEST=<slots>).
fn gen_test(dir: &std::path::Path, slots: usize) {
    use cz_core::backend::{Backend, LlmSpec};
    let b = cz_llama::LlamaBackend;
    let mut llm = b.load_llm(&LlmSpec { model: dir.join("Qwen3.5-4B-Q4_K_M.gguf"), n_ctx: 3072 }, slots).unwrap();
    for s in 0..slots {
        llm.set_prefix(s, &cz_core::prompts::prefix(&[], &[], None)).unwrap();
    }
    for round in 0..3 {
        llm.begin(0).unwrap();
        llm.feed("请用三百字介绍一下杭州西湖的历史和景点，不要分点", false).unwrap();
        llm.feed(cz_core::prompts::SUFFIX, true).unwrap();
        let t = Instant::now();
        let mut first = None;
        let n = llm.generate(200, &mut |_| { first.get_or_insert(Instant::now()); true }).unwrap();
        let f = first.unwrap();
        let tps = (n - 1) as f32 / (Instant::now() - f).as_secs_f32();
        println!("slots {slots} round {round}: ttft {:.0}ms, {n} tok, {tps:.1} tok/s after first", (f - t).as_secs_f32() * 1e3);
        llm.end();
    }
}
