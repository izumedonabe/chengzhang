//! engine_check <asr model dir> <dictionary.txt|-> a.wav b.wav …
//! Feed each recording through the real engine (VAD segmentation + dictionary context,
//! no polishing) in 32 ms blocks, the way the app does, and print the segments and raw text.
//! WHOLE=1 also transcribes each file in one piece for comparison.
use cz_core::backend::{AsrSpec, Backend};
use cz_core::engine::{Command, Config, Engine, Event};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = std::path::PathBuf::from(&a[1]);
    let dictionary: Vec<String> = if a[2] == "-" {
        vec![]
    } else {
        std::fs::read_to_string(&a[2]).unwrap().lines().map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#')).map(String::from).collect()
    };
    let mut cfg = Config::new(AsrSpec { model: model.clone(), mmproj: model.clone() }, None);
    cfg.dictionary = dictionary;
    let (tx, rx) = channel();
    let backend: Arc<dyn Backend> = Arc::new(cz_mlx::backend::MlxBackend);
    let eng = Engine::spawn(cfg, backend, Box::new(move |e| {
        let _ = tx.send(e);
    }));
    loop {
        match rx.recv().unwrap() {
            Event::Ready { .. } => break,
            Event::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
    let whole = std::env::var("WHOLE").is_ok().then(|| cz_mlx::asr::Qwen3Asr::load(&model).unwrap());
    for f in &a[3..] {
        let (pcm, _) = cz_core::wav::read(&std::fs::read(f).unwrap()).unwrap();
        eng.send(Command::Start(Some(String::new())));
        for block in pcm.chunks(512) {
            eng.send(Command::Audio(block.to_vec()));
        }
        eng.send(Command::End(Instant::now()));
        eng.send(Command::Flush);
        let name = f.rsplit('/').next().unwrap();
        println!("== {name} [{:.1}s]", pcm.len() as f32 / 16000.0);
        loop {
            match rx.recv_timeout(Duration::from_secs(60)).unwrap() {
                Event::Segment(s) => println!("   seg {:5.1}-{:5.1}s  {}", s.start_s, s.end_s, s.text),
                Event::Done(o) => {
                    println!("   raw: {}", o.raw);
                    break;
                }
                Event::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
        if let Some(m) = &whole {
            println!("   whole: {}", m.transcribe(&pcm, "", Some("Chinese"), 400).unwrap());
        }
    }
}
