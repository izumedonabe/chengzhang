//! asr_check <model dir> a.wav b.wav … — transcribe and time each file (twice).
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = Instant::now();
    let m = cz_mlx::asr::Qwen3Asr::load(std::path::Path::new(&a[1])).unwrap();
    eprintln!("load {:?}", t.elapsed());
    for f in &a[2..] {
        let (pcm, _) = cz_core::wav::read(&std::fs::read(f).unwrap()).unwrap();
        for round in 0..2 {
            if let Ok(ms) = std::env::var("IDLE_MS") { std::thread::sleep(std::time::Duration::from_millis(ms.parse().unwrap())); }
            let t = Instant::now();
            let s = m.transcribe(&pcm, "", Some("Chinese"), 400).unwrap();
            println!("{} [{:.1}s audio] round {round}: {:.0} ms  {}", f.rsplit('/').next().unwrap(), pcm.len() as f32 / 16000.0, t.elapsed().as_secs_f64() * 1e3, s);
        }
    }
}
