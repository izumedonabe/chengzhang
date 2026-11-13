//! llm_check <mlx model dir> "<raw dictation>"  — greedy-polish one text, print tokens/s.
use cz_mlx::{qwen35::*, tokenizer::Tokenizer};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let t0 = Instant::now();
    let tok = Tokenizer::load(&dir).unwrap();
    let t_tok = t0.elapsed();
    let model = Qwen35::load(&dir).unwrap();
    if std::env::var("WIRED").is_ok() {
        let old = mlx_rs::memory::set_wired_limit(19_000_000_000).unwrap();
        eprintln!("wired limit set (was {old})");
    }
    eprintln!("tokenizer {:?}, model graph {:?}", t_tok, t0.elapsed());
    for (round, text) in std::iter::once(&a[2]).chain(a[2..].iter()).enumerate() {
    let prompt = format!("{}{}{}", cz_core::prompts::prefix(&[], None), text, cz_core::prompts::SUFFIX);
    let ids = tok.encode(&prompt, true);
    println!("prompt tokens: {} first {:?} last {:?}", ids.len(), &ids[..6], &ids[ids.len() - 6..]);
        let mut cache = model.make_cache();
        let t = Instant::now();
        let mut h = None;
        for chunk in ids.chunks(512) {
            let x = model.forward(chunk, &mut cache).unwrap();
            eval_cache(&cache).unwrap();
            h = Some(x);
        }
        let mut y = model.argmax_last(&h.unwrap()).unwrap();
        mlx_rs::transforms::async_eval([&y]).unwrap();
        if std::env::var("PROF").is_ok() {
            let mut c2 = cache.clone();
            let yy = y.reshape(&[1, 1]).unwrap();
            mlx_rs::transforms::eval([&yy]).unwrap();
            let (mut tb, mut te) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
            for _ in 0..20 {
                let t1 = Instant::now();
                let hh = model.forward_ids(&yy, &mut c2).unwrap();
                let ny = model.argmax_last(&hh).unwrap();
                tb += t1.elapsed();
                let t2 = Instant::now();
                mlx_rs::transforms::eval([&ny]).unwrap();
                te += t2.elapsed();
            }
            eprintln!("per token: graph build {:?}, eval {:?}", tb / 20, te / 20);
        }
        let mut out = vec![];
        let mut t_first = None;
        for _ in 0..200 {
            let hh = model.forward_ids(&y.reshape(&[1, 1]).unwrap(), &mut cache).unwrap();
            let ny = model.argmax_last(&hh).unwrap();
            mlx_rs::transforms::async_eval([&ny]).unwrap();
            let id = y.item_cast::<u32>();
            t_first.get_or_insert(t.elapsed());
            if model.cfg.eos.contains(&id) { break; }
            out.push(id);
            y = ny;
        }
        let total = t.elapsed();
        let gen = total - t_first.unwrap();
        println!("round {round}: prefill+first {:?}, {} tok in {:?} = {:.1} tok/s", t_first.unwrap(), out.len(), gen,
                 out.len() as f64 / gen.as_secs_f64());
        println!("{}", tok.decode(&out));
    }
}
