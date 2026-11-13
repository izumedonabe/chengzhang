//! tok_check <model dir> <text>... — print token ids (to diff against Python).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = cz_mlx::tokenizer::Tokenizer::load(std::path::Path::new(&a[1])).unwrap();
    for s in &a[2..] {
        println!("{:?}", t.encode(s, true));
    }
}
