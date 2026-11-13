// size/speed probe: a GPU matmul through MLX
use mlx_rs::Array;
fn main() {
    let a = mlx_rs::random::normal::<f32>(&[1024, 1024], None, None, None).unwrap();
    let t = std::time::Instant::now();
    let b = a.matmul(&a).unwrap();
    b.eval().unwrap();
    let s: Array = b.sum(None).unwrap();
    println!("sum {:?} in {:?}", s.item::<f32>(), t.elapsed());
}
