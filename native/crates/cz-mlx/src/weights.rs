//! safetensors loading and 4-bit affine quantized linear layers (mlx-community format).
use mlx_rs::ops::{dequantize, quantized_matmul, DequantizeOptions, QuantizationMode, QuantizedMatmulOptions};
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::Path;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

/// A read-only memory map of one safetensors file.
struct Mapped {
    ptr: *mut u8,
    len: usize,
}

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

impl Mapped {
    fn open(path: &Path) -> Result<Mapped, String> {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path).map_err(|x| format!("{}: {x}", path.display()))?;
        let len = f.metadata().map_err(e)?.len() as usize;
        // PROT_READ = 1, MAP_PRIVATE = 2
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, f.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(format!("mmap failed: {}", path.display()));
        }
        Ok(Mapped { ptr, len })
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        unsafe { munmap(self.ptr, self.len) };
    }
}

struct Entry {
    file: usize,
    dtype: Dtype,
    shape: Vec<i32>,
    start: usize, // absolute byte offset in the file
    end: usize,
}

/// safetensors reader. MLX's own loader runs on the CPU stream, which our CPU-less MLX
/// build doesn't have; this copies just the tensors that are taken (a vision tower in
/// the same file is never touched) into GPU-visible MLX buffers.
pub struct Weights {
    files: Vec<Mapped>,
    map: HashMap<String, Entry>,
    prefix: String,
    pub group: i32,
    pub bits: i32,
}

fn dtype_of(s: &str) -> Result<Dtype, String> {
    Ok(match s {
        "BF16" => Dtype::Bfloat16,
        "F16" => Dtype::Float16,
        "F32" => Dtype::Float32,
        "U32" => Dtype::Uint32,
        "I32" => Dtype::Int32,
        "U8" => Dtype::Uint8,
        "I8" => Dtype::Int8,
        "I64" => Dtype::Int64,
        "U16" => Dtype::Uint16,
        "I16" => Dtype::Int16,
        "BOOL" => Dtype::Bool,
        _ => return Err(format!("unsupported dtype {s}")),
    })
}

impl Weights {
    /// Index all *.safetensors in `dir`. Keys are looked up as `prefix + name`.
    pub fn load(dir: &Path, prefix: &str) -> Result<Weights, String> {
        let mut map = HashMap::new();
        let mut paths: Vec<_> = std::fs::read_dir(dir).map_err(e)?
            .filter_map(|x| x.ok().map(|x| x.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        let mut files = vec![];
        for p in paths {
            let m = Mapped::open(&p)?;
            let b = m.bytes();
            if b.len() < 8 {
                return Err(format!("{}: truncated", p.display()));
            }
            let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).map_err(e)?;
            let base = 8 + n;
            for (k, v) in header.as_object().ok_or("bad safetensors header")? {
                if k == "__metadata__" {
                    continue;
                }
                let off = v["data_offsets"].as_array().ok_or("no offsets")?;
                let (s0, s1) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
                if base + s1 > b.len() {
                    return Err(format!("{}: tensor {k} out of bounds", p.display()));
                }
                map.insert(k.clone(), Entry {
                    file: files.len(),
                    dtype: dtype_of(v["dtype"].as_str().unwrap_or(""))?,
                    shape: v["shape"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0) as i32).collect()).unwrap_or_default(),
                    start: base + s0,
                    end: base + s1,
                });
            }
            files.push(m);
        }
        let cfg: serde_json::Value = std::fs::read_to_string(dir.join("config.json")).ok()
            .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        let q = &cfg["quantization"];
        Ok(Weights {
            files,
            map,
            prefix: prefix.to_string(),
            group: q["group_size"].as_i64().unwrap_or(64) as i32,
            bits: q["bits"].as_i64().unwrap_or(4) as i32,
        })
    }

    pub fn take(&mut self, name: &str) -> Result<Array, String> {
        let key = format!("{}{name}", self.prefix);
        let t = self.map.remove(&key).ok_or_else(|| format!("missing weight {key}"))?;
        let bytes = &self.files[t.file].bytes()[t.start..t.end];
        let shape = if t.shape.is_empty() { vec![] } else { t.shape.clone() };
        // copies into an MLX (Metal shared) buffer
        let a = unsafe { Array::from_raw_data(bytes.as_ptr() as *const std::ffi::c_void, &shape, t.dtype) };
        Ok(a)
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(&format!("{}{name}", self.prefix))
    }

    /// A linear layer: quantized (weight + scales + biases) or plain (weight [+ bias]).
    pub fn qlinear(&mut self, name: &str) -> Result<QLinear, String> {
        let w = self.take(&format!("{name}.weight"))?;
        let bias = if self.has(&format!("{name}.bias")) { Some(self.take(&format!("{name}.bias"))?) } else { None };
        if self.has(&format!("{name}.scales")) {
            let scales = self.take(&format!("{name}.scales"))?;
            let biases = self.take(&format!("{name}.biases"))?;
            // per-layer bit width: packed columns * 32 / bits = input dim = groups * group size
            let in_dim = scales.shape()[scales.ndim() - 1] * self.group;
            let bits = w.shape()[w.ndim() - 1] * 32 / in_dim;
            Ok(QLinear { w, q: Some((scales, biases)), bias, group: self.group, bits })
        } else {
            Ok(QLinear { w, q: None, bias, group: self.group, bits: self.bits })
        }
    }

}

pub struct QLinear {
    pub w: Array,
    pub q: Option<(Array, Array)>, // scales, biases
    pub bias: Option<Array>,
    pub group: i32,
    pub bits: i32,
}

impl QLinear {
    pub fn fwd(&self, x: &Array) -> Result<Array, String> {
        let y = match &self.q {
            Some((s, b)) => quantized_matmul(x, &self.w, s, Some(b), QuantizedMatmulOptions {
                mode: QuantizationMode::Affine,
                group_size: Some(self.group),
                bits: Some(self.bits),
                transpose: true,
            }).map_err(e)?,
            // like mlx.nn.Linear: addmm(bias, x, W^T)
            None => {
                return match &self.bias {
                    Some(b) => mlx_rs::ops::addmm(b, x, self.w.t(), None, None).map_err(e),
                    None => x.matmul(self.w.t()).map_err(e),
                }
            }
        };
        match &self.bias {
            Some(b) => y.add(b).map_err(e),
            None => Ok(y),
        }
    }

    /// Embedding lookup (rows of the weight matrix), dequantized when needed.
    pub fn embed(&self, ids: &Array) -> Result<Array, String> {
        let rows = |a: &Array| a.take_axis(ids, 0).map_err(e);
        match &self.q {
            Some((s, b)) => dequantize(rows(&self.w)?, rows(s)?, Some(&rows(b)?), DequantizeOptions {
                mode: QuantizationMode::Affine,
                group_size: Some(self.group),
                bits: Some(self.bits),
            }).map_err(e),
            None => rows(&self.w),
        }
    }
}
