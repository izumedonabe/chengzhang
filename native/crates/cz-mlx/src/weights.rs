//! safetensors loading and 4-bit affine quantized linear layers (mlx-community format).
use mlx_rs::ops::{dequantize, quantized_matmul, DequantizeOptions, QuantizationMode, QuantizedMatmulOptions};
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

pub struct Weights {
    map: HashMap<String, Array>,
    prefix: String,
    pub group: i32,
    pub bits: i32,
}

impl Weights {
    /// Load all *.safetensors in `dir` (lazily: data is read when first used).
    /// Keys are looked up as `prefix + name`.
    pub fn load(dir: &Path, prefix: &str) -> Result<Weights, String> {
        let mut map = HashMap::new();
        let mut files: Vec<_> = std::fs::read_dir(dir).map_err(e)?
            .filter_map(|x| x.ok().map(|x| x.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        for f in files {
            map.extend(Array::load_safetensors(&f).map_err(e)?);
        }
        let cfg: serde_json::Value = std::fs::read_to_string(dir.join("config.json")).ok()
            .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        let q = &cfg["quantization"];
        Ok(Weights {
            map,
            prefix: prefix.to_string(),
            group: q["group_size"].as_i64().unwrap_or(64) as i32,
            bits: q["bits"].as_i64().unwrap_or(4) as i32,
        })
    }

    pub fn take(&mut self, name: &str) -> Result<Array, String> {
        let key = format!("{}{name}", self.prefix);
        self.map.remove(&key).ok_or_else(|| format!("missing weight {key}"))
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

    /// Drop everything not taken (e.g. a vision tower we don't use).
    pub fn rest(self) -> Vec<String> {
        self.map.into_keys().collect()
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
            None => x.matmul(self.w.t()).map_err(e)?,
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
