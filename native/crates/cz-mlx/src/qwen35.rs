//! Qwen3.5 (text) on MLX — port of mlx-lm's qwen3_5.py / qwen3_next.py / gated_delta.py.
//! 3 of every 4 layers are Gated DeltaNet (linear attention with a recurrent state),
//! the 4th is gated full attention. Weights: mlx-community 4-bit affine checkpoints.
use crate::weights::{QLinear, Weights};
use mlx_rs::fast::{self, MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg};
use mlx_rs::ops::{self, indexing::argmax_axis};
use mlx_rs::{Array, Dtype};
use std::path::Path;

pub type R<T> = Result<T, String>;
pub fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

#[derive(Clone, Debug)]
pub struct Cfg {
    pub hidden: i32,
    pub layers: usize,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub eps: f32,
    pub v_heads: i32,
    pub k_heads: i32,
    pub k_dim: i32,
    pub v_dim: i32,
    pub conv_k: i32,
    pub interval: usize,
    pub eos: Vec<u32>,
}

impl Cfg {
    pub fn load(dir: &Path) -> R<Cfg> {
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(e)?).map_err(e)?;
        let t = if c.get("text_config").is_some() { &c["text_config"] } else { &c };
        let i = |k: &str| t[k].as_i64().unwrap_or(0) as i32;
        let rp = &t["rope_parameters"];
        let head_dim = i("head_dim");
        let prf = rp["partial_rotary_factor"].as_f64().unwrap_or(0.25);
        let mut eos: Vec<u32> = match &t["eos_token_id"] {
            serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect(),
            v => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        };
        eos.push(248046); // <|im_end|>
        Ok(Cfg {
            hidden: i("hidden_size"),
            layers: i("num_hidden_layers") as usize,
            heads: i("num_attention_heads"),
            kv_heads: i("num_key_value_heads"),
            head_dim,
            rope_dims: (head_dim as f64 * prf) as i32,
            rope_theta: rp["rope_theta"].as_f64().unwrap_or(1e7) as f32,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            v_heads: i("linear_num_value_heads"),
            k_heads: i("linear_num_key_heads"),
            k_dim: i("linear_key_head_dim"),
            v_dim: i("linear_value_head_dim"),
            conv_k: i("linear_conv_kernel_dim"),
            interval: i("full_attention_interval").max(1) as usize,
            eos,
        })
    }
}

// ----------------------------------------------------------------------------- cache

/// Per-layer state. Arrays are immutable and refcounted, so cloning a cache is a
/// cheap snapshot — that is how prompt prefixes are reused.
#[derive(Clone)]
pub enum LayerCache {
    Gdn { conv: Option<Array>, state: Option<Array> },
    Kv { k: Option<Array>, v: Option<Array>, offset: i32 },
}

pub type Cache = Vec<LayerCache>;

// ----------------------------------------------------------------------------- layers

struct Gdn {
    in_qkv: QLinear,
    in_z: QLinear,
    in_b: QLinear,
    in_a: QLinear,
    conv_w: Array,
    dt_bias: Array,
    neg_a: Array, // -exp(A_log), f32
    norm_w: Array,
    out: QLinear,
}

struct Attn {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Array,
    k_norm: Array,
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

struct Layer {
    mixer: Mixer,
    in_norm: Array,
    post_norm: Array,
    mlp: Mlp,
}

pub struct Qwen35 {
    pub cfg: Cfg,
    embed: QLinear,
    layers: Vec<Layer>,
    norm: Array,
    gdn_kernel: MetalKernel,
}

const GDN_SOURCE: &str = r#"
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
"#;

impl Qwen35 {
    pub fn load(dir: &Path) -> R<Qwen35> {
        let cfg = Cfg::load(dir)?;
        let mut w = Weights::load(dir, "language_model.model.")?;
        let mut layers = vec![];
        for i in 0..cfg.layers {
            let p = format!("layers.{i}.");
            let linear = (i + 1) % cfg.interval != 0;
            let mixer = if linear {
                let a = format!("{p}linear_attn.");
                let a_log = w.take(&format!("{a}A_log"))?.as_dtype(Dtype::Float32).map_err(e)?;
                Mixer::Gdn(Gdn {
                    in_qkv: w.qlinear(&format!("{a}in_proj_qkv"))?,
                    in_z: w.qlinear(&format!("{a}in_proj_z"))?,
                    in_b: w.qlinear(&format!("{a}in_proj_b"))?,
                    in_a: w.qlinear(&format!("{a}in_proj_a"))?,
                    conv_w: w.take(&format!("{a}conv1d.weight"))?,
                    dt_bias: w.take(&format!("{a}dt_bias"))?,
                    neg_a: ops::negative(&ops::exp(&a_log).map_err(e)?).map_err(e)?,
                    norm_w: w.take(&format!("{a}norm.weight"))?,
                    out: w.qlinear(&format!("{a}out_proj"))?,
                })
            } else {
                let a = format!("{p}self_attn.");
                Mixer::Attn(Attn {
                    q: w.qlinear(&format!("{a}q_proj"))?,
                    k: w.qlinear(&format!("{a}k_proj"))?,
                    v: w.qlinear(&format!("{a}v_proj"))?,
                    o: w.qlinear(&format!("{a}o_proj"))?,
                    q_norm: w.take(&format!("{a}q_norm.weight"))?,
                    k_norm: w.take(&format!("{a}k_norm.weight"))?,
                })
            };
            layers.push(Layer {
                mixer,
                in_norm: w.take(&format!("{p}input_layernorm.weight"))?,
                post_norm: w.take(&format!("{p}post_attention_layernorm.weight"))?,
                mlp: Mlp {
                    gate: w.qlinear(&format!("{p}mlp.gate_proj"))?,
                    up: w.qlinear(&format!("{p}mlp.up_proj"))?,
                    down: w.qlinear(&format!("{p}mlp.down_proj"))?,
                },
            });
        }
        let embed = w.qlinear("embed_tokens")?;
        let norm = w.take("norm.weight")?;
        let gdn_kernel = MetalKernel::new(
            "cz_gated_delta_step_packed",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GDN_SOURCE,
            MetalKernelOptions { ensure_row_contiguous: true, ..Default::default() },
        ).map_err(e)?;
        Ok(Qwen35 { cfg, embed, layers, norm, gdn_kernel })
    }

    pub fn make_cache(&self) -> Cache {
        (0..self.cfg.layers)
            .map(|i| if (i + 1) % self.cfg.interval != 0 {
                LayerCache::Gdn { conv: None, state: None }
            } else {
                LayerCache::Kv { k: None, v: None, offset: 0 }
            })
            .collect()
    }

    /// Run tokens through the model, updating `cache`. Returns the final hidden states
    /// (after the last norm) — call `logits` on the positions you need.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> R<Array> {
        let ids = Array::from_slice(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(), &[1, tokens.len() as i32]);
        self.forward_ids(&ids, cache)
    }

    /// Same as `forward`, with token ids already on the device ([1, L] int32/uint32).
    pub fn forward_ids(&self, ids: &Array, cache: &mut Cache) -> R<Array> {
        let mut h = self.embed.embed(ids)?;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            let x = fast::rms_norm(&h, Some(&layer.in_norm), self.cfg.eps).map_err(e)?;
            let r = match (&layer.mixer, c) {
                (Mixer::Gdn(g), LayerCache::Gdn { conv, state }) => self.gdn(g, &x, conv, state)?,
                (Mixer::Attn(a), LayerCache::Kv { k, v, offset }) => self.attn(a, &x, k, v, offset)?,
                _ => return Err("cache does not match the model".into()),
            };
            let h2 = h.add(&r).map_err(e)?;
            let x2 = fast::rms_norm(&h2, Some(&layer.post_norm), self.cfg.eps).map_err(e)?;
            let m = &layer.mlp;
            let act = mlx_rs::nn::silu(m.gate.fwd(&x2)?).map_err(e)?.multiply(m.up.fwd(&x2)?).map_err(e)?;
            h = h2.add(m.down.fwd(&act)?).map_err(e)?;
        }
        fast::rms_norm(&h, Some(&self.norm), self.cfg.eps).map_err(e)
    }

    /// Logits for the last position of `hidden` ([1, L, D]) — tied embedding as the head.
    pub fn logits_last(&self, hidden: &Array) -> R<Array> {
        let l = hidden.shape()[1];
        let last = if l > 1 { hidden.split_at_indices(&[l - 1], 1).map_err(e)?.pop().unwrap() } else { hidden.clone() };
        self.embed.fwd(&last)
    }

    pub fn argmax_last(&self, hidden: &Array) -> R<Array> {
        let logits = self.logits_last(hidden)?;
        argmax_axis(&logits, -1, false).map_err(e)
    }

    fn gdn(&self, g: &Gdn, x: &Array, conv: &mut Option<Array>, state: &mut Option<Array>) -> R<Array> {
        let c = &self.cfg;
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let key_dim = c.k_heads * c.k_dim;
        let value_dim = c.v_heads * c.v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv = g.in_qkv.fwd(x)?;
        let z = g.in_z.fwd(x)?.reshape(&[b, s, c.v_heads, c.v_dim]).map_err(e)?;
        let beta = ops::sigmoid(g.in_b.fwd(x)?).map_err(e)?;
        let a = g.in_a.fwd(x)?;

        let conv_state = match conv.take() {
            Some(cs) => cs,
            None => ops::zeros_dtype(&[b, c.conv_k - 1, conv_dim], x.dtype()).map_err(e)?,
        };
        let conv_input = ops::concatenate(&[&conv_state, &qkv], 1).map_err(e)?;
        let n = conv_input.shape()[1];
        let keep = conv_input.split_at_indices(&[n - (c.conv_k - 1)], 1).map_err(e)?.pop().unwrap();
        *conv = Some(keep.contiguous().map_err(e)?);
        let conv_out = mlx_rs::nn::silu(ops::conv1d(&conv_input, &g.conv_w, 1, 0, 1, conv_dim).map_err(e)?).map_err(e)?;

        let parts = conv_out.split_at_indices(&[key_dim, 2 * key_dim], -1).map_err(e)?;
        let q = parts[0].reshape(&[b, s, c.k_heads, c.k_dim]).map_err(e)?;
        let k = parts[1].reshape(&[b, s, c.k_heads, c.k_dim]).map_err(e)?;
        let v = parts[2].reshape(&[b, s, c.v_heads, c.v_dim]).map_err(e)?;
        // normalize_qk: L2-normalize q/k and fold the Dk^-0.5 readout scale into q
        let inv = (c.k_dim as f32).powf(-0.5);
        let rms_eps = 1e-6 * inv * inv;
        // scalars must carry the activation dtype: an f32 array would promote bf16 -> f32
        let dt = q.dtype();
        let q = fast::rms_norm(&q, None, rms_eps).map_err(e)?.multiply(scalar(inv * inv, dt)?).map_err(e)?;
        let k = fast::rms_norm(&k, None, rms_eps).map_err(e)?.multiply(scalar(inv, dt)?).map_err(e)?;

        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        let sp = mlx_rs::nn::softplus(a.add(&g.dt_bias).map_err(e)?).map_err(e)?;
        let gate = ops::exp(g.neg_a.multiply(&sp).map_err(e)?).map_err(e)?;

        let st = match state.take() {
            Some(st) => st,
            None => ops::zeros_dtype(&[b, c.v_heads, c.v_dim, c.k_dim], Dtype::Float32).map_err(e)?,
        };
        let outs = unsafe {
            self.gdn_kernel.apply(
                &[&q, &k, &v, &gate, &beta, &st, &Array::from_int(s)],
                &[
                    MetalKernelOutput { shape: vec![b, s, c.v_heads, c.v_dim], dtype: q.dtype() },
                    MetalKernelOutput { shape: st.shape().to_vec(), dtype: Dtype::Float32 },
                ],
                (32, c.v_dim / 8, b * c.v_heads),
                (32, 2, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("InT".into(), TemplateArg::Dtype(q.dtype())),
                        ("StT".into(), TemplateArg::Dtype(Dtype::Float32)),
                        ("Dk".into(), TemplateArg::Int(c.k_dim)),
                        ("Dv".into(), TemplateArg::Int(c.v_dim)),
                        ("Hk".into(), TemplateArg::Int(c.k_heads)),
                        ("Hv".into(), TemplateArg::Int(c.v_heads)),
                    ],
                    ..Default::default()
                },
            ).map_err(e)?
        };
        let (y, new_state) = (outs[0].clone(), outs[1].clone());
        *state = Some(new_state);

        // gated RMSNorm: silu(z) * rms_norm(y), in float32
        let normed = fast::rms_norm(&y, Some(&g.norm_w), c.eps).map_err(e)?;
        let zf = mlx_rs::nn::silu(z.as_dtype(Dtype::Float32).map_err(e)?).map_err(e)?;
        let out = zf.multiply(normed.as_dtype(Dtype::Float32).map_err(e)?).map_err(e)?.as_dtype(y.dtype()).map_err(e)?;
        g.out.fwd(&out.reshape(&[b, s, -1]).map_err(e)?)
    }

    fn attn(&self, a: &Attn, x: &Array, kc: &mut Option<Array>, vc: &mut Option<Array>, offset: &mut i32) -> R<Array> {
        let c = &self.cfg;
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let qg = a.q.fwd(x)?.reshape(&[b, l, c.heads, 2 * c.head_dim]).map_err(e)?;
        let parts = qg.split_at_indices(&[c.head_dim], -1).map_err(e)?;
        let (queries, gate) = (&parts[0], parts[1].reshape(&[b, l, -1]).map_err(e)?);
        let keys = a.k.fwd(x)?.reshape(&[b, l, c.kv_heads, c.head_dim]).map_err(e)?;
        let values = a.v.fwd(x)?.reshape(&[b, l, c.kv_heads, c.head_dim]).map_err(e)?.transpose_axes(&[0, 2, 1, 3]).map_err(e)?;
        let queries = fast::rms_norm(queries, Some(&a.q_norm), c.eps).map_err(e)?.transpose_axes(&[0, 2, 1, 3]).map_err(e)?;
        let keys = fast::rms_norm(&keys, Some(&a.k_norm), c.eps).map_err(e)?.transpose_axes(&[0, 2, 1, 3]).map_err(e)?;
        let queries = fast::rope(&queries, c.rope_dims, false, c.rope_theta, 1.0, *offset, None).map_err(e)?;
        let keys = fast::rope(&keys, c.rope_dims, false, c.rope_theta, 1.0, *offset, None).map_err(e)?;
        let keys = match kc.take() { Some(k0) => ops::concatenate(&[&k0, &keys], 2).map_err(e)?, None => keys };
        let values = match vc.take() { Some(v0) => ops::concatenate(&[&v0, &values], 2).map_err(e)?, None => values };
        *kc = Some(keys.clone());
        *vc = Some(values.clone());
        *offset += l;
        let scale = (c.head_dim as f32).powf(-0.5);
        let out = if l > 1 {
            fast::scaled_dot_product_attention(&queries, &keys, &values, scale, fast::ScaledDotProductAttentionMask::Causal, None)
        } else {
            fast::scaled_dot_product_attention(&queries, &keys, &values, scale, None, None)
        }.map_err(e)?;
        let out = out.transpose_axes(&[0, 2, 1, 3]).map_err(e)?.reshape(&[b, l, -1]).map_err(e)?;
        a.o.fwd(&out.multiply(ops::sigmoid(&gate).map_err(e)?).map_err(e)?)
    }
}

/// A 0-d array of `dtype` (MLX promotes array-array ops, unlike Python scalars).
pub fn scalar(x: f32, dtype: Dtype) -> R<Array> {
    Array::from_f32(x).as_dtype(dtype).map_err(e)
}

/// Evaluate everything a cache holds (materialize after a prefill chunk).
pub fn eval_cache(cache: &Cache) -> R<()> {
    let mut arrs: Vec<&Array> = vec![];
    for c in cache {
        match c {
            LayerCache::Gdn { conv, state } => arrs.extend(conv.iter().chain(state.iter())),
            LayerCache::Kv { k, v, .. } => arrs.extend(k.iter().chain(v.iter())),
        }
    }
    mlx_rs::transforms::eval(arrs).map_err(e)
}
