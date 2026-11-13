//! Qwen3-ASR on MLX — port of mlx-audio's qwen3_asr.py (+ the Whisper log-mel front end).
//! Audio encoder: 3 strided conv2d + 18 transformer layers with windowed attention.
//! Decoder: Qwen3 (0.6B: 28 layers), weights 8-bit affine; audio embeddings are spliced
//! between <|audio_start|> and <|audio_end|>.
use crate::qwen35::{e, scalar, R};
use crate::tokenizer::Tokenizer;
use crate::weights::{QLinear, Weights};
use mlx_rs::fast;
use mlx_rs::ops::{self, indexing::argmax_axis};
use mlx_rs::{Array, Dtype};
use std::path::Path;

const SR: usize = 16000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;

// ----------------------------------------------------------------------------- log-mel

/// Slaney mel scale (librosa / HF `mel_scale="slaney"`).
fn hz_to_mel(f: f64) -> f64 {
    let logstep = 27.0 / 6.4f64.ln();
    if f >= 1000.0 { 15.0 + (f / 1000.0).ln() * logstep } else { 3.0 * f / 200.0 }
}
fn mel_to_hz(m: f64) -> f64 {
    let logstep = 6.4f64.ln() / 27.0;
    if m >= 15.0 { 1000.0 * (logstep * (m - 15.0)).exp() } else { 200.0 * m / 3.0 }
}

/// HF `mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney")`, [201, 128].
fn mel_filters() -> Vec<f32> {
    let nb = N_FFT / 2 + 1;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
    let filt: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
    let fft: Vec<f64> = (0..nb).map(|i| (SR / 2) as f64 * i as f64 / (nb - 1) as f64).collect();
    let mut out = vec![0f32; nb * N_MELS];
    for (b, &f) in fft.iter().enumerate() {
        for m in 0..N_MELS {
            let down = -(filt[m] - f) / (filt[m + 1] - filt[m]);
            let up = (filt[m + 2] - f) / (filt[m + 2] - filt[m + 1]);
            let enorm = 2.0 / (filt[m + 2] - filt[m]);
            out[b * N_MELS + m] = (0f64.max(down.min(up)) * enorm) as f32;
        }
    }
    out
}

pub struct MelFrontEnd {
    cos: Array,     // [400, 201]
    sin: Array,     // [400, 201]
    filters: Array, // [201, 128]
    window: Vec<f32>,
}

impl MelFrontEnd {
    pub fn new() -> MelFrontEnd {
        let nb = N_FFT / 2 + 1;
        let mut c = vec![0f32; N_FFT * nb];
        let mut s = vec![0f32; N_FFT * nb];
        for n in 0..N_FFT {
            for k in 0..nb {
                let a = 2.0 * std::f64::consts::PI * (n * k) as f64 / N_FFT as f64;
                c[n * nb + k] = a.cos() as f32;
                s[n * nb + k] = a.sin() as f32;
            }
        }
        // periodic Hann window
        let window = (0..N_FFT).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32).collect();
        MelFrontEnd {
            cos: Array::from_slice(&c, &[N_FFT as i32, nb as i32]),
            sin: Array::from_slice(&s, &[N_FFT as i32, nb as i32]),
            filters: Array::from_slice(&mel_filters(), &[nb as i32, N_MELS as i32]),
            window,
        }
    }

    /// Whisper log-mel features, [128, frames] f32 with frames = len / 160.
    pub fn features(&self, pcm: &[f32]) -> R<Array> {
        let pad = N_FFT / 2;
        let n = pcm.len();
        // reflect padding (numpy "reflect": the edge sample is not repeated)
        let at = |i: isize| -> f32 {
            let mut j = i;
            if j < 0 { j = -j; }
            if j >= n as isize { j = 2 * (n as isize - 1) - j; }
            pcm[j.clamp(0, n as isize - 1) as usize]
        };
        let frames = 1 + n / HOP;
        let mut buf = vec![0f32; frames * N_FFT];
        for f in 0..frames {
            for k in 0..N_FFT {
                buf[f * N_FFT + k] = at((f * HOP + k) as isize - pad as isize) * self.window[k];
            }
        }
        let x = Array::from_slice(&buf, &[frames as i32, N_FFT as i32]);
        let re = x.matmul(&self.cos).map_err(e)?;
        let im = x.matmul(&self.sin).map_err(e)?;
        let power = re.square().map_err(e)?.add(im.square().map_err(e)?).map_err(e)?;
        let mel = ops::maximum(power.matmul(&self.filters).map_err(e)?, Array::from_f32(1e-10)).map_err(e)?;
        let log = ops::log10(&mel).map_err(e)?.t();
        // drop the last frame, clamp to (max - 8), scale
        let log = log.split_at_indices(&[(frames - 1) as i32], 1).map_err(e)?.swap_remove(0);
        let floor = log.max(None).map_err(e)?.subtract(Array::from_f32(8.0)).map_err(e)?;
        ops::maximum(&log, &floor).map_err(e)?
            .add(Array::from_f32(4.0)).map_err(e)?
            .divide(Array::from_f32(4.0)).map_err(e)
    }
}

// ----------------------------------------------------------------------------- audio encoder

struct EncLayer {
    ln1: (Array, Array),
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    ln2: (Array, Array),
    fc1: QLinear,
    fc2: QLinear,
}

struct Encoder {
    convs: Vec<(Array, Array)>,
    conv_out: QLinear,
    pos: Array, // [1500, d]
    layers: Vec<EncLayer>,
    ln_post: (Array, Array),
    proj1: QLinear,
    proj2: QLinear,
    heads: i32,
    n_window: i32,
    n_window_infer: i32,
}

fn after_cnn(len: i32) -> i32 {
    let fl = |a: i32, b: i32| a.div_euclid(b);
    let leave = len % 100;
    let feat = fl(leave - 1, 2) + 1;
    fl(fl(feat - 1, 2) + 1 - 1, 2) + 1 + (len / 100) * 13
}

fn layer_norm(x: &Array, p: &(Array, Array)) -> R<Array> {
    fast::layer_norm(x, Some(&p.0), Some(&p.1), 1e-5).map_err(e)
}

impl Encoder {
    fn load(w: &mut Weights, cfg: &serde_json::Value) -> R<Encoder> {
        let a = &cfg["audio_config"];
        let d = a["d_model"].as_i64().unwrap_or(896) as i32;
        let mut ln = |w: &mut Weights, p: &str| -> R<(Array, Array)> { Ok((w.take(&format!("{p}.weight"))?, w.take(&format!("{p}.bias"))?)) };
        let mut convs = vec![];
        for i in 1..=3 {
            convs.push((w.take(&format!("audio_tower.conv2d{i}.weight"))?, w.take(&format!("audio_tower.conv2d{i}.bias"))?));
        }
        let n_layers = a["encoder_layers"].as_i64().unwrap_or(18);
        let mut layers = vec![];
        for i in 0..n_layers {
            let p = format!("audio_tower.layers.{i}");
            layers.push(EncLayer {
                ln1: ln(w, &format!("{p}.self_attn_layer_norm"))?,
                q: w.qlinear(&format!("{p}.self_attn.q_proj"))?,
                k: w.qlinear(&format!("{p}.self_attn.k_proj"))?,
                v: w.qlinear(&format!("{p}.self_attn.v_proj"))?,
                o: w.qlinear(&format!("{p}.self_attn.out_proj"))?,
                ln2: ln(w, &format!("{p}.final_layer_norm"))?,
                fc1: w.qlinear(&format!("{p}.fc1"))?,
                fc2: w.qlinear(&format!("{p}.fc2"))?,
            });
        }
        // sinusoidal positions, f32 [max_source_positions, d]
        let len = a["max_source_positions"].as_i64().unwrap_or(1500) as usize;
        let half = (d / 2) as usize;
        let inc = 10000f64.ln() / (half - 1) as f64;
        let mut pos = vec![0f32; len * d as usize];
        for t in 0..len {
            for j in 0..half {
                let st = (t as f32) * (-(inc as f32) * j as f32).exp();
                pos[t * d as usize + j] = st.sin();
                pos[t * d as usize + half + j] = st.cos();
            }
        }
        Ok(Encoder {
            convs,
            conv_out: w.qlinear("audio_tower.conv_out")?,
            pos: Array::from_slice(&pos, &[len as i32, d]),
            layers,
            ln_post: ln(w, "audio_tower.ln_post")?,
            proj1: w.qlinear("audio_tower.proj1")?,
            proj2: w.qlinear("audio_tower.proj2")?,
            heads: a["encoder_attention_heads"].as_i64().unwrap_or(14) as i32,
            n_window: a["n_window"].as_i64().unwrap_or(50) as i32,
            n_window_infer: a["n_window_infer"].as_i64().unwrap_or(800) as i32,
        })
    }

    /// [128, frames] features -> [tokens, output_dim] audio embeddings.
    fn forward(&self, feats: &Array) -> R<Array> {
        let frames = feats.shape()[1];
        let chunk = self.n_window * 2;
        let n_chunks = (frames + chunk - 1) / chunk;
        let lens: Vec<i32> = (0..n_chunks).map(|j| if j == n_chunks - 1 && frames % chunk != 0 { frames % chunk } else { chunk }).collect();
        let max_len = *lens.iter().max().unwrap();
        // split into chunks along time, pad the last one with zeros
        let cuts: Vec<i32> = (1..n_chunks).map(|j| j * chunk).collect();
        let parts = if cuts.is_empty() { vec![feats.clone()] } else { feats.split_at_indices(&cuts, 1).map_err(e)? };
        let mut padded = vec![];
        for (p, &l) in parts.iter().zip(&lens) {
            let p = if l < max_len {
                ops::concatenate(&[p, &ops::zeros_dtype(&[N_MELS as i32, max_len - l], p.dtype()).map_err(e)?], 1).map_err(e)?
            } else { p.clone() };
            padded.push(p);
        }
        let mut x = ops::stack(&padded, 0).map_err(e)?.expand_dims(-1).map_err(e)?; // [n, 128, T, 1]
        for (w, b) in &self.convs {
            x = mlx_rs::nn::gelu(ops::conv2d(&x, w, (2, 2), (1, 1), (1, 1), 1).map_err(e)?.add(b).map_err(e)?).map_err(e)?;
        }
        let (n, f, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let x = x.transpose_axes(&[0, 2, 3, 1]).map_err(e)?.reshape(&[n, t, c * f]).map_err(e)?;
        let x = self.conv_out.fwd(&x)?;
        let pos = self.pos.split_at_indices(&[t], 0).map_err(e)?.swap_remove(0);
        let x = x.add(pos.expand_dims(0).map_err(e)?).map_err(e)?;
        // keep each chunk's valid positions
        let valid: Vec<i32> = lens.iter().map(|&l| after_cnn(l)).collect();
        let mut hs = vec![];
        for (i, &v) in valid.iter().enumerate() {
            let xi = x.split_at_indices(&[i as i32, i as i32 + 1], 0).map_err(e)?.swap_remove(1).squeeze_axes(&[0]).map_err(e)?;
            hs.push(if v < t { xi.split_at_indices(&[v], 0).map_err(e)?.swap_remove(0) } else { xi });
        }
        let mut h = ops::concatenate(&hs, 0).map_err(e)?;
        // windowed attention: blocks of `window` tokens attend only within themselves
        let total = after_cnn(frames);
        let max_cnn = *valid.iter().max().unwrap();
        let window = max_cnn * (self.n_window_infer / chunk);
        let seq = h.shape()[0];
        let mask = if total > window {
            let mut m = vec![-1e9f32; (seq * seq) as usize];
            let mut start = 0;
            while start < seq {
                let end = (start + window).min(seq);
                for r in start..end {
                    for c in start..end {
                        m[(r * seq + c) as usize] = 0.0;
                    }
                }
                start = end;
            }
            Some(Array::from_slice(&m, &[1, 1, seq, seq]).as_dtype(h.dtype()).map_err(e)?)
        } else {
            None
        };
        h = h.expand_dims(0).map_err(e)?;
        let d = h.shape()[2];
        let hd = d / self.heads;
        let scale = (hd as f32).powf(-0.5);
        for l in &self.layers {
            let x = layer_norm(&h, &l.ln1)?;
            let split = |a: Array| a.reshape(&[1, seq, self.heads, hd]).and_then(|a| a.transpose_axes(&[0, 2, 1, 3]));
            let q = split(l.q.fwd(&x)?.multiply(scalar(scale, h.dtype())?).map_err(e)?).map_err(e)?;
            let k = split(l.k.fwd(&x)?).map_err(e)?;
            let v = split(l.v.fwd(&x)?).map_err(e)?;
            let a = match &mask {
                Some(m) => fast::scaled_dot_product_attention(&q, &k, &v, 1.0, m, None),
                None => fast::scaled_dot_product_attention(&q, &k, &v, 1.0, None, None),
            }.map_err(e)?;
            let a = a.transpose_axes(&[0, 2, 1, 3]).map_err(e)?.reshape(&[1, seq, d]).map_err(e)?;
            h = h.add(l.o.fwd(&a)?).map_err(e)?;
            let x = layer_norm(&h, &l.ln2)?;
            let x = l.fc2.fwd(&mlx_rs::nn::gelu(l.fc1.fwd(&x)?).map_err(e)?)?;
            h = h.add(&x).map_err(e)?;
        }
        let h = layer_norm(&h.squeeze_axes(&[0]).map_err(e)?, &self.ln_post)?;
        let h = mlx_rs::nn::gelu(self.proj1.fwd(&h)?).map_err(e)?;
        self.proj2.fwd(&h)
    }
}

// ----------------------------------------------------------------------------- Qwen3 decoder

struct DecLayer {
    in_norm: Array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Array,
    k_norm: Array,
    post_norm: Array,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

struct Decoder {
    embed: QLinear,
    layers: Vec<DecLayer>,
    norm: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
}

type Kv = Vec<(Option<Array>, Option<Array>)>;

impl Decoder {
    fn load(w: &mut Weights, cfg: &serde_json::Value) -> R<Decoder> {
        let t = &cfg["text_config"];
        let n = t["num_hidden_layers"].as_i64().unwrap_or(28);
        let mut layers = vec![];
        for i in 0..n {
            let p = format!("model.layers.{i}");
            layers.push(DecLayer {
                in_norm: w.take(&format!("{p}.input_layernorm.weight"))?,
                q: w.qlinear(&format!("{p}.self_attn.q_proj"))?,
                k: w.qlinear(&format!("{p}.self_attn.k_proj"))?,
                v: w.qlinear(&format!("{p}.self_attn.v_proj"))?,
                o: w.qlinear(&format!("{p}.self_attn.o_proj"))?,
                q_norm: w.take(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: w.take(&format!("{p}.self_attn.k_norm.weight"))?,
                post_norm: w.take(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: w.qlinear(&format!("{p}.mlp.gate_proj"))?,
                up: w.qlinear(&format!("{p}.mlp.up_proj"))?,
                down: w.qlinear(&format!("{p}.mlp.down_proj"))?,
            });
        }
        Ok(Decoder {
            embed: w.qlinear("model.embed_tokens")?,
            layers,
            norm: w.take("model.norm.weight")?,
            heads: t["num_attention_heads"].as_i64().unwrap_or(16) as i32,
            kv_heads: t["num_key_value_heads"].as_i64().unwrap_or(8) as i32,
            head_dim: t["head_dim"].as_i64().unwrap_or(128) as i32,
            theta: t["rope_theta"].as_f64().unwrap_or(1e6) as f32,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        })
    }

    fn embed_ids(&self, ids: &[u32]) -> R<Array> {
        let a = Array::from_slice(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>(), &[1, ids.len() as i32]);
        self.embed.embed(&a)
    }

    /// [1, L, D] embeddings -> final hidden [1, L, D]; extends `kv` at `offset`.
    fn forward(&self, x: &Array, kv: &mut Kv, offset: i32) -> R<Array> {
        let mut h = x.clone();
        let l = x.shape()[1];
        let scale = (self.head_dim as f32).powf(-0.5);
        for (layer, c) in self.layers.iter().zip(kv.iter_mut()) {
            let x = fast::rms_norm(&h, Some(&layer.in_norm), self.eps).map_err(e)?;
            let q = layer.q.fwd(&x)?.reshape(&[1, l, self.heads, self.head_dim]).map_err(e)?;
            let k = layer.k.fwd(&x)?.reshape(&[1, l, self.kv_heads, self.head_dim]).map_err(e)?;
            let v = layer.v.fwd(&x)?.reshape(&[1, l, self.kv_heads, self.head_dim]).map_err(e)?.transpose_axes(&[0, 2, 1, 3]).map_err(e)?;
            let q = fast::rms_norm(&q, Some(&layer.q_norm), self.eps).map_err(e)?.transpose_axes(&[0, 2, 1, 3]).map_err(e)?;
            let k = fast::rms_norm(&k, Some(&layer.k_norm), self.eps).map_err(e)?.transpose_axes(&[0, 2, 1, 3]).map_err(e)?;
            let q = fast::rope(&q, self.head_dim, false, self.theta, 1.0, offset, None).map_err(e)?;
            let k = fast::rope(&k, self.head_dim, false, self.theta, 1.0, offset, None).map_err(e)?;
            let k = match c.0.take() { Some(k0) => ops::concatenate(&[&k0, &k], 2).map_err(e)?, None => k };
            let v = match c.1.take() { Some(v0) => ops::concatenate(&[&v0, &v], 2).map_err(e)?, None => v };
            c.0 = Some(k.clone());
            c.1 = Some(v.clone());
            let a = if l > 1 {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, fast::ScaledDotProductAttentionMask::Causal, None)
            } else {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)
            }.map_err(e)?;
            let a = a.transpose_axes(&[0, 2, 1, 3]).map_err(e)?.reshape(&[1, l, -1]).map_err(e)?;
            h = h.add(layer.o.fwd(&a)?).map_err(e)?;
            let x = fast::rms_norm(&h, Some(&layer.post_norm), self.eps).map_err(e)?;
            let m = mlx_rs::nn::silu(layer.gate.fwd(&x)?).map_err(e)?.multiply(layer.up.fwd(&x)?).map_err(e)?;
            h = h.add(layer.down.fwd(&m)?).map_err(e)?;
        }
        fast::rms_norm(&h, Some(&self.norm), self.eps).map_err(e)
    }

    fn next_token(&self, hidden: &Array) -> R<Array> {
        let l = hidden.shape()[1];
        let last = if l > 1 { hidden.split_at_indices(&[l - 1], 1).map_err(e)?.pop().unwrap() } else { hidden.clone() };
        argmax_axis(&self.embed.fwd(&last)?, -1, false).map_err(e)
    }
}

// ----------------------------------------------------------------------------- model

pub struct Qwen3Asr {
    tok: Tokenizer,
    mel: MelFrontEnd,
    enc: Encoder,
    dec: Decoder,
    eos: Vec<u32>,
}

impl Qwen3Asr {
    pub fn load(dir: &Path) -> R<Qwen3Asr> {
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(e)?).map_err(e)?;
        let cfg = if cfg.get("thinker_config").is_some() { cfg["thinker_config"].clone() } else { cfg };
        let mut w = Weights::load(dir, "")?;
        let enc = Encoder::load(&mut w, &cfg)?;
        let dec = Decoder::load(&mut w, &cfg)?;
        let tok = Tokenizer::load(dir)?;
        let eos = ["<|endoftext|>", "<|im_end|>"].iter().filter_map(|s| tok.token_id(s)).collect();
        Ok(Qwen3Asr { tok, mel: MelFrontEnd::new(), enc, dec, eos })
    }

    /// Transcribe 16 kHz mono PCM. `context`: biasing text (dictionary words);
    /// `language`: e.g. "Chinese", or None to auto-detect.
    pub fn transcribe(&self, pcm: &[f32], context: &str, language: Option<&str>, max_tokens: usize) -> R<String> {
        if pcm.len() < HOP * 2 {
            return Ok(String::new());
        }
        let feats = self.mel.features(pcm)?;
        let audio = self.enc.forward(&feats)?;
        let sys = if context.is_empty() { String::new() } else { format!("{context}\n") };
        let before = self.tok.encode(&format!("<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n<|audio_start|>"), true);
        let prefix = language.map(|l| format!("language {l}<asr_text>")).unwrap_or_default();
        let after = self.tok.encode(&format!("<|audio_end|><|im_end|>\n<|im_start|>assistant\n{prefix}"), true);
        let e_before = self.dec.embed_ids(&before)?;
        let e_after = self.dec.embed_ids(&after)?;
        let audio = audio.as_dtype(e_before.dtype()).map_err(e)?.expand_dims(0).map_err(e)?;
        let x = ops::concatenate(&[&e_before, &audio, &e_after], 1).map_err(e)?;
        let mut kv: Kv = vec![(None, None); self.dec.layers.len()];
        let mut offset = 0;
        let h = self.dec.forward(&x, &mut kv, offset)?;
        offset += x.shape()[1];
        let mut y = self.dec.next_token(&h)?;
        mlx_rs::transforms::async_eval([&y]).map_err(e)?;
        let mut out: Vec<u32> = vec![];
        for _ in 0..max_tokens {
            let emb = self.dec.embed.embed(&y.reshape(&[1, 1]).map_err(e)?)?;
            let h = self.dec.forward(&emb, &mut kv, offset)?;
            offset += 1;
            let ny = self.dec.next_token(&h)?;
            mlx_rs::transforms::async_eval([&ny]).map_err(e)?;
            let id = y.item_cast::<u32>();
            if self.eos.contains(&id) {
                break;
            }
            out.push(id);
            y = ny;
        }
        let text = self.tok.decode(&out);
        let text = match (language, text.find("<asr_text>")) {
            (None, Some(p)) => text[p + "<asr_text>".len()..].to_string(),
            _ => text,
        };
        Ok(text.trim().to_string())
    }
}
