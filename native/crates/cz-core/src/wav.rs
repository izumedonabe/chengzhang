//! Minimal WAV I/O (16-bit PCM or 32-bit float, any channel count) — enough for
//! saved recordings and the simulator; no dependency.

pub fn read(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let u16le = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let u32le = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut fmt, mut channels, mut rate, mut bits) = (0u16, 1u16, 16000u32, 16u16);
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32le(&bytes[i + 4..]) as usize;
        let body = &bytes[i + 8..(i + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            fmt = u16le(body);
            channels = u16le(&body[2..]);
            rate = u32le(&body[4..]);
            bits = u16le(&body[14..]);
            if fmt == 0xFFFE && body.len() >= 26 {
                fmt = u16le(&body[24..]); // WAVE_FORMAT_EXTENSIBLE sub-format
            }
        } else if id == b"data" {
            let ch = channels.max(1) as usize;
            let samples: Vec<f32> = match (fmt, bits) {
                (1, 16) => body.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect(),
                (3, 32) => body.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                _ => return Err(format!("unsupported WAV format {fmt}/{bits}")),
            };
            let mono = samples.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
            return Ok((mono, rate));
        }
        i += 8 + len + (len & 1);
    }
    Err("WAV has no data".into())
}

pub fn write_pcm16(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + data_len as usize);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        v.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    v
}
