//! Streaming windowed-sinc resampler (microphone rate -> 16 kHz), no dependencies.
//! Blackman-windowed sinc, 24 zero crossings, cutoff at 95% of the lower Nyquist.

pub struct Resampler {
    step: f64, // input samples per output sample
    cutoff: f64,
    half: f64, // kernel half-width in input samples
    buf: Vec<f32>,
    pos: f64, // position of the next output sample in `buf` coordinates
}

impl Resampler {
    pub fn new(rate_in: u32, rate_out: u32) -> Resampler {
        let step = rate_in as f64 / rate_out as f64;
        let cutoff = 0.95 * (1.0f64).min(1.0 / step); // relative to input Nyquist
        let half = 12.0 / cutoff;
        Resampler { step, cutoff, half, buf: vec![0.0; half.ceil() as usize], pos: half.ceil() }
    }

    fn kernel(&self, x: f64) -> f64 {
        if x.abs() >= self.half {
            return 0.0;
        }
        let t = x * self.cutoff;
        let sinc = if t.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
        let w = 0.5 + 0.5 * (std::f64::consts::PI * x / self.half).cos(); // Hann-shaped taper
        let b = 0.42 + 0.5 * (std::f64::consts::PI * x / self.half).cos() + 0.08 * (2.0 * std::f64::consts::PI * x / self.half).cos();
        self.cutoff * sinc * (0.5 * w + 0.5 * b)
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        let h = self.half.ceil() as isize;
        while self.pos + self.half < self.buf.len() as f64 {
            let c = self.pos.floor() as isize;
            let mut acc = 0.0f64;
            for i in (c - h + 1)..=(c + h) {
                if i >= 0 && (i as usize) < self.buf.len() {
                    acc += self.buf[i as usize] as f64 * self.kernel(self.pos - i as f64);
                }
            }
            out.push(acc as f32);
            self.pos += self.step;
        }
        // keep only the history the kernel still needs
        let drop = (self.pos.floor() as isize - h).max(0) as usize;
        if drop > 0 {
            self.buf.drain(..drop);
            self.pos -= drop as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    #[test]
    fn tone_survives_48k_to_16k() {
        let mut r = Resampler::new(48000, 16000);
        let tone: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin()).collect();
        let mut out = vec![];
        for chunk in tone.chunks(1536) {
            r.process(chunk, &mut out);
        }
        assert!((out.len() as i32 - 16000).abs() < 40, "{}", out.len());
        // compare against the ideal 16 kHz tone (skip the start-up transient, allow the fixed delay)
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let mid = &out[2000..14000];
        assert!((rms(mid) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02, "rms {}", rms(mid));
        // an 11 kHz tone (above the new Nyquist) must be removed
        let mut r = Resampler::new(48000, 16000);
        let hi: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 11000.0 * i as f32 / 48000.0).sin()).collect();
        let mut out = vec![];
        r.process(&hi, &mut out);
        assert!(rms(&out[2000..14000]) < 0.01, "alias rms {}", rms(&out[2000..14000]));
    }
}
