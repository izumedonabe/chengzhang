//! Cuts the live stream at pauses so ASR can run while the user is still speaking
//! (port of engine.py's Segmenter, with Earshot instead of Silero).

pub const SR: usize = 16000;
pub const FRAME: usize = 256; // Earshot window: 16 ms at 16 kHz

#[derive(Clone, Debug)]
pub struct VadConfig {
    pub threshold: f32,
    pub cut_silence_ms: u32,
    /// once a segment is long, accept shorter pauses so the tail left after release stays short
    pub soft_cut_after_s: f32,
    pub soft_cut_silence_ms: u32,
    pub min_segment_s: f32,
    pub max_segment_s: f32,
    /// segments with less voiced audio than this are noise, not speech: never transcribed
    pub min_voiced_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            threshold: 0.5,
            cut_silence_ms: 300,
            soft_cut_after_s: 4.0,
            soft_cut_silence_ms: 150,
            min_segment_s: 1.5,
            max_segment_s: 25.0,
            min_voiced_ms: 200,
        }
    }
}

pub struct Segmenter {
    pub cfg: VadConfig,
    vad: Box<earshot::Detector>,
    audio: Vec<f32>,
    vad_pos: usize,            // samples fed to VAD
    cut: usize,                // start of the current (not yet transcribed) segment
    first_speech: Option<usize>, // first speech frame since the last cut
    pub silence_frames: usize,
    voiced_frames: usize, // speech frames since the last cut
}

impl Segmenter {
    pub fn new(cfg: VadConfig) -> Self {
        Segmenter {
            cfg,
            vad: earshot::Detector::default_boxed(),
            audio: Vec::with_capacity(SR * 30),
            vad_pos: 0,
            cut: 0,
            first_speech: None,
            silence_frames: 0,
            voiced_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.vad.reset();
        self.audio.clear();
        self.vad_pos = 0;
        self.cut = 0;
        self.first_speech = None;
        self.silence_frames = 0;
        self.voiced_frames = 0;
    }

    pub fn total(&self) -> usize {
        self.audio.len()
    }
    pub fn audio(&self, start: usize, end: usize) -> &[f32] {
        &self.audio[start.min(self.audio.len())..end.min(self.audio.len())]
    }
    pub fn all_audio(&self) -> &[f32] {
        &self.audio
    }
    /// Speech seen since the last cut (something left to transcribe).
    pub fn pending_speech(&self) -> bool {
        self.first_speech.is_some()
    }
    pub fn silence_ms(&self) -> f32 {
        (self.silence_frames * FRAME) as f32 * 1000.0 / SR as f32
    }

    fn seg_start(&self) -> usize {
        let pad = SR / 10;
        self.cut.max(self.first_speech.unwrap_or(self.cut).saturating_sub(pad))
    }

    /// Add samples; returns finished segments as (start, end) sample ranges.
    pub fn push(&mut self, x: &[f32]) -> Vec<(usize, usize)> {
        self.audio.extend_from_slice(x);
        let mut segs = Vec::new();
        let mut frame = [0f32; FRAME];
        while self.audio.len() - self.vad_pos >= FRAME {
            for (d, s) in frame.iter_mut().zip(&self.audio[self.vad_pos..self.vad_pos + FRAME]) {
                *d = s.clamp(-1.0, 1.0);
            }
            let p = self.vad.predict_f32(&frame);
            let start = self.vad_pos;
            self.vad_pos += FRAME;
            if p >= self.cfg.threshold {
                if self.first_speech.is_none() {
                    self.first_speech = Some(start);
                }
                self.silence_frames = 0;
                self.voiced_frames += 1;
            } else {
                self.silence_frames += 1;
            }
            if self.first_speech.is_none() {
                continue;
            }
            let seg_len = (self.vad_pos - self.seg_start()) as f32 / SR as f32;
            let need_ms = if seg_len >= self.cfg.soft_cut_after_s {
                self.cfg.soft_cut_silence_ms
            } else {
                self.cfg.cut_silence_ms
            } as f32;
            let cut_at = if self.silence_ms() >= need_ms && seg_len >= self.cfg.min_segment_s {
                self.vad_pos - self.silence_frames * FRAME / 2
            } else if seg_len >= self.cfg.max_segment_s {
                self.vad_pos
            } else {
                continue;
            };
            if self.voiced_enough() {
                segs.push((self.seg_start(), cut_at));
            }
            self.cut = cut_at;
            self.first_speech = None;
            self.voiced_frames = 0;
        }
        segs
    }

    fn voiced_enough(&self) -> bool {
        (self.voiced_frames * FRAME) as f32 * 1000.0 / SR as f32 >= self.cfg.min_voiced_ms as f32
    }

    /// The last, unfinished segment (if it contains enough speech).
    pub fn finish(&self) -> Option<(usize, usize)> {
        self.first_speech.filter(|_| self.voiced_enough()).map(|_| (self.seg_start(), self.audio.len()))
    }
}
