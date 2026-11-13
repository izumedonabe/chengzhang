//! 成章 engine core: platform independent. Inference is behind `backend` traits,
//! OS integration (hotkey, paste, permissions) lives in cz-platform.
pub mod backend;
pub mod engine;
pub mod prompts;
pub mod resample;
pub mod text;
pub mod vad;
pub mod wav;

pub use engine::{Command, Config, Engine, Event, Outcome};
