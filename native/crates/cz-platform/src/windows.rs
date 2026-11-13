//! Windows (and other) placeholder. To implement for Windows:
//!   hotkey  — SetWindowsHookExW(WH_KEYBOARD_LL) on a thread with a message loop;
//!             modifier keycodes map to VK_RMENU / VK_RCONTROL …
//!   paste   — clipboard via OpenClipboard/SetClipboardData(CF_UNICODETEXT), then
//!             SendInput Ctrl+V; restore the previous text afterwards
//!   HUD     — WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TRANSPARENT, ShowWindow(SW_SHOWNOACTIVATE)
//!   permissions — none needed (accessibility_trusted() = true)
use crate::{Hotkey, HotkeyCallback};
use std::ffi::c_void;

pub const PRESET_KEYCODES: &[u16] = &[];

pub fn legacy_keycode(_name: &str) -> Option<u16> {
    None
}

pub fn hotkey_label(h: &Hotkey) -> String {
    format!("键 {}", h.keycode)
}

pub struct HotkeyTap;

impl HotkeyTap {
    pub fn install(_spec: Hotkey, _cb: HotkeyCallback) -> Option<(HotkeyTap, &'static str)> {
        None
    }
    pub fn set_active(&self, _on: bool) {}
    pub fn set_paused(&self, _on: bool) {}
}

pub fn copy_text(_text: &str) {}
pub fn paste_text(_text: &str) -> bool {
    false
}
pub fn accessibility_trusted(_prompt: bool) -> bool {
    true
}
pub fn open_privacy_pane(_which: &str) {}
pub fn frontmost_app() -> String {
    String::new()
}
pub fn show_without_focus(_w: *mut c_void) {}
pub fn hide_window(_w: *mut c_void) {}
pub fn style_main_window(_w: *mut c_void) {}
pub fn main_screen_frame() -> (f64, f64, f64, f64) {
    (0.0, 0.0, 1920.0, 1080.0)
}
pub fn system_is_dark() -> bool {
    false
}
pub fn set_app_icon(_path: &str) {}
pub fn open_url(_url: &str) -> bool {
    false // TODO(Windows): ShellExecuteW(NULL, "open", url, …)
}
