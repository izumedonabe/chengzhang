//! OS integration behind one portable API. The engine and UI only use what is
//! exported here; `macos.rs` implements it today, `windows.rs` is the place for
//! the Windows version (low-level keyboard hook, SendInput, clipboard).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as imp;

#[cfg(not(target_os = "macos"))]
mod windows;
#[cfg(not(target_os = "macos"))]
use windows as imp;

// ----------------------------------------------------------------------------- hotkey spec

/// Same JSON shape as the Python app's settings:
///   {"kind": "modifier", "keycode": 61}                  a lone modifier key (right ⌥ …)
///   {"kind": "key", "keycode": 49, "mods": ["control"]}  a key, optionally with modifiers
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hotkey {
    #[serde(default = "default_kind")]
    pub kind: String,
    pub keycode: u16,
    #[serde(default)]
    pub mods: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

fn default_kind() -> String {
    "key".into()
}

impl Default for Hotkey {
    fn default() -> Self {
        Hotkey { kind: "modifier".into(), keycode: 61, mods: vec![], name: None }
    }
}

impl Hotkey {
    /// Settings from before hotkeys were configurable ("right_option", "right_command", "fn").
    pub fn from_legacy(name: &str) -> Hotkey {
        let keycode = imp::legacy_keycode(name).unwrap_or(61);
        Hotkey { kind: "modifier".into(), keycode, mods: vec![], name: None }
    }
    /// Drop unknown modifier names.
    pub fn sanitized(mut self) -> Hotkey {
        self.mods.retain(|m| matches!(m.as_str(), "control" | "option" | "shift" | "command"));
        if self.kind != "modifier" {
            self.kind = "key".into();
        }
        self
    }
    pub fn label(&self) -> String {
        imp::hotkey_label(self)
    }
    pub fn presets() -> Vec<Hotkey> {
        imp::PRESET_KEYCODES
            .iter()
            .map(|&k| Hotkey { kind: "modifier".into(), keycode: k, mods: vec![], name: None })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotkeyEvent {
    /// pressed; `shift` = Shift was held (translate mode) and isn't part of the hotkey
    Down { shift: bool },
    Up,
    /// Esc while a dictation is active
    Cancel,
    /// the correction hotkey: the user fixed the last dictation in place, learn from it
    Learn,
}

pub type HotkeyCallback = Arc<dyn Fn(HotkeyEvent) + Send + Sync>;

/// A live global hotkey. Callbacks run on the main thread: only enqueue work there.
pub struct HotkeyHandle(imp::HotkeyTap);

impl HotkeyHandle {
    /// Must be called on the main thread (the tap is attached to the main run loop).
    /// `learn`: the correction hotkey (a key with modifiers), if any.
    /// Returns the handle and how it was installed ("tap" = full, "listen" = can't swallow keys).
    pub fn install(spec: Hotkey, learn: Option<Hotkey>, cb: HotkeyCallback) -> Option<(HotkeyHandle, &'static str)> {
        imp::HotkeyTap::install(spec, learn, cb).map(|(t, how)| (HotkeyHandle(t), how))
    }
    /// A dictation is running (Esc cancels it).
    pub fn set_active(&self, on: bool) {
        self.0.set_active(on)
    }
    /// The settings page is recording a new hotkey: don't start dictation meanwhile.
    pub fn set_paused(&self, on: bool) {
        self.0.set_paused(on)
    }
}

// ----------------------------------------------------------------------------- text output

/// Put text on the clipboard, send paste to the frontmost app, then restore the previous
/// clipboard text. Without the permission the OS drops synthetic keystrokes, so then we only
/// copy and return false. Call on the main thread.
pub fn paste_text(text: &str) -> bool {
    imp::paste_text(text)
}

pub fn copy_text(text: &str) {
    imp::copy_text(text)
}

/// The text the user is editing in the frontmost app: the selection if there is one,
/// else the whole focused text field (via Accessibility); for apps that don't expose
/// their text fields, the selection copied with a synthetic ⌘C (clipboard restored).
/// Returns the text and where it came from ("selection", "field", "copy").
/// Call OFF the main thread: the ⌘C path waits for the app to answer.
pub fn grab_text() -> Option<(String, &'static str)> {
    imp::grab_text()
}

// ----------------------------------------------------------------------------- permissions & misc

/// macOS: Accessibility (needed for the hotkey tap and for pasting).
pub fn accessibility_trusted(prompt: bool) -> bool {
    imp::accessibility_trusted(prompt)
}

/// "Accessibility" or "Microphone".
pub fn open_privacy_pane(which: &str) {
    imp::open_privacy_pane(which)
}

pub fn frontmost_app() -> String {
    imp::frontmost_app()
}

/// Show a floating window without activating our app, so the user's cursor stays in
/// the app they are typing in (otherwise the paste would land in our window).
/// `ns_window` is the native window handle (NSWindow* on macOS, HWND on Windows).
pub fn show_without_focus(window: *mut std::ffi::c_void) {
    imp::show_without_focus(window)
}

pub fn hide_window(window: *mut std::ffi::c_void) {
    imp::hide_window(window)
}

/// Whether the system has Liquid Glass (macOS 26+).
pub fn has_glass() -> bool {
    imp::has_glass()
}

/// System glass (macOS Liquid Glass) behind a window's page at `frame` = (x, y from the top,
/// width, height) with rounded corners. Returns false where there is none.
pub fn add_glass(window: *mut std::ffi::c_void, frame: (f64, f64, f64, f64), radius: f64) -> bool {
    imp::add_glass(window, frame, radius)
}

/// Main-window chrome: content under a transparent title bar with an inset unified toolbar.
pub fn style_main_window(window: *mut std::ffi::c_void) {
    imp::style_main_window(window)
}

/// Visible frame of the main screen: (x, y-from-top, width, height) in points.
pub fn main_screen_frame() -> (f64, f64, f64, f64) {
    imp::main_screen_frame()
}

/// System appearance is dark.
pub fn system_is_dark() -> bool {
    imp::system_is_dark()
}

/// Replace the app icon at runtime (macOS Dock). Main thread only.
pub fn set_app_icon(path: &str) {
    imp::set_app_icon(path)
}

/// Open a URL in the default browser. Main thread on macOS.
pub fn open_url(url: &str) -> bool {
    imp::open_url(url)
}
