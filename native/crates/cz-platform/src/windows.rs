//! Windows implementation (Win32 via windows-sys):
//!   hotkey  — low-level keyboard hook (WH_KEYBOARD_LL) on its own thread with a message
//!             loop. Lone-modifier hotkeys (right Alt …) are swallowed, otherwise releasing
//!             Alt would open the menu bar of the app you are typing into.
//!   paste   — CF_UNICODETEXT on the clipboard, SendInput Ctrl+V, previous text restored
//!   HUD     — WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW, ShowWindow(SW_SHOWNOACTIVATE)
//!   permissions — none needed
//! Hotkeys are stored with macOS keycodes (shared settings format) and mapped to VK codes here.
use crate::{Hotkey, HotkeyCallback, HotkeyEvent};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::DataExchange::*;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Memory::*;
use windows_sys::Win32::System::Registry::*;
use windows_sys::Win32::System::Threading::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::*;
use windows_sys::Win32::UI::Shell::ShellExecuteW;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const CF_UNICODETEXT: u32 = 13;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

// ----------------------------------------------------------------------------- key mapping

/// macOS keycode of a lone modifier -> (VK code, label)
fn modifier_vk(code: u16) -> Option<(u16, &'static str)> {
    Some(match code {
        61 => (VK_RMENU, "右 Alt"),
        58 => (VK_LMENU, "左 Alt"),
        62 => (VK_RCONTROL, "右 Ctrl"),
        59 => (VK_LCONTROL, "左 Ctrl"),
        60 => (VK_RSHIFT, "右 Shift"),
        56 => (VK_LSHIFT, "左 Shift"),
        54 => (VK_RWIN, "右 Win"),
        55 => (VK_LWIN, "左 Win"),
        _ => return None,
    })
}

/// macOS keycode of an ordinary key -> (VK code, label)
fn key_vk(code: u16) -> Option<(u16, String)> {
    const LETTERS: [(u16, char); 26] = [(0, 'A'), (11, 'B'), (8, 'C'), (2, 'D'), (14, 'E'), (3, 'F'), (5, 'G'), (4, 'H'),
        (34, 'I'), (38, 'J'), (40, 'K'), (37, 'L'), (46, 'M'), (45, 'N'), (31, 'O'), (35, 'P'), (12, 'Q'), (15, 'R'),
        (1, 'S'), (17, 'T'), (32, 'U'), (9, 'V'), (13, 'W'), (7, 'X'), (16, 'Y'), (6, 'Z')];
    const DIGITS: [(u16, char); 10] = [(29, '0'), (18, '1'), (19, '2'), (20, '3'), (21, '4'), (23, '5'), (22, '6'),
        (26, '7'), (28, '8'), (25, '9')];
    const FKEYS: [(u16, u16); 20] = [(122, 1), (120, 2), (99, 3), (118, 4), (96, 5), (97, 6), (98, 7), (100, 8), (101, 9),
        (109, 10), (103, 11), (111, 12), (105, 13), (107, 14), (113, 15), (106, 16), (64, 17), (79, 18), (80, 19), (90, 20)];
    if let Some((_, c)) = LETTERS.iter().chain(DIGITS.iter()).find(|(k, _)| *k == code) {
        return Some((*c as u16, c.to_string()));
    }
    if let Some((_, n)) = FKEYS.iter().find(|(k, _)| *k == code) {
        return Some((VK_F1 + (n - 1), format!("F{n}")));
    }
    Some(match code {
        49 => (VK_SPACE, "Space".into()),
        36 => (VK_RETURN, "Enter".into()),
        48 => (VK_TAB, "Tab".into()),
        51 => (VK_BACK, "Backspace".into()),
        53 => (VK_ESCAPE, "Esc".into()),
        50 => (VK_OEM_3, "`".into()),
        123 => (VK_LEFT, "←".into()),
        124 => (VK_RIGHT, "→".into()),
        125 => (VK_DOWN, "↓".into()),
        126 => (VK_UP, "↑".into()),
        _ => return None,
    })
}

pub const PRESET_KEYCODES: &[u16] = &[61, 62, 60, 54];

pub fn legacy_keycode(name: &str) -> Option<u16> {
    match name {
        "right_option" => Some(61),
        "right_command" => Some(54),
        _ => None,
    }
}

pub fn hotkey_label(h: &Hotkey) -> String {
    if h.kind == "modifier" {
        return modifier_vk(h.keycode).map(|m| m.1.to_string()).unwrap_or(format!("键 {}", h.keycode));
    }
    let mut parts: Vec<String> = vec![];
    for (m, name) in [("control", "Ctrl"), ("option", "Alt"), ("shift", "Shift"), ("command", "Win")] {
        if h.mods.iter().any(|x| x == m) {
            parts.push(name.into());
        }
    }
    parts.push(key_vk(h.keycode).map(|k| k.1).or_else(|| h.name.clone()).unwrap_or(format!("键 {}", h.keycode)));
    parts.join("+")
}

// ----------------------------------------------------------------------------- hotkey hook

struct HookState {
    spec: Hotkey,
    cb: HotkeyCallback,
    down: AtomicBool,
    active: AtomicBool,
    paused: AtomicBool,
}

static HOOK: Mutex<Option<Arc<HookState>>> = Mutex::new(None);

pub struct HotkeyTap {
    state: Arc<HookState>,
    thread: AtomicU32,
}

fn key_down(vk: u16) -> bool {
    unsafe { (GetAsyncKeyState(vk as i32) as u16 & 0x8000) != 0 }
}

unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code == HC_ACTION as i32 {
        let k = &*(lparam as *const KBDLLHOOKSTRUCT);
        let injected = k.flags & LLKHF_INJECTED != 0; // e.g. our own Ctrl+V
        let st = HOOK.lock().ok().and_then(|g| g.clone());
        if let (Some(st), false) = (st, injected) {
            let msg = wparam as u32;
            let is_down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
            let is_up = msg == WM_KEYUP || msg == WM_SYSKEYUP;
            let vk = k.vkCode as u16;
            let swallow = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&st, vk, is_down, is_up)))
                .unwrap_or(false);
            if swallow {
                return 1;
            }
        }
    }
    CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam)
}

/// Returns true when the key event should be swallowed.
fn handle(st: &HookState, vk: u16, is_down: bool, is_up: bool) -> bool {
    if st.paused.load(Ordering::Relaxed) {
        return false;
    }
    if is_down && vk == VK_ESCAPE && st.active.load(Ordering::Relaxed) {
        st.down.store(false, Ordering::Relaxed);
        (st.cb)(HotkeyEvent::Cancel);
        return true;
    }
    let spec = &st.spec;
    if spec.kind == "modifier" {
        let Some((want, _)) = modifier_vk(spec.keycode) else { return false };
        if vk != want {
            return false;
        }
        if is_down && !st.down.swap(true, Ordering::Relaxed) {
            let shift = (key_down(VK_LSHIFT) || key_down(VK_RSHIFT)) && want != VK_LSHIFT && want != VK_RSHIFT;
            (st.cb)(HotkeyEvent::Down { shift });
        } else if is_up && st.down.swap(false, Ordering::Relaxed) {
            (st.cb)(HotkeyEvent::Up);
        }
        return true;
    }
    let Some((want, _)) = key_vk(spec.keycode) else { return false };
    if vk != want {
        return false;
    }
    let held = |m: &str| match m {
        "control" => key_down(VK_LCONTROL) || key_down(VK_RCONTROL),
        "option" => key_down(VK_LMENU) || key_down(VK_RMENU),
        "shift" => key_down(VK_LSHIFT) || key_down(VK_RSHIFT),
        "command" => key_down(VK_LWIN) || key_down(VK_RWIN),
        _ => false,
    };
    let uses = |m: &str| spec.mods.iter().any(|x| x == m);
    if is_down {
        let mods_ok = ["control", "option", "command"].iter().all(|m| held(m) == uses(m)) && (!uses("shift") || held("shift"));
        if !mods_ok {
            return false;
        }
        if !st.down.swap(true, Ordering::Relaxed) {
            (st.cb)(HotkeyEvent::Down { shift: held("shift") && !uses("shift") });
        }
        return true;
    }
    if is_up && st.down.swap(false, Ordering::Relaxed) {
        (st.cb)(HotkeyEvent::Up);
        return true;
    }
    false
}

impl HotkeyTap {
    pub fn install(spec: Hotkey, cb: HotkeyCallback) -> Option<(HotkeyTap, &'static str)> {
        let state = Arc::new(HookState {
            spec,
            cb,
            down: AtomicBool::new(false),
            active: AtomicBool::new(false),
            paused: AtomicBool::new(false),
        });
        *HOOK.lock().unwrap() = Some(state.clone());
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        std::thread::Builder::new()
            .name("cz-hotkey".into())
            .spawn(move || unsafe {
                let hook = SetWindowsHookExW(WH_KEYBOARD_LL, Some(hook_proc), GetModuleHandleW(std::ptr::null()), 0);
                let _ = tx.send(if hook.is_null() { 0 } else { GetCurrentThreadId() });
                if hook.is_null() {
                    return;
                }
                let mut msg: MSG = std::mem::zeroed();
                while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {}
                UnhookWindowsHookEx(hook);
            })
            .ok()?;
        let tid = rx.recv().ok()?;
        if tid == 0 {
            return None;
        }
        Some((HotkeyTap { state, thread: AtomicU32::new(tid) }, "hook"))
    }
    pub fn set_active(&self, on: bool) {
        self.state.active.store(on, Ordering::Relaxed)
    }
    pub fn set_paused(&self, on: bool) {
        self.state.paused.store(on, Ordering::Relaxed)
    }
}

impl Drop for HotkeyTap {
    fn drop(&mut self) {
        unsafe { PostThreadMessageW(self.thread.load(Ordering::Relaxed), WM_QUIT, 0, 0) };
        let mut g = HOOK.lock().unwrap();
        if g.as_ref().map(|s| Arc::ptr_eq(s, &self.state)).unwrap_or(false) {
            *g = None;
        }
    }
}

// ----------------------------------------------------------------------------- clipboard / paste

fn open_clipboard() -> bool {
    // another app may hold the clipboard for a moment
    for _ in 0..10 {
        if unsafe { OpenClipboard(std::ptr::null_mut()) } != 0 {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    false
}

fn clipboard_text() -> Option<String> {
    if !open_clipboard() {
        return None;
    }
    let out = unsafe {
        let h = GetClipboardData(CF_UNICODETEXT);
        if h.is_null() {
            None
        } else {
            let p = GlobalLock(h as HGLOBAL) as *const u16;
            if p.is_null() {
                None
            } else {
                let mut n = 0;
                while *p.add(n) != 0 {
                    n += 1;
                }
                let s = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
                GlobalUnlock(h as HGLOBAL);
                Some(s)
            }
        }
    };
    unsafe { CloseClipboard() };
    out
}

pub fn copy_text(text: &str) {
    let w = wide(text);
    if !open_clipboard() {
        return;
    }
    unsafe {
        EmptyClipboard();
        let h = GlobalAlloc(GMEM_MOVEABLE, w.len() * 2);
        if !h.is_null() {
            let p = GlobalLock(h) as *mut u16;
            if !p.is_null() {
                std::ptr::copy_nonoverlapping(w.as_ptr(), p, w.len());
                GlobalUnlock(h);
                if SetClipboardData(CF_UNICODETEXT, h as HANDLE).is_null() {
                    GlobalFree(h);
                }
            }
        }
        CloseClipboard();
    }
}

fn key_input(vk: u16, up: bool) -> INPUT {
    let mut i: INPUT = unsafe { std::mem::zeroed() };
    i.r#type = INPUT_KEYBOARD;
    i.Anonymous.ki = KEYBDINPUT { wVk: vk, wScan: 0, dwFlags: if up { KEYEVENTF_KEYUP } else { 0 }, time: 0, dwExtraInfo: 0 };
    i
}

pub fn paste_text(text: &str) -> bool {
    let previous = clipboard_text();
    copy_text(text);
    let seq = unsafe { GetClipboardSequenceNumber() };
    let inputs = [key_input(VK_CONTROL, false), key_input(b'V' as u16, false), key_input(b'V' as u16, true), key_input(VK_CONTROL, true)];
    unsafe { SendInput(inputs.len() as u32, inputs.as_ptr(), std::mem::size_of::<INPUT>() as i32) };
    if let Some(prev) = previous {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            if unsafe { GetClipboardSequenceNumber() } == seq {
                copy_text(&prev);
            }
        });
    }
    true
}

// ----------------------------------------------------------------------------- permissions & misc

pub fn accessibility_trusted(_prompt: bool) -> bool {
    true
}

pub fn open_privacy_pane(which: &str) {
    if which == "Microphone" {
        open_url("ms-settings:privacy-microphone");
    }
}

pub fn frontmost_app() -> String {
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return String::new();
        }
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, &mut pid);
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return String::new();
        }
        let mut buf = [0u16; 512];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len);
        CloseHandle(h);
        if ok == 0 {
            return String::new();
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        std::path::Path::new(&path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    }
}

pub fn show_without_focus(w: *mut c_void) {
    let hwnd = w as HWND;
    if hwnd.is_null() {
        return;
    }
    unsafe {
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let want = ex | (WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT) as isize;
        if ex != want {
            SetWindowLongPtrW(hwnd, GWL_EXSTYLE, want);
        }
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
    }
}

pub fn hide_window(w: *mut c_void) {
    if !w.is_null() {
        unsafe { ShowWindow(w as HWND, SW_HIDE) };
    }
}

pub fn style_main_window(_w: *mut c_void) {}

pub fn main_screen_frame() -> (f64, f64, f64, f64) {
    unsafe {
        let mut r: RECT = std::mem::zeroed();
        if SystemParametersInfoW(SPI_GETWORKAREA, 0, &mut r as *mut RECT as *mut c_void, 0) == 0 {
            return (0.0, 0.0, 1920.0, 1080.0);
        }
        let dpi = windows_sys::Win32::UI::HiDpi::GetDpiForSystem();
        let s = if dpi > 0 { dpi as f64 / 96.0 } else { 1.0 };
        (r.left as f64 / s, r.top as f64 / s, (r.right - r.left) as f64 / s, (r.bottom - r.top) as f64 / s)
    }
}

pub fn system_is_dark() -> bool {
    let key = wide("Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize");
    let name = wide("AppsUseLightTheme");
    let mut v: u32 = 1;
    let mut len = 4u32;
    let rc = unsafe {
        RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), name.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(),
                     &mut v as *mut u32 as *mut c_void, &mut len)
    };
    rc == 0 && v == 0
}

pub fn set_app_icon(_path: &str) {}

pub fn open_url(url: &str) -> bool {
    let (op, u) = (wide("open"), wide(url));
    let r = unsafe {
        ShellExecuteW(std::ptr::null_mut(), op.as_ptr(), u.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL)
    };
    r as isize > 32
}

/// Show and focus the top-level window with exactly this title (another running copy's
/// main window, possibly hidden after its close button was pressed).
pub fn bring_to_front(title: &str) -> bool {
    let t = wide(title);
    unsafe {
        let hwnd = FindWindowW(std::ptr::null(), t.as_ptr());
        if hwnd.is_null() {
            return false;
        }
        ShowWindow(hwnd, SW_SHOW);
        ShowWindow(hwnd, SW_RESTORE);
        SetForegroundWindow(hwnd) != 0
    }
}
