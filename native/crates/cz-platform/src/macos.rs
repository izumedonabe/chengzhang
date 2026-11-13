//! macOS implementation: Quartz event tap for the hotkey, NSPasteboard + synthetic ⌘V
//! for paste, AX trust for permissions. AppKit calls must run on the main thread.
use crate::{Hotkey, HotkeyCallback, HotkeyEvent};
use objc2::runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel};
use objc2::{class, msg_send, sel};
use objc2_foundation::{NSRect, NSString};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

// ----------------------------------------------------------------------------- CoreGraphics FFI

type CFTypeRef = *const c_void;
type CGEventRef = *mut c_void;
type CGEventTapCallBack = extern "C" fn(*mut c_void, u32, CGEventRef, *mut c_void) -> CGEventRef;

#[link(name = "CoreGraphics", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn CGEventTapCreate(tap: u32, place: u32, options: u32, mask: u64, cb: CGEventTapCallBack, user: *mut c_void) -> CFTypeRef;
    fn CGEventTapEnable(tap: CFTypeRef, enable: bool);
    fn CGEventGetIntegerValueField(ev: CGEventRef, field: u32) -> i64;
    fn CGEventGetFlags(ev: CGEventRef) -> u64;
    fn CGEventSourceCreate(state: i32) -> CFTypeRef;
    fn CGEventCreateKeyboardEvent(src: CFTypeRef, keycode: u16, down: bool) -> CGEventRef;
    fn CGEventSetFlags(ev: CGEventRef, flags: u64);
    fn CGEventPost(tap: u32, ev: CGEventRef);
    fn CFMachPortCreateRunLoopSource(alloc: CFTypeRef, port: CFTypeRef, order: isize) -> CFTypeRef;
    fn CFMachPortInvalidate(port: CFTypeRef);
    fn CFRunLoopGetMain() -> CFTypeRef;
    fn CFRunLoopAddSource(rl: CFTypeRef, src: CFTypeRef, mode: CFTypeRef);
    fn CFRunLoopRemoveSource(rl: CFTypeRef, src: CFTypeRef, mode: CFTypeRef);
    fn CFRelease(p: CFTypeRef);
    static kCFRunLoopCommonModes: CFTypeRef;
    fn AXIsProcessTrustedWithOptions(options: CFTypeRef) -> bool;
    fn AXUIElementCreateSystemWide() -> CFTypeRef;
    fn AXUIElementCreateApplication(pid: i32) -> CFTypeRef;
    fn AXUIElementCopyAttributeValue(el: CFTypeRef, attr: CFTypeRef, out: *mut CFTypeRef) -> i32;
    fn AXUIElementSetMessagingTimeout(el: CFTypeRef, secs: f32) -> i32;
    fn CFGetTypeID(p: CFTypeRef) -> usize;
    fn CFStringGetTypeID() -> usize;
    fn CGEventSourceFlagsState(state: i32) -> u64;
}

const KEY_DOWN: u32 = 10;
const KEY_UP: u32 = 11;
const FLAGS_CHANGED: u32 = 12;
const TAP_DISABLED_TIMEOUT: u32 = 0xFFFF_FFFE;
const TAP_DISABLED_USER: u32 = 0xFFFF_FFFF;
const FIELD_AUTOREPEAT: u32 = 8;
const FIELD_KEYCODE: u32 = 9;

const F_SHIFT: u64 = 0x20000;
const F_CONTROL: u64 = 0x40000;
const F_OPTION: u64 = 0x80000;
const F_COMMAND: u64 = 0x100000;
const F_FN: u64 = 0x800000;

const ESC: u16 = 53;

fn mod_flag(name: &str) -> u64 {
    match name {
        "control" => F_CONTROL,
        "option" => F_OPTION,
        "shift" => F_SHIFT,
        "command" => F_COMMAND,
        _ => 0,
    }
}

/// keycode -> (flag set while it's down, label)
fn modifier_key(code: u16) -> Option<(u64, &'static str)> {
    Some(match code {
        54 => (F_COMMAND, "右 ⌘"),
        55 => (F_COMMAND, "左 ⌘"),
        61 => (F_OPTION, "右 ⌥"),
        58 => (F_OPTION, "左 ⌥"),
        60 => (F_SHIFT, "右 ⇧"),
        56 => (F_SHIFT, "左 ⇧"),
        62 => (F_CONTROL, "右 ⌃"),
        59 => (F_CONTROL, "左 ⌃"),
        63 => (F_FN, "fn"),
        _ => return None,
    })
}

fn key_name(code: u16) -> Option<&'static str> {
    Some(match code {
        49 => "Space", 36 => "Return", 48 => "Tab", 51 => "Delete", 53 => "esc", 50 => "`",
        122 => "F1", 120 => "F2", 99 => "F3", 118 => "F4", 96 => "F5", 97 => "F6", 98 => "F7",
        100 => "F8", 101 => "F9", 109 => "F10", 103 => "F11", 111 => "F12", 105 => "F13",
        107 => "F14", 113 => "F15", 106 => "F16", 64 => "F17", 79 => "F18", 80 => "F19", 90 => "F20",
        123 => "←", 124 => "→", 125 => "↓", 126 => "↑",
        _ => return None,
    })
}

pub const PRESET_KEYCODES: &[u16] = &[61, 54, 60, 62, 63];

pub fn legacy_keycode(name: &str) -> Option<u16> {
    match name {
        "right_option" => Some(61),
        "right_command" => Some(54),
        "fn" => Some(63),
        _ => None,
    }
}

pub fn hotkey_label(h: &Hotkey) -> String {
    if h.kind == "modifier" {
        return modifier_key(h.keycode).map(|m| m.1.to_string()).unwrap_or(format!("键 {}", h.keycode));
    }
    let mut mods = String::new();
    for (m, sym) in [("control", "⌃"), ("option", "⌥"), ("shift", "⇧"), ("command", "⌘")] {
        if h.mods.iter().any(|x| x == m) {
            mods += sym;
        }
    }
    let key = h.name.clone().or_else(|| key_name(h.keycode).map(String::from)).unwrap_or(format!("键 {}", h.keycode));
    format!("{mods} {key}").trim().to_string()
}

// ----------------------------------------------------------------------------- hotkey tap

struct TapState {
    spec: Hotkey,
    learn: Option<Hotkey>,
    cb: HotkeyCallback,
    down: AtomicBool,
    learn_down: AtomicBool,
    active: AtomicBool,
    paused: AtomicBool,
    tap: std::cell::Cell<CFTypeRef>,
}

pub struct HotkeyTap {
    state: Box<TapState>,
    source: CFTypeRef,
}

unsafe impl Send for HotkeyTap {}
unsafe impl Sync for HotkeyTap {}

extern "C" fn tap_callback(_proxy: *mut c_void, etype: u32, event: CGEventRef, user: *mut c_void) -> CGEventRef {
    let st = unsafe { &*(user as *const TapState) };
    if etype == TAP_DISABLED_TIMEOUT || etype == TAP_DISABLED_USER {
        unsafe { CGEventTapEnable(st.tap.get(), true) }; // the system turned us off; turn back on
        return event;
    }
    let code = unsafe { CGEventGetIntegerValueField(event, FIELD_KEYCODE) } as u16;
    let repeat = unsafe { CGEventGetIntegerValueField(event, FIELD_AUTOREPEAT) } != 0;
    let flags = unsafe { CGEventGetFlags(event) };
    // never let a panic unwind into the event system
    let swallow = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(st, etype, code, flags, repeat)))
        .unwrap_or(false);
    if swallow { std::ptr::null_mut() } else { event }
}

/// Returns true when the event should be swallowed.
fn handle(st: &TapState, kind: u32, code: u16, flags: u64, repeat: bool) -> bool {
    if st.paused.load(Ordering::Relaxed) {
        return false;
    }
    let spec = &st.spec;
    if kind == KEY_DOWN && code == ESC && st.active.load(Ordering::Relaxed) {
        st.down.store(false, Ordering::Relaxed);
        (st.cb)(HotkeyEvent::Cancel);
        return true;
    }
    // the correction hotkey: exactly its modifiers, fires once per press
    if let Some(l) = st.learn.as_ref().filter(|l| l.keycode == code && kind != FLAGS_CHANGED) {
        if kind == KEY_DOWN {
            let want = l.mods.iter().fold(0, |a, m| a | mod_flag(m));
            if flags & (F_CONTROL | F_OPTION | F_COMMAND | F_SHIFT) == want {
                if !repeat {
                    st.learn_down.store(true, Ordering::Relaxed);
                    (st.cb)(HotkeyEvent::Learn);
                }
                return true;
            }
        } else if kind == KEY_UP && st.learn_down.swap(false, Ordering::Relaxed) {
            return true;
        }
    }
    if spec.kind == "modifier" {
        if kind != FLAGS_CHANGED || code != spec.keycode {
            return false;
        }
        let flag = modifier_key(code).map(|m| m.0).unwrap_or(0);
        let pressed = flags & flag != 0;
        let was = st.down.load(Ordering::Relaxed);
        if pressed && !was {
            st.down.store(true, Ordering::Relaxed);
            (st.cb)(HotkeyEvent::Down { shift: flags & F_SHIFT != 0 && code != 56 && code != 60 });
        } else if !pressed && was {
            st.down.store(false, Ordering::Relaxed);
            (st.cb)(HotkeyEvent::Up);
        }
        return false; // lone modifiers pass through untouched
    }
    // key (+ modifiers)
    if code != spec.keycode || kind == FLAGS_CHANGED {
        return false;
    }
    let want = spec.mods.iter().fold(0, |a, m| a | mod_flag(m));
    let held = flags & (F_CONTROL | F_OPTION | F_COMMAND | F_SHIFT);
    let uses_shift = spec.mods.iter().any(|m| m == "shift");
    if kind == KEY_DOWN {
        if (held & !F_SHIFT) != (want & !F_SHIFT) || (uses_shift && held & F_SHIFT == 0) {
            return false;
        }
        if !repeat && !st.down.load(Ordering::Relaxed) {
            st.down.store(true, Ordering::Relaxed);
            (st.cb)(HotkeyEvent::Down { shift: held & F_SHIFT != 0 && !uses_shift });
        }
        return true;
    }
    if kind == KEY_UP && st.down.load(Ordering::Relaxed) {
        st.down.store(false, Ordering::Relaxed);
        (st.cb)(HotkeyEvent::Up);
        return true;
    }
    false
}

impl HotkeyTap {
    pub fn install(spec: Hotkey, learn: Option<Hotkey>, cb: HotkeyCallback) -> Option<(HotkeyTap, &'static str)> {
        let state = Box::new(TapState {
            spec,
            learn: learn.filter(|l| l.kind == "key"),
            cb,
            down: AtomicBool::new(false),
            learn_down: AtomicBool::new(false),
            active: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            tap: std::cell::Cell::new(std::ptr::null()),
        });
        let mask = (1u64 << KEY_DOWN) | (1u64 << KEY_UP) | (1u64 << FLAGS_CHANGED);
        let user = &*state as *const TapState as *mut c_void;
        // full tap (can swallow key combos) needs Accessibility; listen-only needs Input Monitoring
        for (options, how) in [(0u32, "tap"), (1u32, "listen")] {
            let tap = unsafe { CGEventTapCreate(1, 0, options, mask, tap_callback, user) };
            if tap.is_null() {
                continue;
            }
            state.tap.set(tap);
            unsafe {
                let source = CFMachPortCreateRunLoopSource(std::ptr::null(), tap, 0);
                CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
                CGEventTapEnable(tap, true);
                return Some((HotkeyTap { state, source }, how));
            }
        }
        None
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
        unsafe {
            let tap = self.state.tap.get();
            CGEventTapEnable(tap, false);
            CFRunLoopRemoveSource(CFRunLoopGetMain(), self.source, kCFRunLoopCommonModes);
            CFMachPortInvalidate(tap);
            CFRelease(self.source);
            CFRelease(tap);
        }
    }
}

// ----------------------------------------------------------------------------- clipboard / paste

fn ns(s: &str) -> objc2::rc::Retained<NSString> {
    NSString::from_str(s)
}

fn pasteboard() -> *mut AnyObject {
    unsafe { msg_send![class!(NSPasteboard), generalPasteboard] }
}

const PB_STRING: &str = "public.utf8-plain-text";

fn pb_string() -> Option<String> {
    unsafe {
        let s: *mut NSString = msg_send![pasteboard(), stringForType: &*ns(PB_STRING)];
        s.as_ref().map(|s| s.to_string())
    }
}

fn pb_change_count() -> isize {
    unsafe { msg_send![pasteboard(), changeCount] }
}

pub fn copy_text(text: &str) {
    unsafe {
        let pb = pasteboard();
        let _: isize = msg_send![pb, clearContents];
        let _: Bool = msg_send![pb, setString: &*ns(text), forType: &*ns(PB_STRING)];
    }
}

/// Press and release ⌘ + `keycode` in the frontmost app.
fn post_command_key(keycode: u16) {
    unsafe {
        let src = CGEventSourceCreate(1); // HID system state
        for down in [true, false] {
            let ev = CGEventCreateKeyboardEvent(src, keycode, down);
            CGEventSetFlags(ev, F_COMMAND);
            CGEventPost(0, ev);
            CFRelease(ev);
        }
        if !src.is_null() {
            CFRelease(src);
        }
    }
}

pub fn paste_text(text: &str) -> bool {
    if !accessibility_trusted(false) {
        copy_text(text);
        return false;
    }
    let previous = pb_string();
    copy_text(text);
    let change = pb_change_count();
    post_command_key(9); // "v"
    if let Some(prev) = previous {
        // put the user's clipboard back, unless someone else touched it meanwhile
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            if pb_change_count() == change {
                copy_text(&prev);
            }
        });
    }
    true
}

// ----------------------------------------------------------------------------- reading the user's text

/// Copy a string-valued attribute of an AX element.
unsafe fn ax_string(el: CFTypeRef, attr: &str) -> Option<String> {
    let key = ns(attr);
    let mut v: CFTypeRef = std::ptr::null();
    let err = AXUIElementCopyAttributeValue(el, objc2::rc::Retained::as_ptr(&key) as CFTypeRef, &mut v);
    if err != 0 || v.is_null() {
        return None;
    }
    let s = (CFGetTypeID(v) == CFStringGetTypeID()).then(|| (*(v as *const NSString)).to_string());
    CFRelease(v);
    s
}

/// Copy an element-valued attribute (caller releases). Null on failure.
unsafe fn ax_element(el: CFTypeRef, attr: &str) -> CFTypeRef {
    let key = ns(attr);
    let mut v: CFTypeRef = std::ptr::null();
    let err = AXUIElementCopyAttributeValue(el, objc2::rc::Retained::as_ptr(&key) as CFTypeRef, &mut v);
    if err != 0 { std::ptr::null() } else { v }
}

fn frontmost_pid() -> Option<i32> {
    unsafe {
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        let app: *mut AnyObject = msg_send![ws, frontmostApplication];
        if app.is_null() {
            return None;
        }
        let pid: i32 = msg_send![app, processIdentifier];
        (pid > 0).then_some(pid)
    }
}

/// The focused text field's selection, else its whole value. Native apps, WeChat, most
/// text views; Electron apps and some browsers don't expose their fields this way.
fn ax_focused_text() -> Option<(String, &'static str)> {
    unsafe {
        // ask the frontmost app for its focused element; the system-wide element is the
        // fallback (it answers "cannot complete" for some apps)
        let mut el: CFTypeRef = std::ptr::null();
        if let Some(pid) = frontmost_pid() {
            let app = AXUIElementCreateApplication(pid);
            if !app.is_null() {
                AXUIElementSetMessagingTimeout(app, 0.5); // a hung app must not hang us
                el = ax_element(app, "AXFocusedUIElement");
                CFRelease(app);
            }
        }
        if el.is_null() {
            let sys = AXUIElementCreateSystemWide();
            if !sys.is_null() {
                AXUIElementSetMessagingTimeout(sys, 0.5);
                let app = ax_element(sys, "AXFocusedApplication");
                if !app.is_null() {
                    el = ax_element(app, "AXFocusedUIElement");
                    CFRelease(app);
                }
                if el.is_null() {
                    el = ax_element(sys, "AXFocusedUIElement");
                }
                CFRelease(sys);
            }
        }
        if el.is_null() {
            return None;
        }
        let nonblank = |s: &String| !s.trim().is_empty();
        let r = ax_string(el, "AXSelectedText").filter(nonblank).map(|s| (s, "selection"))
            .or_else(|| ax_string(el, "AXValue").filter(nonblank).map(|s| (s, "field")));
        CFRelease(el);
        r
    }
}

/// The selection, via a synthetic ⌘C; the user's clipboard text is put back afterwards.
fn copy_selection() -> Option<String> {
    if !accessibility_trusted(false) {
        return None;
    }
    // the correction hotkey's modifiers may still be down; ⌃⌥⌘C isn't a copy everywhere
    let t0 = std::time::Instant::now();
    while unsafe { CGEventSourceFlagsState(1) } & (F_CONTROL | F_OPTION | F_SHIFT | F_COMMAND) != 0
        && t0.elapsed().as_millis() < 1000
    {
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
    let previous = pb_string();
    let before = pb_change_count();
    post_command_key(8); // "c"
    let t0 = std::time::Instant::now();
    while pb_change_count() == before && t0.elapsed().as_millis() < 400 {
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
    if pb_change_count() == before {
        return None; // nothing selected
    }
    let got = pb_string();
    if let Some(prev) = previous {
        copy_text(&prev);
    }
    got.filter(|s| !s.trim().is_empty())
}

pub fn grab_text() -> Option<(String, &'static str)> {
    ax_focused_text().or_else(|| copy_selection().map(|s| (s, "copy")))
}

// ----------------------------------------------------------------------------- permissions & misc

pub fn accessibility_trusted(prompt: bool) -> bool {
    unsafe {
        let value: *mut AnyObject = msg_send![class!(NSNumber), numberWithBool: Bool::new(prompt)];
        let key = ns("AXTrustedCheckOptionPrompt");
        let dict: *mut AnyObject = msg_send![class!(NSDictionary), dictionaryWithObject: value, forKey: &*key];
        AXIsProcessTrustedWithOptions(dict as CFTypeRef)
    }
}

pub fn open_privacy_pane(which: &str) {
    let url = format!("x-apple.systempreferences:com.apple.preference.security?Privacy_{which}");
    unsafe {
        let u: *mut AnyObject = msg_send![class!(NSURL), URLWithString: &*ns(&url)];
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        let _: Bool = msg_send![ws, openURL: u];
    }
}

pub fn frontmost_app() -> String {
    unsafe {
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        let app: *mut AnyObject = msg_send![ws, frontmostApplication];
        if app.is_null() {
            return String::new();
        }
        let name: *mut NSString = msg_send![app, localizedName];
        name.as_ref().map(|s| s.to_string()).unwrap_or_default()
    }
}

/// NSPanel subclass that never takes key/main status (the HUD must not steal focus).
fn hud_panel_class() -> &'static AnyClass {
    static CLS: OnceLock<&'static AnyClass> = OnceLock::new();
    CLS.get_or_init(|| {
        extern "C-unwind" fn no(_: &AnyObject, _: Sel) -> Bool {
            Bool::NO
        }
        let mut b = ClassBuilder::new(c"CZHudPanel", class!(NSPanel)).expect("CZHudPanel");
        unsafe {
            b.add_method(sel!(canBecomeKeyWindow), no as extern "C-unwind" fn(_, _) -> _);
            b.add_method(sel!(canBecomeMainWindow), no as extern "C-unwind" fn(_, _) -> _);
        }
        b.register()
    })
}

/// Turn the (tao) NSWindow into a non-activating NSPanel, once. A regular app's NSWindow is kept
/// out of another app's full-screen Space even with FullScreenAuxiliary; a non-activating panel
/// is allowed in. Swapping the class is sound: NSPanel adds no ivars to NSWindow (both 528 bytes),
/// so tao's own ivars after them stay where they are. Same trick as the tauri-nspanel plugin.
unsafe fn make_panel(w: *mut AnyObject) {
    let cls = hud_panel_class();
    if std::ptr::eq(objc2::ffi::object_getClass(w), cls) {
        return;
    }
    objc2::ffi::object_setClass(w, cls);
    let mask: usize = msg_send![w, styleMask];
    let _: () = msg_send![w, setStyleMask: mask | (1usize << 7)]; // NonactivatingPanel
    let _: () = msg_send![w, setFloatingPanel: Bool::YES];
    let _: () = msg_send![w, setHidesOnDeactivate: Bool::NO];
    let _: () = msg_send![w, setBecomesKeyOnlyIfNeeded: Bool::YES];
}

pub fn show_without_focus(w: *mut c_void) {
    let w = w as *mut AnyObject;
    if w.is_null() {
        return;
    }
    unsafe {
        make_panel(w);
        let _: () = msg_send![w, setIgnoresMouseEvents: Bool::YES];
        // CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary
        let _: () = msg_send![w, setCollectionBehavior: (1usize << 0) | (1usize << 4) | (1usize << 6) | (1usize << 8)];
        let _: () = msg_send![w, setLevel: 25isize]; // NSStatusWindowLevel
        let _: () = msg_send![w, orderFrontRegardless];
    }
}

pub fn hide_window(w: *mut c_void) {
    let w = w as *mut AnyObject;
    if !w.is_null() {
        unsafe {
            let _: () = msg_send![w, orderOut: std::ptr::null::<AnyObject>()];
        }
    }
}

pub fn style_main_window(w: *mut c_void) {
    let w = w as *mut AnyObject;
    if w.is_null() {
        return;
    }
    unsafe {
        let _: () = msg_send![w, setTitleVisibility: 1isize]; // hidden
        let _: () = msg_send![w, setTitlebarAppearsTransparent: Bool::YES];
        let tb: *mut AnyObject = msg_send![class!(NSToolbar), alloc];
        let tb: *mut AnyObject = msg_send![tb, initWithIdentifier: &*ns("cz-main")];
        let _: () = msg_send![tb, setShowsBaselineSeparator: Bool::NO];
        let _: () = msg_send![w, setToolbar: tb];
        let _: () = msg_send![w, setToolbarStyle: 3isize]; // unified
        let _: () = msg_send![w, setTitlebarSeparatorStyle: 1isize]; // none
    }
}

pub fn main_screen_frame() -> (f64, f64, f64, f64) {
    unsafe {
        let screen: *mut AnyObject = msg_send![class!(NSScreen), mainScreen];
        if screen.is_null() {
            return (0.0, 0.0, 1440.0, 900.0);
        }
        let full: NSRect = msg_send![screen, frame];
        let vis: NSRect = msg_send![screen, visibleFrame];
        let top = full.size.height - (vis.origin.y + vis.size.height);
        (vis.origin.x, top, vis.size.width, vis.size.height)
    }
}

pub fn system_is_dark() -> bool {
    unsafe {
        let d: *mut AnyObject = msg_send![class!(NSUserDefaults), standardUserDefaults];
        let s: *mut NSString = msg_send![d, stringForKey: &*ns("AppleInterfaceStyle")];
        s.as_ref().map(|s| s.to_string() == "Dark").unwrap_or(false)
    }
}

/// Dock icon (sunrise by day, moon by night). Call on the main thread.
pub fn set_app_icon(path: &str) {
    unsafe {
        let img: *mut AnyObject = msg_send![class!(NSImage), alloc];
        let img: *mut AnyObject = msg_send![img, initWithContentsOfFile: &*ns(path)];
        if img.is_null() {
            return;
        }
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let _: () = msg_send![app, setApplicationIconImage: img];
    }
}

/// Open a URL in the default browser.
pub fn open_url(url: &str) -> bool {
    unsafe {
        let u: *mut AnyObject = msg_send![class!(NSURL), URLWithString: &*ns(url)];
        if u.is_null() {
            return false;
        }
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        let ok: Bool = msg_send![ws, openURL: u];
        ok.as_bool()
    }
}

/// macOS brings a running app forward through LaunchServices instead (see app/single.rs).
pub fn bring_to_front(_title: &str) -> bool {
    false
}
