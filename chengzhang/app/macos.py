"""macOS glue: global push-to-talk hotkey, paste into the frontmost app,
permission checks. Everything AppKit here must run on the main thread."""
from __future__ import annotations

import threading
import time

import AppKit
import Quartz
from PyObjCTools import AppHelper

# ----------------------------------------------------------------------------- hotkeys
# A hotkey is a dict:
#   {"kind": "modifier", "keycode": 61}                      a lone modifier key (right ⌥ …)
#   {"kind": "key", "keycode": 49, "mods": ["control"]}      a key, optionally with modifiers

MODIFIER_KEYS = {  # keycode -> (flag that is set while it's down, label)
    54: (Quartz.kCGEventFlagMaskCommand, "右 ⌘"), 55: (Quartz.kCGEventFlagMaskCommand, "左 ⌘"),
    61: (Quartz.kCGEventFlagMaskAlternate, "右 ⌥"), 58: (Quartz.kCGEventFlagMaskAlternate, "左 ⌥"),
    60: (Quartz.kCGEventFlagMaskShift, "右 ⇧"), 56: (Quartz.kCGEventFlagMaskShift, "左 ⇧"),
    62: (Quartz.kCGEventFlagMaskControl, "右 ⌃"), 59: (Quartz.kCGEventFlagMaskControl, "左 ⌃"),
    63: (Quartz.kCGEventFlagMaskSecondaryFn, "fn"),
}
MOD_FLAGS = {"control": Quartz.kCGEventFlagMaskControl, "option": Quartz.kCGEventFlagMaskAlternate,
             "shift": Quartz.kCGEventFlagMaskShift, "command": Quartz.kCGEventFlagMaskCommand}
MOD_SYMBOLS = {"control": "⌃", "option": "⌥", "shift": "⇧", "command": "⌘"}
KEY_NAMES = {49: "Space", 36: "Return", 48: "Tab", 51: "Delete", 53: "esc", 50: "`",
             122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5", 97: "F6", 98: "F7", 100: "F8",
             101: "F9", 109: "F10", 103: "F11", 111: "F12", 105: "F13", 107: "F14", 113: "F15",
             106: "F16", 64: "F17", 79: "F18", 80: "F19", 90: "F20",
             123: "←", 124: "→", 125: "↓", 126: "↑"}
LEGACY = {"right_option": 61, "right_command": 54, "fn": 63}
PRESETS = [{"kind": "modifier", "keycode": k} for k in (61, 54, 60, 62, 63)]


def normalize_hotkey(spec) -> dict:
    if isinstance(spec, str):  # settings from before hotkeys were configurable
        return {"kind": "modifier", "keycode": LEGACY.get(spec, 61)}
    if isinstance(spec, dict) and "keycode" in spec:
        return {"kind": spec.get("kind", "key"), "keycode": int(spec["keycode"]),
                "mods": [m for m in spec.get("mods", []) if m in MOD_FLAGS], "name": spec.get("name")}
    return {"kind": "modifier", "keycode": 61}


def hotkey_label(spec) -> str:
    spec = normalize_hotkey(spec)
    if spec["kind"] == "modifier":
        return MODIFIER_KEYS.get(spec["keycode"], (0, f"键 {spec['keycode']}"))[1]
    mods = "".join(MOD_SYMBOLS[m] for m in ("control", "option", "shift", "command") if m in spec["mods"])
    key = spec.get("name") or KEY_NAMES.get(spec["keycode"], f"键 {spec['keycode']}")
    return f"{mods} {key}".strip()


class HotkeyTap:
    """Global hotkey via a Quartz event tap.

    Calls on_down(translate) / on_up() / on_cancel(). Translate = Shift held when
    the hotkey goes down (unless Shift is part of the hotkey). Key-combo hotkeys
    are swallowed so they don't also type into the frontmost app; lone modifier
    keys pass through untouched. Needs Accessibility; without it, falls back to
    an NSEvent monitor (lone modifier keys only, nothing swallowed)."""

    def __init__(self, spec, on_down, on_up, on_cancel=None):
        self.spec = normalize_hotkey(spec)
        self.on_down, self.on_up, self.on_cancel = on_down, on_up, on_cancel
        self.down = False
        self.active = False  # a dictation is running (Esc cancels it)
        self.paused = False  # while the settings page records a new hotkey
        self.tap = self.source = None
        self._monitors = []

    # -- install / remove (main thread)
    def install(self):
        mask = (Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
                | Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged))
        self.tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                           Quartz.kCGEventTapOptionDefault, mask, self._callback, None)
        if self.tap is not None:
            self.source = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
            Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetMain(), self.source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(self.tap, True)
            return "tap"
        # no Accessibility permission: listen-only fallback for lone modifier keys
        def handle(event):
            try:
                self._handle(event.type(), event.keyCode(), int(event.modifierFlags()), False)
            except Exception as e:
                print("hotkey error:", e)
            return event
        m = AppKit.NSEventMaskFlagsChanged | AppKit.NSEventMaskKeyDown | AppKit.NSEventMaskKeyUp
        self._monitors = [x for x in (AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(m, handle),
                                      AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(m, handle)) if x]
        return "monitor"

    def uninstall(self):
        if self.tap is not None:
            Quartz.CGEventTapEnable(self.tap, False)
            Quartz.CFRunLoopRemoveSource(Quartz.CFRunLoopGetMain(), self.source, Quartz.kCFRunLoopCommonModes)
            self.tap = self.source = None
        for m in self._monitors:
            AppKit.NSEvent.removeMonitor_(m)
        self._monitors = []

    # -- event handling
    def _callback(self, proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self.tap, True)  # the system turned us off; turn back on
            return event
        try:
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            repeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
            kind = {Quartz.kCGEventKeyDown: AppKit.NSEventTypeKeyDown, Quartz.kCGEventKeyUp: AppKit.NSEventTypeKeyUp,
                    Quartz.kCGEventFlagsChanged: AppKit.NSEventTypeFlagsChanged}.get(etype)
            swallow = self._handle(kind, code, int(Quartz.CGEventGetFlags(event)), bool(repeat))
            return None if swallow else event
        except Exception as e:  # never let an exception escape into the event system
            print("hotkey error:", e)
            return event

    def _handle(self, kind, code, flags, repeat) -> bool:
        """returns True when the event should be swallowed"""
        if self.paused or kind is None:
            return False
        spec = self.spec
        if kind == AppKit.NSEventTypeKeyDown and code == 53 and self.active and self.on_cancel:
            self.down = False
            self.on_cancel()
            return True
        if spec["kind"] == "modifier":
            if kind != AppKit.NSEventTypeFlagsChanged or code != spec["keycode"]:
                return False
            flag = MODIFIER_KEYS.get(code, (0,))[0]
            pressed = bool(flags & flag)
            if pressed and not self.down:
                self.down = True
                self.on_down(bool(flags & Quartz.kCGEventFlagMaskShift) and code not in (56, 60))
            elif not pressed and self.down:
                self.down = False
                self.on_up()
            return False
        # key (+ modifiers)
        if code != spec["keycode"] or kind == AppKit.NSEventTypeFlagsChanged:
            return False
        want = 0
        for m in spec["mods"]:
            want |= MOD_FLAGS[m]
        held = flags & (MOD_FLAGS["control"] | MOD_FLAGS["option"] | MOD_FLAGS["command"] | MOD_FLAGS["shift"])
        uses_shift = "shift" in spec["mods"]
        if kind == AppKit.NSEventTypeKeyDown:
            if (held & ~MOD_FLAGS["shift"]) != (want & ~MOD_FLAGS["shift"]) or (uses_shift and not held & MOD_FLAGS["shift"]):
                return False
            if not repeat and not self.down:
                self.down = True
                self.on_down(bool(held & MOD_FLAGS["shift"]) and not uses_shift)
            return True
        if kind == AppKit.NSEventTypeKeyUp and self.down:
            self.down = False
            self.on_up()
            return True
        return False


def accessibility_trusted(prompt: bool = False) -> bool:
    from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

    return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))


def open_privacy_pane(which: str = "Accessibility"):
    url = f"x-apple.systempreferences:com.apple.preference.security?Privacy_{which}"
    AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(url))


def frontmost_app() -> str:
    app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(app.localizedName()) if app else ""


def copy_text(text: str):
    pb = AppKit.NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, AppKit.NSPasteboardTypeString)


def paste_text(text: str, restore_after: float = 0.6) -> bool:
    """Put text on the clipboard, send ⌘V to the frontmost app, then put the
    previous clipboard text back so dictation doesn't clobber it.
    Without Accessibility permission macOS silently drops synthetic keystrokes,
    so in that case we only copy and return False."""
    if not accessibility_trusted(False):
        copy_text(text)
        return False
    pb = AppKit.NSPasteboard.generalPasteboard()
    previous = pb.stringForType_(AppKit.NSPasteboardTypeString)
    copy_text(text)
    change = pb.changeCount()

    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, 9, down)  # 9 = "v"
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    if previous is not None:
        def restore():
            time.sleep(restore_after)
            # only restore if nobody else touched the clipboard meanwhile
            if pb.changeCount() == change:
                AppHelper.callAfter(copy_text, str(previous))

        threading.Thread(target=restore, daemon=True).start()
    return True


def show_without_focus(nswindow):
    """Show a floating window without activating our app (keeps the user's
    cursor in whatever app they're typing in)."""
    nswindow.setIgnoresMouseEvents_(True)
    nswindow.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
    )
    nswindow.setLevel_(AppKit.NSStatusWindowLevel)
    nswindow.orderFrontRegardless()


def hide_window(nswindow):
    nswindow.orderOut_(None)


def main_screen_frame():
    f = AppKit.NSScreen.mainScreen().visibleFrame()
    return f.origin.x, f.origin.y, f.size.width, f.size.height
