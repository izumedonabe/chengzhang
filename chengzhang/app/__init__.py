"""成章 Chengzhang desktop app: `chengzhang` (or `otl-app`, `python -m chengzhang.app`)."""
import os
from pathlib import Path

WEB = Path(__file__).parent / "web"
ASSETS = Path(__file__).parent / "assets"
_theme_observer = None  # keep a reference so the observer isn't garbage-collected


def _system_is_dark() -> bool:
    from Foundation import NSUserDefaults

    return NSUserDefaults.standardUserDefaults().stringForKey_("AppleInterfaceStyle") == "Dark"


def apply_icon():
    """Day icon (sunrise) in light mode, night icon (moon) in dark mode."""
    import AppKit

    name = "chengzhang-dark.icns" if _system_is_dark() else "chengzhang.icns"
    img = AppKit.NSImage.alloc().initWithContentsOfFile_(str(ASSETS / name))
    if img is not None:
        AppKit.NSApplication.sharedApplication().setApplicationIconImage_(img)


def brand_process():
    """Running as `python`, macOS would show Python's rocket and name. Give the
    process our name (menu bar, Cmd-Tab) and icon (Dock) before AppKit starts,
    and follow the system appearance afterwards."""
    global _theme_observer
    try:
        from Foundation import NSBundle, NSDistributedNotificationCenter, NSObject
        from PyObjCTools import AppHelper

        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = "成章"
            info["CFBundleDisplayName"] = "成章"
        apply_icon()

        class ThemeObserver(NSObject):
            def themeChanged_(self, _note):
                AppHelper.callAfter(apply_icon)

        _theme_observer = ThemeObserver.alloc().init()
        NSDistributedNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            _theme_observer, "themeChanged:", "AppleInterfaceThemeChangedNotification", None)
    except Exception as e:  # cosmetic only — never block startup
        print("brand_process:", e)


def main():
    import webview

    from .controller import Api, Controller
    from .store import Store

    brand_process()
    store = Store()
    ctl = Controller(store)
    ctl.main_win = webview.create_window(
        "成章", url=str(WEB / "index.html"), js_api=Api(ctl),
        width=1120, height=740, min_size=(900, 620), background_color="#D9E0DC",
        # frameless = content under the title bar; controller restores the traffic lights
        frameless=True, easy_drag=False,
    )
    ctl.hud_win = webview.create_window(
        "成章 HUD", url=str(WEB / "hud.html"), width=320, height=96,
        frameless=True, easy_drag=False, transparent=True, on_top=True, focus=False,
        hidden=True, shadow=False, background_color="#000000",
    )
    webview.start(ctl.boot, debug=os.environ.get("OTL_DEBUG") == "1")


if __name__ == "__main__":
    main()
