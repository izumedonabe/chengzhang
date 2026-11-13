//! 成章 native app: Tauri shell around the existing web UI (chengzhang/app/web, unchanged).
//! The UI calls `window.pywebview.api.<method>(...)`; a tiny bridge script maps that onto
//! one Tauri command, so the same HTML/CSS/JS runs on the Python and the native app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod controller;
mod mic;
#[cfg(target_os = "macos")]
#[path = "models_mlx.rs"]
mod models;
#[cfg(not(target_os = "macos"))]
#[path = "models_gguf.rs"]
mod models;
mod single;
mod store;

use controller::Ctl;
use serde_json::Value;
use std::sync::Arc;
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

const BRIDGE: &str = r#"
(() => {
  const invoke = (method, args) => window.__TAURI_INTERNALS__.invoke("api", { method, args });
  const api = new Proxy({}, { get: (_, m) => (...args) => invoke(String(m), args) });
  window.pywebview = { api, native: true };
  window.addEventListener("DOMContentLoaded", () => window.dispatchEvent(new Event("pywebviewready")));
  // The page runs under the (transparent) title bar, so the window has no native drag
  // area: drag it by the top strip or the empty sidebar, double-click to zoom.
  const DRAG_STRIP = 52;
  const draggable = (e) => {
    if (e.button !== 0) return false;
    const t = e.target instanceof Element ? e.target : null;
    if (!t || t.closest("a, button, input, select, textarea, label, summary, [contenteditable], [role=button], [role=radio], [role=switch], .seg, dialog, .no-drag")) return false;
    return e.clientY < DRAG_STRIP || !!t.closest(".rail");
  };
  window.addEventListener("mousedown", (e) => {
    if (!draggable(e)) return;
    e.preventDefault();
    window.__TAURI_INTERNALS__.invoke(e.detail === 2 ? "zoom_window" : "drag_window");
  }, true);
})();
"#;

/// The HUD sits on system Liquid Glass: hud.html drops its own background.
const HUD_GLASS: &str = r#"window.addEventListener("DOMContentLoaded", () => document.documentElement.classList.add("glass"));"#;

/// Move the window with the mouse (the press that started it is still down).
#[tauri::command]
fn drag_window(window: tauri::WebviewWindow) {
    let _ = window.start_dragging();
}

/// Double-click on the title area: zoom / unzoom, like a native title bar.
#[tauri::command]
fn zoom_window(window: tauri::WebviewWindow) {
    if window.is_maximized().unwrap_or(false) {
        let _ = window.unmaximize();
    } else {
        let _ = window.maximize();
    }
}

#[tauri::command]
fn api(state: tauri::State<'_, Arc<Ctl>>, method: String, args: Vec<Value>) -> Result<Value, String> {
    state.api(&method, &args)
}

fn icon_path(app: &tauri::AppHandle) -> Option<String> {
    let dark = cz_platform::system_is_dark();
    let name = if dark { "chengzhang-dark.icns" } else { "chengzhang.icns" };
    app.path().resource_dir().ok().map(|d| d.join(name)).filter(|p| p.exists()).map(|p| p.to_string_lossy().into_owned())
}

fn apply_icon(app: &tauri::AppHandle) {
    if let Some(p) = icon_path(app) {
        cz_platform::set_app_icon(&p);
    }
}

fn main() {
    if let single::Instance::Second(location) = single::claim() {
        eprintln!("[cz] 成章 is already running{}; bringing it forward",
                  location.as_deref().map(|l| format!(" at {l}")).unwrap_or_default());
        if let Some(l) = location {
            single::show_other(&l);
        }
        return;
    }
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![api, drag_window, zoom_window])
        .setup(|app| {
            let handle = app.handle().clone();
            let ctl = Ctl::new(handle.clone());
            app.manage(ctl.clone());

            // CZ_TEST_JS: extra script for automated checks (runs after the page has loaded)
            let test_js = std::env::var("CZ_TEST_JS").map(|js| format!("window.addEventListener('load', () => setTimeout(() => {{ {js} }}, 1500));")).unwrap_or_default();
            let main = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("成章")
                .inner_size(1120.0, 740.0)
                .min_inner_size(900.0, 620.0)
                .background_color(tauri::window::Color(0xD9, 0xE0, 0xDC, 0xFF))
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .initialization_script(BRIDGE)
                .initialization_script(&test_js)
                .build()?;
            if let Ok(ns) = main.ns_window() {
                cz_platform::style_main_window(ns);
            }
            let glass = cz_platform::has_glass();
            let hud = WebviewWindowBuilder::new(app, "hud", WebviewUrl::App("hud.html".into()))
                .title("成章 HUD")
                .inner_size(320.0, 96.0)
                .decorations(false)
                .transparent(true)
                .shadow(false)
                .always_on_top(true)
                .focused(false)
                .visible(false)
                .skip_taskbar(true)
                .resizable(false)
                .initialization_script(BRIDGE)
                .initialization_script(if glass { HUD_GLASS } else { "" })
                .build()?;
            let _ = hud.set_ignore_cursor_events(true);
            // both pages animate at the display's own rate (120 Hz on ProMotion)
            for w in [&main, &hud] {
                let _ = w.with_webview(|v| {
                    #[cfg(target_os = "macos")]
                    if !cz_platform::unlock_frame_rate(v.inner()) {
                        eprintln!("[cz] web view stays near 60 fps");
                    }
                    #[cfg(not(target_os = "macos"))]
                    let _ = v;
                });
            }
            if let Ok(spec) = std::env::var("CZ_TEST_HUD") {
                let c = ctl.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    c.test_hud(&spec);
                });
            }
            if glass {
                if let Ok(ns) = hud.ns_window() {
                    // where hud.html draws its pill: 288×60, centred, 14 pt above the bottom
                    cz_platform::add_glass(ns, (16.0, 22.0, 288.0, 60.0), 30.0);
                }
            }
            apply_icon(&handle);
            ctl.boot();
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building 成章");

    app.run(|app, event| match event {
        // closing the window keeps dictation running; the Dock icon brings it back
        RunEvent::WindowEvent { label, event: WindowEvent::CloseRequested { api, .. }, .. } if label == "main" => {
            api.prevent_close();
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.hide();
            }
        }
        // Without tauri's `custom-protocol` feature the build counts as "dev", and tauri then swaps
        // the Dock icon for the 32×32 icons/icon.png on Ready (after setup), which looks blurry.
        RunEvent::Ready | RunEvent::WindowEvent { event: WindowEvent::ThemeChanged(_), .. } => apply_icon(app),
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.show();
                let _ = w.set_focus();
            }
        }
        _ => {}
    });
}
