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
})();
"#;

/// NSWindow* on macOS, HWND on Windows.
pub fn native_handle(w: &tauri::WebviewWindow) -> Option<*mut std::ffi::c_void> {
    #[cfg(target_os = "macos")]
    return w.ns_window().ok();
    #[cfg(windows)]
    return w.hwnd().ok().map(|h| h.0 as *mut std::ffi::c_void);
    #[cfg(not(any(target_os = "macos", windows)))]
    return None;
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
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![api])
        .setup(|app| {
            let handle = app.handle().clone();
            let ctl = Ctl::new(handle.clone());
            app.manage(ctl.clone());

            // CZ_TEST_JS: extra script for automated checks (runs after the page has loaded)
            let test_js = std::env::var("CZ_TEST_JS").map(|js| format!("window.addEventListener('load', () => setTimeout(() => {{ {js} }}, 1500));")).unwrap_or_default();
            let builder = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("成章")
                .inner_size(1120.0, 740.0)
                .min_inner_size(900.0, 620.0)
                .background_color(tauri::window::Color(0xD9, 0xE0, 0xDC, 0xFF))
                .initialization_script(BRIDGE)
                .initialization_script(&test_js);
            // macOS: the painting runs under a transparent title bar
            #[cfg(target_os = "macos")]
            let builder = builder.title_bar_style(tauri::TitleBarStyle::Overlay).hidden_title(true);
            let main = builder.build()?;
            if let Some(h) = native_handle(&main) {
                cz_platform::style_main_window(h);
            }
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
                .build()?;
            let _ = hud.set_ignore_cursor_events(true);
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
        RunEvent::WindowEvent { event: WindowEvent::ThemeChanged(_), .. } => apply_icon(app),
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
