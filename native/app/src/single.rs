//! One 成章 at a time. Two running copies (say one opened from a mounted installer and
//! one from Applications) would both listen to the hotkey; in toggle mode each press then
//! stops one and starts the other, and you get two interleaved, garbled dictations.
//!
//! The first instance holds an exclusive lock on <app dir>/instance.lock and writes its
//! location into it. A later one finds the lock taken, asks macOS to open that location
//! (which brings the running copy's window forward) and exits.
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

pub enum Instance {
    /// We are the only one; keep the lock for the life of the process.
    First,
    /// Another 成章 is running at this location.
    Second(Option<String>),
}

/// The .app bundle we run from (or the executable itself outside a bundle).
fn own_location() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let bundle = exe.ancestors().nth(3).filter(|p| p.extension().is_some_and(|e| e == "app"));
    Some(bundle.map(|p| p.to_path_buf()).unwrap_or(exe))
}

pub fn claim() -> Instance {
    let dir = crate::store::app_dir();
    let _ = std::fs::create_dir_all(&dir);
    let Ok(mut f) = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(dir.join("instance.lock"))
    else {
        return Instance::First; // can't tell: rather run than refuse
    };
    match f.try_lock() {
        Ok(()) => {
            let _ = f.set_len(0);
            if let Some(p) = own_location() {
                let _ = f.write_all(p.to_string_lossy().as_bytes());
                let _ = f.flush();
            }
            // the lock lives as long as the file stays open: for the whole process
            Box::leak(Box::new(f));
            Instance::First
        }
        Err(TryLockError::WouldBlock) => {
            let mut s = String::new();
            let _ = f.read_to_string(&mut s);
            let s = s.trim().to_string();
            Instance::Second((!s.is_empty()).then_some(s))
        }
        Err(TryLockError::Error(_)) => Instance::First,
    }
}

/// Bring the running copy forward. At most one request per 10 s, so a copy that the
/// system doesn't know is running (started outside Finder) can't make us loop.
pub fn show_other(location: &str) {
    let marker = crate::store::app_dir().join("instance.reopen");
    let recent = std::fs::metadata(&marker)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| SystemTime::now().duration_since(t).ok())
        .is_some_and(|age| age < Duration::from_secs(10));
    if recent {
        return;
    }
    let _ = File::create(&marker);
    #[cfg(target_os = "macos")]
    {
        // a running app gets a "reopen" event, which shows its main window
        let _ = std::process::Command::new("/usr/bin/open").arg(location).status();
    }
    #[cfg(not(target_os = "macos"))]
    {
        // the running copy's main window (hidden if its close button was pressed: on
        // Windows there is no Dock icon, so starting 成章 again is how you get it back)
        let _ = location;
        cz_platform::bring_to_front("成章");
    }
}
