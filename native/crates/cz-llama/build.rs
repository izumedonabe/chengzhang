//! Builds the pinned llama.cpp (third_party/llama.cpp) as static libraries with only
//! what we use (libllama, ggml + the platform GPU backend, libmtmd), plus our C shim.
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama = root
        .join("../../third_party/llama.cpp")
        .canonicalize()
        .expect("third_party/llama.cpp missing: run `git submodule update --init`");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();

    let mut cfg = cmake::Config::new(&llama);
    cfg.profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_MTMD", "ON")
        .define("MTMD_VIDEO", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("LLAMA_SUBPROCESS", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_NATIVE", "OFF");
    match target_os.as_str() {
        "macos" => {
            cfg.define("GGML_METAL", "ON")
                .define("GGML_METAL_EMBED_LIBRARY", "ON")
                .define("GGML_BLAS", "OFF");
        }
        "windows" => {
            // Vulkan keeps the installer small (the CUDA runtime alone is ~400 MB)
            let vk = if std::env::var("CZ_VULKAN").is_ok() { "ON" } else { "OFF" };
            cfg.define("GGML_VULKAN", vk);
        }
        _ => {}
    }
    let dst = cfg.build();

    for sub in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={}", dst.join(sub).display());
    }
    // our shim first (it references the rest)
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .file(root.join("shim/cz_llama.cpp"))
        .include(llama.join("include"))
        .include(llama.join("ggml/include"))
        .include(llama.join("tools/mtmd"))
        .compile("cz_llama_shim");

    for lib in ["mtmd", "llama"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=static=ggml-metal");
        if dst.join("lib/libggml-blas.a").exists() {
            println!("cargo:rustc-link-lib=static=ggml-blas");
        }
    }
    for lib in ["ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if target_os == "macos" {
        for fw in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        println!("cargo:rustc-link-lib=c++");
        // ggml-metal uses @available(), which needs ___isPlatformVersionAtLeast from compiler-rt
        if let Ok(out) = std::process::Command::new("clang").arg("--print-resource-dir").output() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("cargo:rustc-link-search=native={dir}/lib/darwin");
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
        }
    }
    println!("cargo:rerun-if-changed=shim/cz_llama.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}
