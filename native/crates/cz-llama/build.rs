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
            // Vulkan keeps the installer small (the CUDA runtime alone is ~400 MB);
            // the GPU driver provides vulkan-1.dll at run time.
            let vk = if std::env::var("CZ_VULKAN").is_ok() { "ON" } else { "OFF" };
            cfg.define("GGML_VULKAN", vk);
            // CPU fallback: AVX2-era x86 (2013+), not just the build machine's CPU
            cfg.define("GGML_AVX", "ON").define("GGML_AVX2", "ON").define("GGML_FMA", "ON").define("GGML_F16C", "ON");
            // match Rust's dynamic CRT (/MD)
            cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
        }
        _ => {}
    }
    let dst = cfg.build();

    for sub in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={}", dst.join(sub).display());
    }
    let mut shim = cc::Build::new();
    if target_os == "macos" && std::env::var("CZ_RAW_METAL").is_err() {
        pack_metal_sources(&dst);
        shim.define("CZ_PACKED_METAL", "1");
    }
    // our shim first (it references the rest)
    shim
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
    if target_os == "windows" && dst.join("lib/ggml-vulkan.lib").exists() {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            println!("cargo:rustc-link-search=native={sdk}/Lib");
        }
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    }
    for lib in ["ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if target_os == "windows" {
        for lib in ["advapi32", "user32", "kernel32", "shell32", "ole32"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
    println!("cargo:rerun-if-env-changed=CZ_VULKAN");
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

// ----------------------------------------------------------------------------- Metal sources

#[cfg(target_os = "macos")]
#[link(name = "compression")]
extern "C" {
    fn compression_encode_buffer(dst: *mut u8, dst_size: usize, src: *const u8, src_size: usize,
                                 scratch: *mut u8, algorithm: u32) -> usize;
}
#[cfg(target_os = "macos")]
const COMPRESSION_LZFSE: u32 = 0x801;

/// ggml embeds its Metal kernels as plain-text source, one copy of the shared headers per
/// kernel group: ~5 MB of the binary. Replace those objects in libggml-metal.a with
///   - zero-filled `ggml_metallib_<kind>_{start,end}` symbols of the same size (no file bytes)
///   - the same text LZFSE-compressed (~0.3 MB)
///   - cz_metal_unpack(), which the shim calls before ggml starts, to fill them in.
/// ggml itself is untouched and sees exactly the bytes it would have embedded.
#[cfg(not(target_os = "macos"))]
fn pack_metal_sources(_dst: &std::path::Path) {}

#[cfg(target_os = "macos")]
fn pack_metal_sources(dst: &std::path::Path) {
    let gen = dst.join("build/ggml/src/ggml-metal/autogenerated");
    let mut kinds: Vec<String> = std::fs::read_dir(&gen)
        .expect("ggml-metal autogenerated dir")
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_prefix("ggml-metal-embed-")?.strip_suffix(".metal").map(String::from))
        .collect();
    kinds.sort();
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("metalpack");
    std::fs::create_dir_all(&out).unwrap();
    let mut asm = String::from(".bss\n");
    let mut data = String::from(".section __TEXT,__const\n");
    let mut table = String::from("#include <compression.h>\n#include <stddef.h>\n");
    let mut entries = String::new();
    let (mut raw_total, mut packed_total) = (0usize, 0usize);
    for (i, kind) in kinds.iter().enumerate() {
        let src = std::fs::read(gen.join(format!("ggml-metal-embed-{kind}.metal"))).unwrap();
        let mut buf = vec![0u8; src.len() + 4096];
        let n = unsafe {
            compression_encode_buffer(buf.as_mut_ptr(), buf.len(), src.as_ptr(), src.len(), std::ptr::null_mut(), COMPRESSION_LZFSE)
        };
        assert!(n > 0, "LZFSE failed for {kind}");
        buf.truncate(n);
        let blob = out.join(format!("{kind}.lzfse"));
        std::fs::write(&blob, &buf).unwrap();
        raw_total += src.len();
        packed_total += n;
        let sym = kind.replace('-', "_");
        asm += &format!(".globl _ggml_metallib_{sym}_start\n.p2align 4\n_ggml_metallib_{sym}_start:\n.space {}\n", src.len());
        asm += &format!(".globl _ggml_metallib_{sym}_end\n_ggml_metallib_{sym}_end:\n.space 1\n");
        data += &format!(".globl _cz_blob_{i}\n_cz_blob_{i}:\n.incbin \"{}\"\n", blob.display());
        table += &format!("extern char ggml_metallib_{sym}_start[]; extern const unsigned char cz_blob_{i}[];\n");
        entries += &format!("  {{ggml_metallib_{sym}_start, {}, cz_blob_{i}, {n}}},\n", src.len());
    }
    table += "static const struct { char *dst; size_t n; const unsigned char *blob; size_t blob_n; } T[] = {\n";
    table += &entries;
    table += "};\n/* returns the number of kernel groups that failed to unpack */\n\
int cz_metal_unpack(void) {\n  int bad = 0;\n  for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)\n\
    if (compression_decode_buffer((uint8_t *) T[i].dst, T[i].n, T[i].blob, T[i].blob_n, NULL, COMPRESSION_LZFSE) != T[i].n) bad++;\n\
  return bad;\n}\n";
    std::fs::write(out.join("pack.s"), asm + &data).unwrap();
    std::fs::write(out.join("unpack.c"), table).unwrap();
    cc::Build::new().file(out.join("pack.s")).file(out.join("unpack.c")).opt_level(2).compile("cz_metal_pack");
    println!("cargo:rustc-link-lib=compression");

    // drop ggml's own plain-text copies from the archive
    let lib = dst.join("lib/libggml-metal.a");
    let members = std::process::Command::new("ar").arg("t").arg(&lib).output().unwrap();
    let embed: Vec<String> = String::from_utf8_lossy(&members.stdout)
        .lines().filter(|m| m.starts_with("ggml-metal-embed-")).map(String::from).collect();
    if !embed.is_empty() {
        let ok = std::process::Command::new("ar").arg("d").arg(&lib).args(&embed).status().unwrap();
        assert!(ok.success(), "ar d failed");
        let _ = std::process::Command::new("ranlib").arg(&lib).status();
    }
    println!("cargo:warning=Metal kernels: {:.1} MB source packed into {:.2} MB", raw_total as f64 / 1e6, packed_total as f64 / 1e6);
}
