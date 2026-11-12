# 成章 原生版（Rust）

Python 版的原生重写。目标：安装后 10–20 MB（模型另外下载），性能和 Python 版持平或更好，结构上能接 Windows。

## 结构

```
native/
  crates/cz-core      引擎：VAD 切段 → 边说边识别 → 边说边预填充 → 收尾录音 → 整理/回退
                      （engine.py 的逐条移植），提示词，推理后端接口（backend.rs）。与平台无关。
  crates/cz-llama     推理后端：llama.cpp（子模块，固定版本）静态链接 + 约 300 行 C 封装
                      （shim/cz_llama.cpp）。Qwen3-ASR 走 libmtmd；整理模型每个模式一个缓存前缀，
                      说话时往工作序列里追加。
  crates/cz-platform  系统接口：快捷键（事件 tap）、粘贴、剪贴板、权限、不抢焦点的浮窗。
                      lib.rs 是统一接口，macos.rs 是实现，windows.rs 是占位和实现要点。
  crates/cz-sim       命令行模拟器：按实时速度把 wav 喂给引擎，打印延迟（相当于 otl --simulate）。
  app/                Tauri 外壳。直接加载 ../chengzhang/app/web（界面一行没改），
                      用一段桥接脚本把 window.pywebview.api 转成 Tauri 命令。
                      controller.rs / store.rs / models.rs / mic.rs 对应 Python 版同名模块。
  tools/              打包（bundle_macos.sh）和与 Python 版的对比（compare.py）。
```

## 构建

```bash
git submodule update --init native/third_party/llama.cpp
native/tools/bundle_macos.sh          # → native/target/release/成章.app
native/target/release/cz-sim data/dictation/*.wav   # 引擎延迟（需先 cargo build --release -p cz-sim）
```

需要：Rust 1.89+、CMake、Xcode 命令行工具。模型下载到 `~/Library/Application Support/Chengzhang/gguf/`。

## 体积是怎么压下来的

| 手段 | 效果 |
|---|---|
| 只编 libllama + libmtmd + ggml（不编 common、server、工具） | 推理部分约 9 MB |
| 外壳（Tauri/wry/tao）按体积优化（opt-level=s），推理代码保持 -O3 | 19.2 → 14.4 MB |
| Metal 内核源码 LZFSE 压缩：构建时替换 libggml-metal.a 里的明文源码对象，启动时解压到原来的符号地址，ggml 源码不改 | 14.4 → 10.0 MB |
| 模型权重用 mmap 加载 | 内存压力大时系统能直接丢弃权重页，不进 swap |

## 和 Python 版共用的数据

`~/Library/Application Support/Chengzhang/` 下的 `history.jsonl`、`dictionary.txt` 两边共用；设置单独存在 `native-settings.json`（模型 id 不同），第一次启动时从 Python 版的 `settings.json` 继承偏好。

## Windows

- 推理：同一套 llama.cpp，`CZ_VULKAN=1` 打开 Vulkan 后端（CUDA 运行时太大，不用）；Metal 内核压缩只在 macOS 生效。
- 系统接口：实现 `crates/cz-platform/src/windows.rs` 里列出的几项（低级键盘钩子、SendInput 粘贴、WS_EX_NOACTIVATE 浮窗）。
- 外壳：Tauri 在 Windows 上用系统自带的 WebView2，界面不用改。
- 麦克风：cpal 在 Windows 上走 WASAPI，代码不用改。

## 实测（M4 Pro 24 GB，2026-11-13；系统 swap 已用 27 GB，Python 版同时开着）

| | Python 版 0.1.1 | 原生版 |
|---|---|---|
| 安装后体积 | 491 MB | **11.2 MB**（可执行文件 10.0 MB + 两个图标 1.8 MB） |
| 模型 | MLX：ASR 1.0 GB + Qwen3.5-4B 2.9 GB | GGUF：ASR 1.0 GB + Qwen3.5-4B Q4_K_M 2.7 GB |
| 待机 CPU / 唤醒 | 0.0% / 4 次每分钟 | 0.0% / 12 次每分钟 |
| 内存（phys_footprint） | 5.26 GB | 1.04 GB（另有 3.2 GB 模型权重是文件映射，内存紧张时系统可直接丢弃） |
| 模型加载 | 5.1 s | 3.3–4.4 s（首次运行要编译 Metal 内核，约 9.6 s） |

延迟（`tools/compare.py`，两边交替各跑 2 轮取中位数，单位 ms）：

| 录音 | 松开→首字 Python / 原生 | 松开→完成 Python / 原生 |
|---|---|---|
| d01–d07（6–9 s，一口气说完） | 252–312 / 289–364 | 401–703 / 482–912 |
| d08（2 s 短句，不走整理） | 179 / 126 | 179 / 126 |
| long（32 s，中间有停顿） | 61 / 38 | 1148 / 1888 |

- 识别结果、整理质量和 Python 版一致（9 条里 6 条逐字相同，其余是贪心解码在数值误差下的分叉，两边各有对错）。
- **差距在 Qwen3.5 的生成速度**：llama.cpp 今天早些时候在这台机器上测到 62 tok/s（当时 MLX 约 76），但在当前内存压力下掉到 30–36 tok/s，而同一时刻 MLX 是 86 tok/s。稠密模型（ASR 解码器）不受影响（249 tok/s），只有 Qwen3.5 的线性注意力路径变慢（预填充也从 760 掉到 400 tok/s）。原因还没查清，需要 Metal System Trace。
