# 成章 原生版（Rust）

Python 版的原生重写。安装后 **9.9 MB**（模型另外下载），Mac 用 MLX，Windows 用 llama.cpp，界面沿用 Python 版。

## 结构

```
native/
  crates/cz-core      引擎：VAD 切段 → 边说边识别 → 边说边预填充 → 收尾录音 → 整理/回退
                      （engine.py 的逐条移植）、提示词、推理后端接口（backend.rs）、重采样。与平台无关。
  crates/cz-mlx       macOS 推理后端（mlx-rs）：Qwen3.5（含线性注意力 Metal 内核）、Qwen3-ASR
                      （梅尔特征、音频编码器、Qwen3 解码器）、BPE 分词器、safetensors 读取。
                      模型就是 Python 版用的 mlx-community 权重，两个版本共用一份。
  crates/cz-llama     Windows（及其他平台）推理后端：llama.cpp（子模块）静态链接 + C 封装，GGUF 模型。
  crates/cz-platform  系统接口：快捷键、粘贴、剪贴板、权限、不抢焦点的浮窗。
                      lib.rs 是统一接口，macos.rs 是实现，windows.rs 是占位和实现要点。
  crates/cz-sim       命令行模拟器：按实时速度把 wav 喂给引擎，打印延迟。
  app/                Tauri 外壳，直接加载 ../chengzhang/app/web（界面一行没改）。
  third_party/        llama.cpp（子模块）、mlx-sys（改了构建参数：不编 MLX 的 CPU 后端）。
  tools/              打包、和 Python 版对比的脚本。
```

## 构建

```bash
git submodule update --init native/third_party/llama.cpp
native/tools/bundle_macos.sh          # → native/target/release/成章.app
```

需要：Rust 1.89+、CMake、Xcode 和它的 Metal 工具链（`xcodebuild -downloadComponent MetalToolchain`，约 840 MB，只在编译时用）。

## 版本号与发布

版本号是 `MAJOR.MINOR.PATCH`，只写在 `native/Cargo.toml` 的 `[workspace.package] version` 一处，App 的 Info.plist、「关于」卡片和 DMG 文件名都从这里读。

| 这次发布里有… | 升哪一位 | 例子 |
|---|---|---|
| 不兼容的大改：数据要迁移、旧设置或旧数据不能用了、换了整个实现 | MAJOR | 0.x → 1.0.0 |
| 至少一个新功能（新的设置、快捷键、页面、下载源等） | MINOR，PATCH 归零 | 0.3.2 → 0.4.0 |
| 只有修复和小改进，没有新功能 | PATCH | 0.3.0 → 0.3.1 |

发布步骤：

1. 在 `main` 上改 `native/Cargo.toml` 的版本号，提交 `Release x.y.z`。
2. `native/tools/build_dmg.sh` → `native/target/release/Chengzhang-x.y.z-arm64.dmg`。
3. 打标签 `vx.y.z` 并推送，用 `gh release create` 发布，附上 DMG。发布说明写给用户看：怎么更新、新功能、修复、需要注意的变化。

## 体积

| 手段 | 效果 |
|---|---|
| MLX 打开 JIT：GPU 内核首次使用时编译，只预编一个 1.5 MB 的 metallib（Python 版是 214 MB） | |
| 不编 MLX 的 CPU 后端（全部计算在 GPU 上；safetensors 自己读，因为 MLX 的读取要用 CPU 流） | 可执行文件 18.6 → 7.1 MB |
| 外壳按体积优化（opt-level=s），推理代码保持 -O3 | |
| 分词器自己写（不用 HF tokenizers），重采样自己写 | |

App 构成：可执行文件 7.1 MB + mlx.metallib 1.5 MB + 两个图标 1.8 MB。

## 和 Python 版共用的数据

- `~/Library/Application Support/Chengzhang/` 下的 `history.jsonl`、`dictionary.txt` 两边共用；设置存 `native-settings.json`，第一次启动时继承 Python 版的设置（包括模型选择）。
- 模型在 Hugging Face 缓存（`~/.cache/huggingface/hub`），两个版本共用；原生版下载也写进这个目录。

## Windows

- 推理：`cz-llama`（llama.cpp），`CZ_VULKAN=1` 打开 Vulkan 后端；模型用 GGUF（`app/src/models_gguf.rs`）。
- 系统接口：实现 `crates/cz-platform/src/windows.rs` 里列出的几项。
- 外壳：Tauri 在 Windows 上用系统自带的 WebView2；麦克风走 cpal（WASAPI）。

## 实测（M4 Pro 24 GB，2026-11-13；系统 swap 用到 30 GB 以上，Python 版同时开着）

| | Python 版 0.1.1 | 原生版（MLX） |
|---|---|---|
| 安装后体积 | 491 MB | **9.9 MB** |
| 待机 CPU / 唤醒 | 0.0% / 4 次每分钟 | 0.0% / 1 次每分钟 |
| 内存（phys_footprint，模型加载后） | 5.26 GB | **3.85 GB** |
| 输出 | | 整理结果 4/4 逐字一致；识别 8/9 逐字一致（第 9 条差一个语气字） |

单独测模型（两边交替跑）：

| | Python（mlx-lm / mlx-audio） | 原生版 |
|---|---|---|
| Qwen3.5-4B 生成 | 82–85 tok/s | 81 tok/s |
| Qwen3-ASR 识别 2–32 s 录音 | 80–595 ms | 63–574 ms（每条快 10–20%） |

整条流水线（`tools/compare.py`、`cz-sim`）：这台机器内存压力太大，同一配置两次运行能差 3–8 倍
（松开→首字 293 ms 到 2.6 s），**数字没法用来比较**。较平稳的一次：原生版松开→首字 357–402 ms，
Python 版 259–324 ms。要在退出 Python 版、腾出内存之后重测。

几个和内存压力有关的取舍（`crates/cz-mlx/src/backend.rs`）：
- 只在识别和生成时把模型内存锁定在物理内存（mlx-lm 也是这样）：一直锁定的话，空闲后第一次推理要等 1 s 以上；完全不锁定的话，整理模型的预填充从 30 ms 变成 250 ms。
- MLX 的缓冲区缓存限制在 256 MB，否则预填充留下的临时缓冲会让常驻内存多出约 2 GB。

### llama.cpp 版（Windows 路线）在 Mac 上的数据，供参考

可执行文件 10.0 MB。Qwen3.5-4B 生成速度在内存压力下只有 30–36 tok/s（同时刻 MLX 86），稠密模型不受影响，原因没有查清。
