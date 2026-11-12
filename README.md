<p align="center">
  <img src="chengzhang/app/assets/chengzhang.png" width="148" alt="成章的图标：雾中的太阳与水面上的倒影">
</p>

<h1 align="center">成章</h1>

<p align="center">
  出口成章。在任何 App 里按住一个键说话，松开后，整理好的文字就出现在光标处。<br>
  完全在你的 Mac 上运行。
</p>

<p align="center">
  <em>Chengzhang — private, on-device voice typing for macOS, built on MLX.</em>
</p>

<p align="center">
  <a href="../../releases/latest"><img src="https://img.shields.io/badge/下载-macOS%20dmg-23323A?style=flat-square" alt="下载"></a>
  <img src="https://img.shields.io/badge/Apple%20Silicon-macOS%2014+-23323A?style=flat-square" alt="Apple Silicon, macOS 14+">
  <img src="https://img.shields.io/badge/license-MIT-E8572A?style=flat-square" alt="MIT">
</p>

<p align="center">
  <img src="docs/shot-home-light.jpg" width="49%" alt="白天：莫奈《日出》风格的首页">
  <img src="docs/shot-home-dark.jpg" width="49%" alt="夜间：同一片港口，月亮升起">
</p>

---

说话比打字快得多，但语音识别的原文没法直接用：满是"嗯""那个"，说错了改口，没有标点，也不会分条。成章在识别之后再做一步整理，把口语变成可以直接发出去的文字。从识别到整理，全部用 Apple Silicon 上的 MLX 在本机完成，声音和文字都不会离开你的电脑。

## 它会做什么

- **在任何地方输入**：微信、Slack、邮件、备忘录、代码编辑器都可以。按住快捷键说话，松开后文字直接粘贴到光标处。
- **删口头禅，理解改口**：说"嗯那个，我们周二开会，不对，是周三"，得到的是"我们周三开会"。
- **自动排版**：补标点，中文和英文之间加空格，口述清单会整理成编号列表。
- **顺手翻译**：按住 ⇧ 再按快捷键，说中文，出来的是英文。也可以改成翻成日文、韩文等。
- **个人词典**：把人名、产品名、术语加进去，识别和整理时都会优先写成这些词。
- **自己选模型**：语音识别和文字整理各有几档，可以按速度、准确度和内存来选，也可以填任意 Hugging Face 上的 MLX 模型。

## 下载安装

> 需要搭载 Apple 芯片（M1 及以后）的 Mac，macOS 14 或更新版本。建议 16 GB 内存。

1. 从 [Releases](../../releases/latest) 下载 `Chengzhang-<版本>-arm64.dmg`，打开后把 **成章** 拖进"应用程序"。
2. **第一次打开**：成章没有 Apple 开发者签名，macOS 会拦下它。在"应用程序"里**右键点成章 → 打开**，再点一次"打开"即可。也可以在终端里运行：
   ```bash
   xattr -dr com.apple.quarantine /Applications/成章.app
   ```
3. **授权**：第一次说话时，系统会询问是否允许使用**麦克风**。另外，要在"系统设置 → 隐私与安全性 → **辅助功能**"里打开成章的开关，这样才能在其他 App 里使用快捷键和自动粘贴。
4. **下载模型**：第一次启动时会自动下载默认模型，约 4 GB，首页那幅画里的太阳会随下载和加载慢慢升起。之后不再需要联网。

## 怎么用

| 操作 | 默认快捷键 |
|---|---|
| 说话，松开后上屏 | 按住 **右 ⌥** |
| 说中文，输出译文 | 按住 **⇧**，再按住 **右 ⌥** |
| 放弃这一次 | 说话时按 **esc** |

在「设置」里可以：
- **换快捷键**：右 ⌘、fn，或 ⌃ Space、F5 这样的组合键都可以。
- **换说话方式**："按住说话"或"按一下开始，再按一下结束"。
- **换外观**：选白天或夜间。切换时，画里的太阳会落下、月亮升起。

<p align="center">
  <img src="docs/shot-history-light.jpg" width="49%" alt="记录页：整理后的文字和原话">
  <img src="docs/shot-settings-dark.jpg" width="49%" alt="设置页：快捷键、说话方式、外观">
</p>

## 模型

全部模型来自 Hugging Face 上的 [mlx-community](https://huggingface.co/mlx-community)，在「模型」页切换后会自动下载。

| 用途 | 模型 | 大小 | 说明 |
|---|---|---|---|
| 语音识别 | **Qwen3-ASR 0.6B**（默认） | 1.0 GB | 快，日常口述足够准 |
| | Qwen3-ASR 1.7B | 2.5 GB | 口音重、环境吵或专业词多时更稳 |
| 文字整理 | **Qwen3.5 4B**（默认） | 3.1 GB | 整理得最干净，支持翻译 |
| | Qwen3.5 2B | 1.8 GB | 更快、更省内存 |
| | AgenticASR Refiner 1B | 0.7 GB | 专为口语转书面语微调，很快、忠实于原话，不支持翻译 |
| | Qwen3.5 9B | 6.0 GB | 理解力更强，适合长段口述 |
| | 不整理 | — | 只做语音识别，原样输出 |

## 有多快

在 M4 Pro 上实测，从松开按键算起：

| | 时间 |
|---|---|
| 第一个字出现 | 约 0.1–0.5 秒 |
| 一两句话全部上屏 | 约 0.5–0.8 秒 |
| 30 秒的长段口述全部上屏 | 约 1.1–1.7 秒 |

能这么快，是因为大部分工作在你说话的时候就做完了：成章在你停顿时切出一段，立刻识别，并把识别好的文字提前交给整理模型。所以松开按键后，只剩最后半句需要处理。完整的测试数据和方法见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

## 工作原理

```
麦克风 ─▶ 停顿检测（Silero VAD）─▶ 在停顿处切段 ─▶ 语音识别（Qwen3-ASR，边说边认）
                                                    └─▶ 识别出的文字先交给整理模型
松开按键 ─▶ 认完最后一段 ─▶ 整理模型（Qwen3.5 等）─▶ 粘贴到光标处
```

- 界面是一个 pywebview 窗口，首页那幅画是用代码生成的笔触，启动时一次性烘焙成图片，动画全部交给 GPU。
- 设计思路见 [docs/DESIGN.md](docs/DESIGN.md)。

## 从源码运行

```bash
git clone <本仓库> && cd <仓库目录>
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/chengzhang          # 桌面 App
.venv/bin/otl                 # 命令行版本：按 Enter 开始和结束说话
```

自己打包 dmg（需要 Xcode 命令行工具）：

```bash
.venv/bin/python tools/build_dmg.py      # 输出到 dist/
```

你的记录、词典和设置保存在 `~/Library/Application Support/Chengzhang/`，模型保存在 `~/.cache/huggingface/`。

## 致谢

- 语音识别：[Qwen3-ASR](https://arxiv.org/abs/2601.21337)
- 文字整理：[Qwen3.5](https://huggingface.co/Qwen)；专用整理模型 [AgenticASR Refiner](https://github.com/AnXMuy/AgenticASR)（MiniCPM-5-1B 微调），以及用来评估整理效果的 [AASR-Bench](https://huggingface.co/datasets/Andrew0425/AASR-Bench)
- 推理框架：[MLX](https://github.com/ml-explore/mlx)、[mlx-lm](https://github.com/ml-explore/mlx-lm)、[mlx-audio](https://github.com/Neywa-Labs/mlx-audio)
- 停顿检测：[Silero VAD](https://github.com/snakers4/silero-vad)
- 窗口：[pywebview](https://pywebview.flowrl.com)
- 字体：[霞鹜文楷](https://github.com/lxgw/LxgwWenKai)（装了才会用到，否则使用系统的楷体或宋体）
- 配色和意象：克劳德·莫奈，《印象·日出》（1872）

## 许可

[MIT](LICENSE)。各个模型有自己的许可，以它们在 Hugging Face 上的说明为准。
