成章 Windows 预览版
====================

在任何地方用语音输入：按住快捷键说话，松开后整理好的文字会粘贴到光标处。
所有模型都在本机运行，除了第一次下载模型，不需要联网。

这是预览版，还没有在很多电脑上测试过。遇到问题请到
https://github.com/izumedonabe/chengzhang/issues 反馈。


运行
----

1. 解压后双击 Chengzhang.exe。程序不用安装，放在哪个文件夹都可以。
2. 程序没有签名，Windows 可能会弹出"Windows 已保护你的电脑"。
   点"更多信息"→"仍要运行"。
3. 第一次启动会下载语音识别和整理模型（默认约 3.7 GB），从 Hugging Face 下载。
   在国内访问 Hugging Face 不稳定时可能下载失败，需要能访问 Hugging Face 的网络。


使用
----

  按住 右 Alt          录音，松开后输出
  按住 Shift + 右 Alt  录音并翻译（默认译成英文）
  Esc                  取消这一次录音
  Ctrl + Alt + C       识别有错时，在输入框里改好再按：
                       把改正后的人名、术语加进词典

快捷键、录音方式、翻译语言可以在"设置"里修改。


系统要求
--------

- Windows 10 或 11，64 位
- Microsoft Edge WebView2 运行时（Windows 11 自带；Windows 10 一般已经装过，
  没有的话从 https://developer.microsoft.com/microsoft-edge/webview2/ 下载）
- 有显卡时用 Vulkan 加速，没有也能用 CPU 运行，但会慢一些
- 建议 16 GB 内存


数据放在哪里
------------

设置、词典、记录和模型都在 %APPDATA%\Chengzhang\ 下（模型在其中的 gguf 文件夹）。
删掉这个文件夹和 Chengzhang.exe 就完全卸载了。
