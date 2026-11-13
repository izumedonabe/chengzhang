成章（Windows 预览版）

在任何地方用语音输入：按住快捷键说话，松开后，整理好的文字会粘贴到光标处。
语音识别和整理都在本机运行，不上传录音。


怎么打开
- 双击 Chengzhang.exe。程序没有签名，Windows 可能弹出"已保护你的电脑"，
  点"更多信息"→"仍要运行"。
- 第一次打开会自动下载模型（默认约 3.7 GB）。在国内连不上 Hugging Face 时，
  会自动改从 ModelScope 下载；也可以在"设置 → 下载"里固定下载源。
- 打不开、只有空白窗口时：安装 Microsoft Edge WebView2 Runtime
  （https://go.microsoft.com/fwlink/p/?LinkId=2124703）。Windows 11 一般自带。


怎么用
  按住 右 Alt              录音，松开后输出
  按住 Shift + 右 Alt      录音并翻译（默认译成英文）
  Esc                      取消本次录音；整理太慢时也可以放弃
  Ctrl + Alt + C           识别有错时，在输入框里改好再按：把改正后的人名、术语加进词典

快捷键、录音方式（按住 / 按一下开始再按一下结束）、翻译语言都可以在"设置"里改。
人名、产品名、专业术语可以加到"词典"页。


系统要求
- Windows 10 或 11（64 位），建议 16 GB 内存。
- 有显卡（包括集成显卡）时用 Vulkan 加速；更新显卡驱动通常能更快。

用户数据（设置、词典、记录、模型）保存在 %APPDATA%\Chengzhang\。
项目主页：https://github.com/izumedonabe/chengzhang
