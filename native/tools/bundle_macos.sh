#!/bin/sh
# Build 成章.app from the release binary: native/target/release/成章.app
set -e
cd "$(dirname "$0")/.."
cargo build --release -p chengzhang
APP="target/release/成章.app"
ASSETS="../chengzhang/app/assets"
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/chengzhang "$APP/Contents/MacOS/chengzhang"
cp "$ASSETS/chengzhang.icns" "$ASSETS/chengzhang-dark.icns" "$APP/Contents/Resources/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>成章</string>
  <key>CFBundleDisplayName</key><string>成章</string>
  <key>CFBundleIdentifier</key><string>app.chengzhang.native</string>
  <key>CFBundleExecutable</key><string>chengzhang</string>
  <key>CFBundleIconFile</key><string>chengzhang</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>成章需要麦克风来听你说话。录音只在本机处理，不会上传。</string>
</dict></plist>
PLIST
codesign --force --sign - --identifier app.chengzhang.native "$APP" >/dev/null 2>&1 || true
du -sh "$APP"
