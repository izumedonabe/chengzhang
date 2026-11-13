#!/bin/sh
# 成章.app -> native/target/release/成章-<version>.dmg (LZMA-compressed, drag-to-Applications)
set -e
cd "$(dirname "$0")/.."
tools/bundle_macos.sh >/dev/null
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
STAGE=$(mktemp -d)
cp -R "target/release/成章.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="target/release/Chengzhang-$VERSION-arm64.dmg"
rm -f "$DMG"
hdiutil create -volname "成章" -srcfolder "$STAGE" -ov -format ULMO "$DMG" >/dev/null
rm -rf "$STAGE"
ls -la "$DMG"
