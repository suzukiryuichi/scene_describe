#!/usr/bin/env bash
# ============================================================
# R9700 + llama.cpp Vulkan セットアップスクリプト
# Ubuntu 24.04 LTS 用
# ============================================================
set -euo pipefail

echo "=== 0. RDNA4 カーネル確認 ==="
KVER=$(uname -r)
echo "現在のカーネル: $KVER"
KMAJOR=$(echo "$KVER" | cut -d. -f1)
KMINOR=$(echo "$KVER" | cut -d. -f2)
if [ "$KMAJOR" -lt 6 ] || ([ "$KMAJOR" -eq 6 ] && [ "$KMINOR" -lt 12 ]); then
    echo ""
    echo "WARNING: カーネル $KVER は RDNA4 (R9700) 非対応です。"
    echo "以下を実行してカーネルを更新し、再起動後にこのスクリプトを再実行してください:"
    echo ""
    echo "  sudo apt install linux-oem-24.04d linux-firmware"
    echo "  sudo reboot"
    echo ""
    read -p "今すぐカーネルをインストールしますか？ [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get update
        sudo apt-get install -y linux-oem-24.04d linux-firmware
        echo ""
        echo "カーネルをインストールしました。再起動してからこのスクリプトを再実行してください。"
        echo "  sudo reboot"
        exit 0
    else
        echo "カーネル更新をスキップします。Vulkan が R9700 を認識しない可能性があります。"
    fi
fi

echo ""
echo "=== 1. システムパッケージ ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc \
    python3-pip python3-venv \
    v4l-utils

# Vulkan が R9700 を認識しているか確認
echo ""
echo "=== Vulkan デバイス確認 ==="
vulkaninfo --summary 2>/dev/null | grep -E "GPU|deviceName" || echo "WARNING: vulkaninfo でGPUが見つかりません。AMDGPUドライバを確認してください。"

echo ""
echo "=== 2. llama.cpp クローン & Vulkan ビルド ==="
LLAMA_DIR="$HOME/llama.cpp"
if [ -d "$LLAMA_DIR" ]; then
    echo "既存の $LLAMA_DIR を更新..."
    cd "$LLAMA_DIR"
    git pull
else
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
fi

# ビルドディレクトリ
rm -rf build-vulkan
cmake -B build-vulkan \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON
cmake --build build-vulkan --config Release -j$(nproc)

echo ""
echo "=== ビルド完了 ==="
echo "llama-server: $LLAMA_DIR/build-vulkan/bin/llama-server"
echo "llama-bench:  $LLAMA_DIR/build-vulkan/bin/llama-bench"

echo ""
echo "=== 3. Python 環境 ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install opencv-python-headless pillow requests numpy

echo ""
echo "=== 4. モデルダウンロード（手動） ==="
cat << 'EOF'
以下を手動でダウンロードしてください:

# 02_start_server.sh の既定値では、次のファイルを使用します。
# 使用するモデルの配布元からダウンロードし、このパスへ配置してください。
mkdir -p ~/models/Qwen3.6-35B-A3B-UD-IQ4_NL

# メインモデル:
~/models/Qwen3.6-35B-A3B-UD-IQ4_NL/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf

# mmproj:
~/models/Qwen3.6-35B-A3B-UD-IQ4_NL/mmproj-F16.gguf
EOF

echo ""
echo "=== セットアップ完了 ==="
