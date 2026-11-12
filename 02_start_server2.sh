#!/usr/bin/env bash
# ============================================================
# llama-server 起動スクリプト
# R9700 + Qwen3.5-35B-A3B VLM 用
# ============================================================
set -euo pipefail

LLAMA_SERVER="$HOME/llama.cpp/build-vulkan/bin/llama-server"

# ---- モデルパス（環境に合わせて変更） ----
MODEL="${MODEL:-$HOME/models/Qwen3.5-27B-Q8_0.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/mmproj-27B-f16.gguf}"

# ---- パラメータ ----
PORT="${PORT:-8080}"
CTX=65536          # 画像認識最低限
NGL=99            # 全レイヤー GPU オフロード
BATCH=2048
UBATCH=256
PARALLEL=1        # 単一ストリーム（ウェブカム1台）
SLOTS=1

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: $LLAMA_SERVER が見つかりません。01_setup.sh を先に実行してください。"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: モデルが見つかりません: $MODEL"
    echo "MODEL 環境変数でパスを指定するか、01_setup.sh のダウンロード手順を実行してください。"
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: mmproj が見つかりません: $MMPROJ"
    exit 1
fi

echo "=== llama-server 起動 ==="
echo "Model:   $MODEL"
echo "mmproj:  $MMPROJ"
echo "Context: $CTX"
echo "Port:    $PORT"
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    -ngl "$NGL" \
    -c "$CTX" \
    --jinja \
    --reasoning-budget 0 \
    -b "$BATCH" \
    -ub "$UBATCH" \
    --flash-attn on \
    -np "$PARALLEL" \
    --host 0.0.0.0 \
    --port "$PORT"
