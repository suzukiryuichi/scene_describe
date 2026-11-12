# Webcam Scene Describer

ウェブカメラで周囲を撮影し、Qwen VLMで周辺状況を日本語で分析します。既定では直近1秒を10 FPSで取得し、時系列フレームをまとめて分析します。

既定では `./02_start_server.sh` で起動したローカル llama-server に画像を送り、Qwen VLMで説明文を生成します。リアルタイム性より分析内容を優先する設定で、連続実行時の既定撮影間隔は10秒、VLM APIタイムアウトは120秒です。

## 1. Qwen VLM サーバーを起動

別ターミナルで実行します。

```bash
cd /home/suzuki/vlm-analysis
./02_start_server.sh
```

既定では次のモデルを使います。

- `/home/suzuki/models/Qwen3.6-35B-A3B-UD-IQ4_NL/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`
- `/home/suzuki/models/Qwen3.6-35B-A3B-UD-IQ4_NL/mmproj-F16.gguf`

## 2. ウェブカメラをVLMで分析

1枚だけ確認する場合:

```bash
cd /home/suzuki/vlm-analysis
source venv/bin/activate
python webcam_scene_describer.py --count 1
```

継続して分析する場合:

```bash
python webcam_scene_describer.py --continuous
```

撮影間隔を変える場合:

```bash
python webcam_scene_describer.py --continuous --interval 30
```

## 出力

- 標準出力: 最新の説明文
- `scene_logs/images/`: 撮影画像
- `scene_logs/descriptions.jsonl`: 説明ログ
- `scene_logs/YYYYMMDDHHMM.html`: 起動時刻ごとの画像と説明を見られる HTML
- `scene_logs/index.html`: 最新HTMLへの互換用コピー

## 撮影済み動画を分析

動画ファイルを指定して、10 FPSで取得した1秒分の時系列フレームをまとめて分析し、注意点のみを出力します。
VLM推論は既定で1秒ごとに1回実行します。
先に `./02_start_server.sh` を起動してから実行します。

```bash
python video_scene_describer.py --video path/to/movie.mp4
```

一部の区間だけ分析する場合:

```bash
python video_scene_describer.py --video path/to/movie.mp4 --start-sec 10 --end-sec 60 --frame-interval 2
```

動き確認用の取得FPSや、1回の分析に渡すフレーム数を変える場合:

```bash
python video_scene_describer.py --video path/to/movie.mp4 --sample-fps 5 --temporal-frames 10
```

詳細説明を出したい場合:

```bash
python video_scene_describer.py --video path/to/movie.mp4 --vlm-prompt-mode detailed --max-samples 5
```

HTMLファイル名を明示する場合:

```bash
python video_scene_describer.py --video path/to/movie.mp4 --html-file 202605011625.html
```

## 前後の撮影済み動画を分析

フロントカメラ動画とバックカメラ動画を同時刻で結合し、危険箇所を最大3件まで出力します。
各注意点は `【前方】...`、`【左後方】...`、`【右後方】...`、`【後方】...` などの短い1行で出力します。

```bash
python video_scene_describer2.py --front-video path/to/front.mp4 --back-video path/to/back.mp4
```

時間を短縮したい場合は、チャンク統合の追加推論を使わない既定の `--chunk-merge-mode local` のまま実行します。
統合も VLM に任せたい場合だけ、従来相当の `--chunk-merge-mode vlm` を指定してください。
さらに速く粗く確認する場合は、渡すフレーム数や画像サイズを下げます。

```bash
python video_scene_describer2.py --front-video path/to/front.mp4 --back-video path/to/back.mp4 --sample-fps 4 --temporal-frames 4 --max-image-side 512 --jpeg-quality 70
```

## 停車時に特化して前後の撮影済み動画を分析

停車中の自車について、発進可否、停車継続、乗降・ドア開閉時の安全確認に関する注意点を最大3件まで出力します。

```bash
python video_scene_describer3.py --front-video path/to/front.mp4 --back-video path/to/back.mp4
```
