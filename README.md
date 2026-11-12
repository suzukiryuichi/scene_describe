# scene_describe

配送ドライバー向けの、前方・後方映像を使ったAI安全確認支援の研究用コードです。通知はドライバー自身による直接の安全確認を前提とします。

現在の改善版は [`video_scene_describer5.py`](video_scene_describer5.py) です。使い方、設定、設計、検証結果と制約は [5版の説明](README_scene_describer5.md) を参照してください。

```bash
pip install opencv-python requests pillow numpy
python3 video_scene_describer5.py --front-video path/to/front.mp4 --back-video path/to/rear.mp4 --notification-style object_only
python3 -m unittest discover -s tests
```

VLM推論には、従来の設定に対応する llama-server が必要です。[従来のセットアップ](README_scene_describer.md) も参照してください。撮影動画・モデル・解析ログはこのリポジトリに含めていません。
