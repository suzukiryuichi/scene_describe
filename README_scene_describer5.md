# video_scene_describer5

`video_scene_describer4.py` を複製し、映像入力・VLM接続・チャンク並列処理・再試行・ログ再開・障害物比較・HTML出力を維持した改善版です。4版は変更していません。新しい依存ライブラリや外部APIは追加していません。

## 起動

従来の llama-server を起動した環境で実行します。

```bash
python3 video_scene_describer5.py \
  --front-video video/front.mp4 \
  --back-video video/back.mp4 \
  --notification-config notification_config5.json \
  --notification-style object_only \
  --output-dir scene_logs_v5
```

通知形式は `object_only` / `situation` / `action_oriented`。従来同様、既定は `action_oriented` です。既定のサンプリングは10 FPS、1シーン10枚、1秒間隔で、6枚ずつ2枚重複のチャンクに分けます。これは推論が実時間で完了する保証ではありません。

## 処理構成と変更点

4版は、前後動画の同期抽出 → 前方上半分と後方の結合 → 時刻付きVLM解析 → チャンク統合 → 初回／最新画像の障害物比較 → `build_final_report` 内での選択・文章生成という構成でした。

5版では、この入出力フローを維持しながら以下を分離しています。

1. **観測**: `validate_observations` / `validate_frame_evidence`。VLMには対象・位置・行動・根拠時刻を出力させ、通知は選ばせません。バイク・原付を `motorcycle` として独立させ、接触、滞留、物体移動、道路閉塞、出現などを記録します。正規の応答形式は `observation_schema5.json` です。Python側で型・列挙値・有限時刻・時刻範囲・イベント参照・実際に送った画像との一致も検証します。不正応答は既存の修正指示付き再生成処理で最大3回試行します。
2. **時系列統合**: `integrate_temporal_events`。根拠が重なる観測だけを統合し、最初／最後の時刻、状態履歴、状態変化、現在の存在の確実性、観測IDを保持します。接近・通過・離脱の根拠が1画像だけ、同じカメラ画像の反復、または画像間隔が長い場合は `unknown` に補正します。補正も `state_changes` に理由を記録します。
3. **通知選択**: `select_notifications`。発進安全、後方、交通弱者、鮮度、重要な変化、気づきにくさを設定した重みで加点します。全イベントについて得点の内訳、候補理由、選択結果、未採用理由を残します。通常の存在だけで高危険度と断定する処理はありません。
4. **通知生成**: `generate_notifications` / `notification_variants`。選択済みイベントから3形式の簡潔な日本語を生成します。モデルが出した自由文の警告や確認行動は表示に流用しません。方向と対象の制御された語彙を使います。

## 設定と研究比較

設定値は `notification_config5.json` に集約しています。指定した設定は全キーが必要で、不正な値・未知のキーはエラーにします。実験時は設定ファイルをコピーして `--notification-config` に指定してください。

| キー | 用途 |
| --- | --- |
| `preferred_notifications` | 通常件数。既定2件 |
| `max_notifications` | 最大件数。0～3、既定3件 |
| `third_notification_min_score` | 通常件数を超えて採用するための最低得点 |
| `minimum_score` | 候補として採用可能な最低得点 |
| `current_max_age_sec` | 対象別の元カメラ画像の許容経過秒 |
| `max_motion_gap_sec` | 移動状態を採用する根拠画像間の最大間隔 |
| `historical_max_age_sec` | 任意の過去変化通知の有効期間 |
| `notify_historical_changes` | 通常の過去変化の通知を許可するか。既定false |
| `weights` | 6つの評価軸の非負の重み |

鮮度の閾値内でも、取得済み最新画像の根拠がなければ「今いる」とは通知しません。通常解析の最新画像が失敗した場合や、障害物比較の方が新しい場合にも古い対象を現在扱いしません。前後カメラの片方が同じフレームを繰り返した場合は、その方向の実フレーム時刻で鮮度を評価します。

明確に離脱した通常通行は通知しません。過去の接触事故・物体移動は重要な出来事として保持でき、通知する際はどの形式でも過去形になります。既定では通常の過去の通過を通知しません。`notify_historical_changes` を有効にしても、明確な離脱は抑制し、`change_type` の根拠がある変化だけを候補にします。

同じ最終JSONの `notifications` のIDと順番を固定し、`variants["object_only"]` などの文面だけ切り替えれば、再推論・再選択なしで比較できます。過去の重要イベントでは、現在の存在を誤認させないため、3形式の文面が一部共通になる場合があります。

## 出力と互換性

従来の `final_report.json` / `.txt` / `.html`、`observations.json`、シーン別HTML、中間JSONLを維持しています。`risks` は `notifications` と同じ配列で、`direction` / `message` を引き続き利用できます。出力先の既定値は `scene_logs_v5` です。観測プロファイルを変更しており、4版の中間結果は自動再利用しません。

最終JSONに次を追加しています。

- `observations`: 全内部観測。選択のために削除しません。
- `events`: 統合結果。`observation_ids`、根拠フレーム、`state_history`、`state_changes`、`current_presence`、`age_sec` を含みます。
- `notification_candidates`: 全イベントの評価。`score_contributions`、`selected`、`exclusion_reason`、重複先の `duplicate_of` から採否を確認できます。
- `security_events`: 接近、滞留、車体接触、のぞき込みなどの客観的な防犯観測。交通通知枠には入れません。犯罪意図を判定しません。
- `metadata`: 実際の設定、形式、上限、スキーマ版、外部コンテキストの出典。

根拠が共有され、方向・対象・位置・観測文などが一致する場合に限り観測を統合します。別カメラ間、時間が離れた観測、対応が曖昧な対象を同一対象とは判定しません。そのため、前後に同じ種類が写っても、同一性の証拠なしには1イベントにまとめません。同じ方向の同じ表示文は通知段階でまとめ、内部イベントと重複除外理由は残します。

障害物比較で `initial_state=absent` が確認された場合は、新たな出現として記録します。障害物があることと道路閉塞は区別し、`blocks_path=true` の根拠があるときだけ「道路をふさいでいます」と表現します。

## 地域・時刻・天候の拡張

`ContextProvider.facts_at(video_sec)` と、出典を必須とする `ContextFact` を設けています。現在は選択・推論には使わず、`metadata.external_context` に分離して保存します。既存の `--location-info` / `--weather-info` / `--time-info` は利用者入力として `source=user_supplied_cli` で保存し、観測VLMには渡しません。映像にない地域情報や、その情報を根拠とした子ども・危険対象は生成しません。外部API接続はありません。

## 検証と制約

```bash
python3 -m unittest discover -s tests
python3 video_scene_describer5.py --help
```

5版のテストでは、要求された10ケースに加えて、鮮度設定、最新画像の失敗、不確かな観測の抑制、重大な過去イベント、防犯カテゴリの分離、モーターサイクルの分類、実フレームの反復、JSON修正再試行を検証します。模擬VLMを使い、一時生成した前後動画の読み込みからチャンク解析、障害物比較、全レポート出力、ログからの再開まで通しています。

実VLMを接続した映像認識の精度・処理時間や、実運転での有用性は未評価です。観測内容そのものの正しさはVLMと画質に依存します。前方上半分の切り出しと後方映像という既存の観測範囲も維持しており、側方・死角は確認できません。長時間の追跡や正確な人数・台数は保証しません。

今後は、アノテーション付き実映像による誤検出・見逃し・鮮度の評価、通知の読解時間と確認行動の比較、重みと閾値の感度分析、客観的な同一性根拠を備えた追跡、出典付き外部データを用いた条件の評価が必要です。設定の既定値は検証前の暫定値です。通知が0件でも安全を保証せず、ドライバー自身の直接確認を前提とします。
