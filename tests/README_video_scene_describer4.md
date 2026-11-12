video_scene_describer4.py の通知表現は `--notification-style object_only|situation|action_oriented` で選択する（既定 action_oriented）。

既定の動画サンプリングは10 FPS（0.1秒間隔）。`--frame-interval 1 --sample-fps 10 --temporal-frames 10` に相当し、各秒の0.0〜0.9秒の10枚をまとめて解析する。既定のチャンク設定では6枚ずつ、2枚重複で2チャンクに分割する。実時間で毎秒10回の推論を保証する設定ではない。入力動画が10 FPS未満の場合、同じ元フレームが複数回選ばれる場合がある。

`final_report.json` の observations は全観測、notifications は実際の通知。risks は notifications の互換用別名。通知不要時は両方空配列。observations.json にも内部観測を保存する。最終通知は通常2件以内、優先度の高い直接影響が3件ある場合のみ3件。txt/html の生成は従来どおり。

研究比較は、一度作った final_report.json の notifications の event_id と順序を固定し、各通知の variants から表示文のみ選ぶ。再推論や再選択は不要。object_only は対象・位置、situation は状態、action_oriented は状態と必要な確認行動。過去の接触等は後者2形式で過去形になる。

description は構造化JSONを保持し、descriptions.jsonl と intermediate ログの保存・再開経路に引き継ぐ。プロンプト変更後は解析プロファイルが変わる。古い文章ログには状態の証拠がないため、内部記録に残すが現在の通知に昇格させない。

イベントIDはチャンク・シーン内のIDを名前空間で分離する。観測件数は実人数・台数ではない。カメラ死角、距離、同一性、接触の認識精度はVLMの出力に依存する。実際に送信したフレーム以外の証拠時刻は拒否する。

最新状態と通知の重複防止:
- 各結合画像に元動画のフレーム番号から `front:N/rear:M` というIDを付け、根拠の `evidence_frame_ids` をPython側で設定する。`frame_timeline` の `sec` は前後共通のサンプリング時刻、`front_sec` / `rear_sec` は各カメラのフレーム番号÷FPSによる時刻。低FPS等で片方の画像が繰り返されても、別の画像を根拠に追加しない。
- `latest_frame_sec` / `latest_frame_id` は取得済みの最終画像、`latest_analyzed_frame_sec` は通常解析に成功した最終時刻。`latest_frame_analysis_status` が `unavailable` の場合は古い通行状況を現在の状態として通知しない。過去の接触・のぞき込みの通知は保持する。チャンクの部分失敗で未解析フレームまで成功扱いにしない。
- 障害物比較は通常解析と同じ最終フレーム番号を指定して再取得し、`environment_latest_frame_sec` と `environment.latest_frame_id` に別管理する。障害物比較だけが成功しても、通常解析の成功時刻を進めない。フレーム情報は保存・再開時にも保持する。旧構造化ログには秒数の丸め誤差だけを許容する互換経路がある。解析プロファイルを更新したため、旧キャッシュは自動再利用しない。
- 内部観測は削除せず、通知の選択前に別チャンク・別シーンの候補を照合する。同じ方向・根拠画像・対象種別・位置・行動・状態・通知理由・観測文が一致し、双方の出典に候補が1件ずつある場合にまとめる。同じ出典の複数対象や、曖昧な対応はまとめない。文言が異なる観測や、離れた時間・別カメラをまたぐ追跡は行わない。
- 通常解析と障害物比較の重複は、同じ根拠画像・方向・具体的な対象名・空でない位置が一致する静止障害物に限定する。位置不明時は統合しない。
- 通知の `related_event_ids` に統合元を残す。`observed_at_sec` / `evidence_frame_ids` でも根拠を確認できる。通知枠の最大件数は重複排除後に適用し、3種類の表示形式で同じ通知対象を使用する。

検証: `python3 -m unittest discover -s tests`

JSON応答の耐性改善:
- `--max-tokens` の既定は2048。`--retry-on-length` は既定で有効で、上限到達または構文不正時は2倍ずつ最大3回まで再生成する。`--no-retry-on-length` で増量再試行を無効化できる。
- 項目・根拠フレームの検証エラーは、直前の応答と具体的な修正指示を渡して最大3回まで再生成する。シーン・チャンクで同じ処理を使用する。
- 対象種別の既知の別名（car/person/motorcycle等）は正規化する。未知の種別や証拠のない時刻は引き続き拒否し、壊れたJSONを推測で補完しない。
- プロファイルを更新したため、旧プロファイルの結果は自動再利用されない。
