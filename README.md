# figure-rename

画像ファイルを `YYYYMMDD_NN_プレゼン|展示|食事|その他_画像の内容.ext` に改名するツールです。分類と画像の内容は、OpenAI 互換 API（`/v1/chat/completions`）で画像入力に対応したモデル（vLLM / llama-swap などで動かすもの）で解析して決めます。

- **プレゼン**: 会議・セッションのプレゼン資料（スライドの撮影、スライドのスクショ、資料の画面）
- **展示**: 展示会・ブースでの実物や実演（製品・機器・基板・ラックなどの写真、展示ポスター）
- **食事**: 食べ物・飲み物（会場内のケータリング、料理、弁当、コーヒー）
- **その他**: 上記に当てはまらないもの（人物、ブレて判別できない写真など）

```bash
export FIGURE_RENAME_MODEL=<モデル名>                  # /v1/models に出る id
export FIGURE_RENAME_HOSTS=gpu-host1,gpu-host2          # リモートで動かす場合の SSH ホスト（任意）

./figure_rename.py ~/Desktop/figs                        # dry-run（改名案を表示するだけ）
./figure_rename.py ~/Desktop/figs --convert --apply      # HEIC を JPEG に変換しつつ改名
./figure_rename.py ~/Desktop/figs --apply                # 改名する
./figure_rename.py ~/Desktop/figs --undo                 # 直前の改名を元に戻す
```

- 必要なもの: `uv`、画像入力に対応したモデルを OpenAI 互換 API で提供するサーバー
- `--convert` は HEIC/HEIF を JPEG（quality 95・EXIF 保持）に変換する。元の HEIC は `.heic-original/` に退避され、`--undo` で復元できます
- `localhost:<port>`（既定 8000、`--port` で変更）でモデルが見つかれば、それをそのまま使います
- 見つからなければ、`--host` / `FIGURE_RENAME_HOSTS` のホストへ SSH トンネル（`ssh -L <port>:localhost:8000`）を自動で張り、終了時に閉じます。モデルが `loaded` のホストを優先し、失敗したら次のホストに切り替えます
- pi-agent 用のスキル: `.pi/skills/figure-rename/SKILL.md`
- 設計判断: `docs/adr/0001-figure-rename-design.md`、`docs/adr/0002-category-in-name.md`、`docs/adr/0003-heic-to-jpeg.md`

オプションは `./figure_rename.py --help` で確認できます。
