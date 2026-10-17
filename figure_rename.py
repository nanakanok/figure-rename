#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10",
#   "pillow-heif>=0.16",
# ]
# ///
"""
画像ファイルを YYYYMMDD_NN_プレゼン|展示|食事|その他_画像の内容.ext に改名する。

- 日付: EXIF DateTimeOriginal（無ければ DateTime → mtime）
- NN: フォルダ × 日付ごとに 01 から撮影時刻順。整形済みファイルは飛ばし、その続きから振る
- 分類: 画像解析モデルが プレゼン（スライド・資料の撮影やスクショ）/ 展示（ブース・実物・ポスター）/ 食事（料理・飲み物）/ その他 のいずれか付ける
- 画像の内容: OpenAI 互換 API で画像入力に対応したモデル（vLLM / llama-swap など）で解析。
  リモートのサーバーには ssh -L <port>:localhost:8000 のトンネルを自動で張って使う
- 既定は dry-run。--apply で改名、--undo で直前の実行分を戻す

設計は docs/adr/0001-figure-rename-design.md を参照。
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # HEIC が読めないだけで他は動かす
    pass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".gif"}
HEIF_EXTS = {".heic", ".heif"}
ORIGINAL_DIR = ".heic-original"  # 変換で失う元ファイルを退避する先（対象収集からは除外される）
JPEG_QUALITY = 95
FORMATTED_RE = re.compile(r"^(\d{8})_(\d{2,})_.+$")
PLAN_FILE = ".figure-rename-plan.json"
LOG_FILE = ".figure-rename-log.jsonl"
DEFAULT_HOSTS = [h for h in re.split(r"[,\s]+", os.environ.get("FIGURE_RENAME_HOSTS", "")) if h]
DEFAULT_MODEL = os.environ.get("FIGURE_RENAME_MODEL")
REMOTE_PORT = 8000
MAX_TITLE = 20
CATEGORIES = ("プレゼン", "展示", "食事", "その他")
CATEGORY_HINTS = {
    "プレゼン": ("プレゼン", "スライド", "資料", "画面", "スクショ", "セッション", "講演"),
    "展示": ("展示", "ブース", "ポスター", "実演", "製品", "ラック", "基板", "ハードウェア"),
    "食事": ("食事", "料理", "食べ物", "飲食", "ケータリング", "弁当", "飲み物", "コーヒー", "レストラン", "カフェ"),
}
MAX_EDGE = 1568
TIMEOUT = 240  # llama-swap がモデルを読み込む時間を見込む

SYSTEM_PROMPT = """あなたは画像ファイルに名前を付けるアシスタントです。画像を見て、ファイル名に使う短い日本語のタイトルと、画像の分類を決めてください。

分類（category は次から 1 つ選ぶ）:
- プレゼン: 会議・セッションのプレゼン資料。スクリーンに写したスライド、スライドのスクリーンショット、資料の画面、ホワイトボードの解説
- 展示: 展示会・ブースでの実物や実演。製品・機器・基板・ラック・モジュールなどの写真、展示ポスター、手元で撮ったハードウェア
- 食事: 食べ物・飲み物。会場内のケータリング、料理、弁当、コーヒーなど
- その他: 上記のどれにも当てはまらない。人物、意味の取れない写真など

タイトルのルール:
- 画像内に見出し（スライドのタイトル、図のキャプション、ウィンドウやアプリの名前など）が読み取れる場合は、それを優先して使う
- 見出しが無い場合は「主題＋図の種類」で名付ける。図の種類の例: 構成図、グラフ、表、スクショ、写真、フロー図、画面
- 20 文字以内。日本語で書く。製品名・技術用語・固有名詞（SONiC, BGP など）は原語のまま
- 日付・連番・拡張子・分類名・記号（/ \\ : * ? " < > |）は含めない
- 出力は JSON のみ: {"title": "タイトル", "category": "プレゼン|展示|食事|その他", "reason": "名付けと分類の根拠を一文で"}"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 名前の整形


def sanitize_title(title: str) -> str:
    t = unicodedata.normalize("NFC", title).strip()
    t = re.sub(r'[/\\:*?"<>|\x00-\x1f\x7f]', "", t)
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-_. ")
    return t or "無題"


def truncate(title: str) -> str:
    return title[:MAX_TITLE].rstrip("-_. ") or "無題"


def normalize_category(value: object) -> str:
    """モデルの返した文字列を CATEGORIES のいずれかに寄せる"""
    s = unicodedata.normalize("NFC", str(value or "")).strip()
    if s in CATEGORIES:
        return s
    for cat, hints in CATEGORY_HINTS.items():
        if any(h in s for h in hints):
            return cat
    return "その他"


# ---------------------------------------------------------------- 対象の収集


@dataclass
class Target:
    path: Path
    size: int
    mtime: float
    taken: dt.datetime  # 撮影日時（無ければ mtime）
    date_source: str  # "exif" | "mtime"
    out_ext: str = ""  # 変換後の拡張子。空なら元の拡張子の小文字を使う

    @property
    def ext(self) -> str:
        return self.out_ext or self.path.suffix.lower()

    @property
    def date(self) -> str:
        return self.taken.strftime("%Y%m%d")


def read_taken(path: Path) -> tuple[dt.datetime, str]:
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal / DateTime
        if raw:
            return dt.datetime.strptime(str(raw).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S"), "exif"
    except Exception:
        pass
    return mtime, "mtime"


def is_image(p: Path) -> bool:
    return p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS


def collect(paths: list[str], recursive: bool) -> dict[Path, list[Path]]:
    """フォルダごとに対象ファイル（未整形のもの）をまとめる"""
    by_dir: dict[Path, list[Path]] = {}
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files = [f for f in it if is_image(f) and not any(part.startswith(".") for part in f.relative_to(p).parts[:-1])]
        elif p.is_file():
            if not is_image(p):
                log(f"skip (対象外の拡張子): {p}")
                continue
            files = [p]
        else:
            log(f"skip (見つからない): {p}")
            continue
        for f in files:
            if FORMATTED_RE.match(f.stem):
                if p.is_file():
                    log(f"skip (整形済み): {f}")
                continue
            by_dir.setdefault(f.parent, [])
            if f not in by_dir[f.parent]:
                by_dir[f.parent].append(f)
    return by_dir


def existing_max_nn(directory: Path) -> dict[str, int]:
    """フォルダ内の整形済みファイルから、日付ごとの最大連番を求める"""
    out: dict[str, int] = {}
    for f in directory.iterdir():
        m = FORMATTED_RE.match(f.stem)
        if m:
            out[m.group(1)] = max(out.get(m.group(1), 0), int(m.group(2)))
    return out


def find_vault(directory: Path) -> Path | None:
    for d in [directory, *directory.parents]:
        if (d / ".obsidian").is_dir():
            return d
    return None


# ---------------------------------------------------------------- モデルへの接続


def http_json(url: str, payload: dict | None = None, timeout: float = 5) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


class BackendError(Exception):
    """接続できない・5xx。別のホストに切り替えれば直る可能性がある"""


class StructuredOutputError(BackendError):
    """response_format を要求したのにモデルが JSON を返せなかった。ホストの問題ではない"""


class Backend:
    """モデルへの接続。トンネルを自分で張った場合だけ、失敗時に別ホストへ切り替えて閉じる"""

    def __init__(self, hosts: list[str], port: int, model: str):
        self.hosts = hosts
        self.port = port
        self.model = model
        self.base = f"http://localhost:{port}/v1"
        self.proc: subprocess.Popen | None = None
        self.host: str | None = None
        self.candidates: list[str] = []
        self.generation = 0
        self.lock = threading.Lock()

    # ---- 状態の確認

    def local_state(self) -> str:
        """'free' | 'ready'（モデルあり） | 'other'（別の用途で使われている）"""
        try:
            models = http_json(f"{self.base}/models")
        except urllib.error.URLError as e:
            if isinstance(e.reason, ConnectionRefusedError):
                return "free"
            return "other"
        except (ConnectionRefusedError, ConnectionResetError):
            return "free"
        except Exception:
            return "other"
        ids = [m.get("id") for m in models.get("data", [])]
        return "ready" if self.model in ids else "other"

    def remote_status(self, host: str) -> str | None:
        """ホスト上のモデルの状態（loaded / unloaded ...）。つながらなければ None"""
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
                 f"curl -s -m 5 http://localhost:{REMOTE_PORT}/v1/models"],
                capture_output=True, text=True, timeout=20,
            ).stdout
            for m in json.loads(out).get("data", []):
                if m.get("id") == self.model:
                    return (m.get("status") or {}).get("value", "unknown")
        except Exception:
            return None
        return None

    # ---- 開始・終了

    def start(self) -> None:
        state = self.local_state()
        if state == "ready":
            log(f"既存のトンネル localhost:{self.port} の {self.model} を使います（終了時も閉じません）")
            return
        if state == "other":
            raise SystemExit(
                f"localhost:{self.port} は {self.model} 以外の用途で使われています。"
                f"--port で別のポートを指定するか、使っているプロセスを止めてください"
            )
        ranked: list[tuple[int, int, str]] = []
        for i, host in enumerate(self.hosts):
            status = self.remote_status(host)
            log(f"{host}: {self.model} = {status or 'つながらない / 見つからない'}")
            if status is not None:
                ranked.append((0 if status == "loaded" else 1, i, host))
        self.candidates = [h for _, _, h in sorted(ranked)]
        if not self.hosts:
            raise SystemExit(
                f"localhost:{self.port} で {self.model} が見つかりません。"
                f"--host か環境変数 FIGURE_RENAME_HOSTS でトンネル先のホストを指定してください"
            )
        if not self.candidates:
            raise SystemExit(f"{self.model} を提供しているホストにつながりません: {', '.join(self.hosts)}")
        self._open_next()

    def _open_next(self) -> None:
        self._close_tunnel()
        while self.candidates:
            host = self.candidates.pop(0)
            log(f"トンネルを張ります: ssh -L {self.port}:localhost:{REMOTE_PORT} {host}")
            proc = subprocess.Popen(
                ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
                 "-L", f"{self.port}:localhost:{REMOTE_PORT}", host],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            for _ in range(60):
                if proc.poll() is not None:
                    break
                if self.local_state() == "ready":
                    self.proc, self.host = proc, host
                    self.generation += 1
                    return
                time.sleep(0.5)
            err = proc.stderr.read().decode(errors="replace").strip() if proc.poll() is not None else ""
            proc.terminate()
            log(f"{host} へのトンネルを張れませんでした {err}")
        raise BackendError("使えるホストが残っていません")

    def _close_tunnel(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def close(self) -> None:
        if self.proc:
            log(f"トンネル（{self.host}）を閉じます")
        self._close_tunnel()

    # ---- 呼び出し

    def chat(self, payload: dict) -> dict:
        while True:
            gen = self.generation
            try:
                return http_json(f"{self.base}/chat/completions", payload, timeout=TIMEOUT)
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise
                try:
                    body = e.read().decode(errors="replace")
                except Exception:
                    body = ""
                if "structured_output" in body or "valid JSON" in body:
                    raise StructuredOutputError(f"{self.model} が JSON を返しませんでした") from e
                err: Exception = e
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                err = e
            if self.proc is None:  # 他人のトンネルは切り替えられない
                raise BackendError(f"{self.model} の呼び出しに失敗しました: {err}")
            with self.lock:
                if gen == self.generation:  # 他のスレッドがまだ切り替えていなければ切り替える
                    log(f"{self.host} で失敗しました（{err}）。別のホストに切り替えます")
                    self._open_next()


# ---------------------------------------------------------------- 画像解析


def encode_image(path: Path) -> str:
    with Image.open(path) as im:
        im.seek(0)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((MAX_EDGE, MAX_EDGE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def parse_reply(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("title"):
                return {
                    "title": str(obj["title"]),
                    "category": normalize_category(obj.get("category")),
                    "reason": str(obj.get("reason", "")),
                }
        except json.JSONDecodeError:
            pass
    first = next((line for line in text.splitlines() if line.strip()), "")
    return {
        "title": first.strip(' "「」'),
        "category": "その他",
        "reason": "(JSON で返らなかったため本文の 1 行目を使用)",
    }


def structured_failed(e: urllib.error.HTTPError) -> bool:
    """5xx の応答本文が structured output の失敗かどうか"""
    try:
        body = e.read().decode(errors="replace")
    except Exception:
        return False
    return "structured_output" in body or "valid JSON" in body


def analyze(backend: Backend, path: Path) -> dict:
    image = encode_image(path)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "この画像のファイル名用タイトルを JSON で返してください。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
        ]},
    ]
    base = {
        "model": backend.model, "temperature": 0, "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def call(msgs: list[dict]) -> str:
        def send(extra: dict) -> str:
            res = backend.chat({**base, "messages": msgs, **extra})
            return res["choices"][0]["message"].get("content") or ""

        try:
            return send({"response_format": {"type": "json_object"}})
        except (urllib.error.HTTPError, StructuredOutputError) as e:
            # response_format 非対応、または JSON を返せなかったときは無指定でやり直す
            if isinstance(e, urllib.error.HTTPError) and e.code >= 500 and not structured_failed(e):
                raise
            return send({})

    reply = call(messages)
    result = parse_reply(reply)
    title = sanitize_title(result["title"])
    if len(title) > MAX_TITLE:
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"title が {len(title)} 文字です。意味を保ったまま {MAX_TITLE} 文字以内に短くし、同じ JSON 形式で返してください。"},
        ]
        again = parse_reply(call(messages))
        result = {
            "title": again["title"],
            "category": again["category"] if again.get("category") != "その他" else result["category"],
            "reason": again.get("reason") or result.get("reason", ""),
        }
        title = sanitize_title(result["title"])
    result["title"] = truncate(title)
    result["category"] = normalize_category(result.get("category"))
    return result


# ---------------------------------------------------------------- HEIC → JPEG


def convert_to_jpeg(src: Path, dst: Path) -> None:
    """HEIC/HEIF を dst へ JPEG で書き出し、元ファイルは .heic-original/ に退避する"""
    with Image.open(src) as im:
        im.seek(0)
        im = ImageOps.exif_transpose(im)  # 回転をピクセルに反映し、orientation を 1 にする
        exif = im.getexif()
        im.convert("RGB").save(dst, "JPEG", quality=JPEG_QUALITY, exif=exif)
    keep = src.parent / ORIGINAL_DIR
    keep.mkdir(exist_ok=True)
    os.rename(src, keep / src.name)


# ---------------------------------------------------------------- plan


def load_plan(directory: Path) -> dict:
    try:
        return json.loads((directory / PLAN_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}


def save_plan(directory: Path, plan: dict) -> None:
    (directory / PLAN_FILE).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


def cache_hit(entry: dict | None, t: Target) -> bool:
    return bool(
        entry
        and entry.get("title")
        and entry.get("category") in CATEGORIES
        and entry.get("size") == t.size
        and abs(entry.get("mtime", 0) - t.mtime) < 1e-3
    )


def assign(directory: Path, targets: list[Target], entries: dict) -> list[dict]:
    """日付ごとに撮影時刻順で連番を振る"""
    nn = existing_max_nn(directory)
    taken_names = {f.name for f in directory.iterdir()}
    rows = []
    for t in sorted(targets, key=lambda t: (t.taken, t.path.name)):
        n = nn.get(t.date, 0)
        while True:
            n += 1
            width = 2 if n <= 99 else len(str(n))
            entry = entries[t.path.name]  # plan を手で編集した場合も整える
            title = truncate(sanitize_title(entry["title"]))
            cat = normalize_category(entry.get("category"))
            dst = f"{t.date}_{n:0{width}d}_{cat}_{title}{t.ext}"
            if dst not in taken_names:
                break
        nn[t.date] = n
        taken_names.add(dst)
        rows.append({"src": t.path.name, "dst": dst, "date_source": t.date_source,
                     "category": cat, "convert": bool(t.out_ext),
                     "reason": entries[t.path.name].get("reason", "")})
    return rows


# ---------------------------------------------------------------- 実行


def run_plan(args: argparse.Namespace) -> int:
    by_dir = collect(args.paths, args.recursive)
    if not by_dir:
        log("対象の画像がありません")
        return 0

    overrides: dict[str, tuple[str, str | None]] = {}
    for s in args.set or []:
        if "=" not in s:
            raise SystemExit(f"--set は '元ファイル名=新しい内容' の形で指定してください: {s}")
        k, v = s.split("=", 1)
        cat: str | None = None
        for c in CATEGORIES:  # "プレゼン:タイトル" の形で分類も上書きできる
            if v.startswith(c) and len(v) > len(c) and v[len(c)] in ":：":
                cat, v = c, v[len(c) + 1 :]
                break
        overrides[Path(k).name] = (truncate(sanitize_title(v)), cat)

    plans: dict[Path, dict] = {}
    targets: dict[Path, list[Target]] = {}
    todo: list[tuple[Path, Target]] = []
    for d, files in sorted(by_dir.items()):
        plan = load_plan(d)
        plans[d] = plan
        ts = []
        for f in files:
            st = f.stat()
            taken, src = read_taken(f)
            out_ext = ".jpg" if args.convert and f.suffix.lower() in HEIF_EXTS else ""
            t = Target(f, st.st_size, st.st_mtime, taken, src, out_ext)
            ts.append(t)
            entry = plan["entries"].get(f.name)
            if f.name in overrides:
                title, cat = overrides[f.name]
                prev = plan["entries"].get(f.name) or {}
                plan["entries"][f.name] = {"size": t.size, "mtime": t.mtime, "title": title,
                                           "category": cat or prev.get("category", "その他"),
                                           "reason": "(--set で指定)"}
            elif args.refresh or not cache_hit(entry, t):
                todo.append((d, t))
        targets[d] = ts

    backend = None
    if todo:
        backend = Backend(args.host or DEFAULT_HOSTS, args.port, args.model)
        try:
            backend.start()
            log(f"{len(todo)} 枚を解析します（同時 {args.jobs} 件）")
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(analyze, backend, t.path): (d, t) for d, t in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    d, t = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        log(f"[{i}/{len(todo)}] 失敗 {t.path.name}: {e}")
                        continue
                    plans[d]["entries"][t.path.name] = {"size": t.size, "mtime": t.mtime, **r}
                    log(f"[{i}/{len(todo)}] {t.path.name} → {r['title']}")
                    save_plan(d, plans[d])  # 途中で止まっても解析済みの分は残す
        finally:
            backend.close()

    failed = 0
    for d in sorted(targets):
        plan = plans[d]
        ok = [t for t in targets[d] if t.path.name in plan["entries"] and cache_hit(plan["entries"][t.path.name], t)]
        failed += len(targets[d]) - len(ok)
        rows = assign(d, ok, plan["entries"])
        plan["plan"] = rows
        plan["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        save_plan(d, plan)

        print(f"\n## {d}")
        vault = find_vault(d)
        if vault:
            print(f"⚠ Obsidian vault（{vault}）の中です。ノート内の ![[…]] リンクは書き換えません")
        for r in rows:
            mark = "" if r["date_source"] == "exif" else "  (日付は mtime)"
            conv = "  (HEIC→JPEG)" if r.get("convert") else ""
            print(f"  {r['src']}  →  {r['dst']}{mark}{conv}")
        if not rows:
            print("  （改名するファイルはありません）")
        elif len(rows) > 1:
            tally: dict[str, int] = {}
            for r in rows:
                tally[r["category"]] = tally.get(r["category"], 0) + 1
            print("  内訳: " + " / ".join(f"{c} {tally[c]}" for c in CATEGORIES if tally.get(c)))

        if args.apply and rows:
            run_id = uuid.uuid4().hex[:8]
            now = dt.datetime.now().isoformat(timespec="seconds")
            done = 0
            with (d / LOG_FILE).open("a") as lf:
                for r in rows:
                    src, dst = d / r["src"], d / r["dst"]
                    if dst.exists():
                        print(f"  ✗ {r['dst']} が既にあるため飛ばしました")
                        continue
                    if r.get("convert"):
                        try:
                            convert_to_jpeg(src, dst)
                        except Exception as e:
                            print(f"  ✗ {r['src']} の変換に失敗しました: {e}")
                            continue
                        lf.write(json.dumps({"run": run_id, "time": now, "src": r["src"], "dst": r["dst"],
                                             "convert": True, "original": f"{ORIGINAL_DIR}/{src.name}"},
                                            ensure_ascii=False) + "\n")
                    else:
                        os.rename(src, dst)
                        lf.write(json.dumps({"run": run_id, "time": now, "src": r["src"], "dst": r["dst"]},
                                            ensure_ascii=False) + "\n")
                    done += 1
            print(f"  ✓ {done} 件を改名しました（run {run_id}、--undo で戻せます）")

    if failed:
        print(f"\n解析に失敗した {failed} 件は改名案に含めていません。再実行すると失敗分だけ解析します")
    if not args.apply:
        print("\n(dry-run) 改名するには --apply を付けて再実行してください。"
              "個別に直すには --set \"元ファイル名=新しい内容\" を使ってください")
    return 1 if failed else 0


def run_undo(args: argparse.Namespace) -> int:
    dirs = {Path(p).expanduser().resolve() if Path(p).is_dir() else Path(p).expanduser().resolve().parent for p in args.paths}
    for d in sorted(dirs):
        path = d / LOG_FILE
        if not path.exists():
            print(f"{d}: 記録がありません")
            continue
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        undone = {r["undo"] for r in recs if "undo" in r}
        runs = [r["run"] for r in recs if "run" in r and r["run"] not in undone]
        if not runs:
            print(f"{d}: 戻せる実行がありません")
            continue
        last = runs[-1]
        print(f"\n## {d}（run {last} を戻します）")
        for r in reversed([r for r in recs if r.get("run") == last]):
            src, dst = d / r["src"], d / r["dst"]
            if r.get("convert"):
                orig = d / ORIGINAL_DIR / src.name
                if not orig.exists():
                    print(f"  ✗ {ORIGINAL_DIR}/{src.name} が見つからず戻せません")
                    continue
                if dst.exists():
                    os.remove(dst)
                os.rename(orig, src)
                print(f"  {r['dst']} を削除し、{ORIGINAL_DIR}/{src.name} → {r['src']}")
                continue
            if dst.exists() and not src.exists():
                os.rename(dst, src)
                print(f"  {r['dst']}  →  {r['src']}")
            else:
                print(f"  ✗ {r['dst']} を戻せません（無いか、{r['src']} が既にあります）")
        with path.open("a") as lf:
            lf.write(json.dumps({"undo": last, "time": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="画像ファイルを YYYYMMDD_NN_プレゼン|展示|食事|その他_画像の内容.ext に改名する（既定は dry-run）")
    ap.add_argument("paths", nargs="+", help="対象のフォルダまたはファイル")
    ap.add_argument("--apply", action="store_true", help="実際に改名する")
    ap.add_argument("--undo", action="store_true", help="直前の --apply を元に戻す")
    ap.add_argument("-r", "--recursive", action="store_true", help="サブフォルダも対象にする")
    ap.add_argument("--set", action="append", metavar="元ファイル名=新しい内容",
                    help="改名案を手で指定する（複数可）。「展示:タイトル」で分類も変えられる")
    ap.add_argument("--convert", action="store_true",
                    help="HEIC/HEIF を JPEG に変換してから改名する（元ファイルは .heic-original/ に残す）")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを無視して解析し直す")
    ap.add_argument("--host", action="append",
                    help="トンネルを張る SSH ホスト（複数可。既定: 環境変数 FIGURE_RENAME_HOSTS。"
                         "localhost:--port でモデルが動いていれば不要）")
    ap.add_argument("--port", type=int, default=8000, help="ローカル側のポート（既定: 8000）")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="モデル名（/v1/models の id。既定: 環境変数 FIGURE_RENAME_MODEL）")
    ap.add_argument("-j", "--jobs", type=int, default=2, help="同時に送るリクエスト数（既定: 2）")
    args = ap.parse_args()
    if not args.model and not args.undo:
        ap.error("--model か環境変数 FIGURE_RENAME_MODEL でモデル名を指定してください")
    if args.undo:
        return run_undo(args)
    try:
        return run_plan(args)
    except BackendError as e:
        log(str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
