#!/usr/bin/env python3
"""vector-web 语义搜索服务（服务器端检索，客户端零模型下载）。

设计（2026-09-22，用户两项要求）：
1. 客户端不再下载 ~70MB 模型/pack，改为 POST /api/search，服务端返回 top-k
2. 短查询（如「教你」）纯语义向量区分度差（bge-small-zh 2 字查询余弦普遍
   ~0.42 且彼此接近），在向量分基础上加**词汇命中 boost**：查询 token 命中
   doc_path/heading 各 +0.12、命中正文 +0.03（上限 +0.6），使目录/标题词
   查询（如「教你炒股票」目录名）能排到前面，同时长语义查询不受影响。

依赖：stdlib + numpy + onnxruntime + tokenizers（vector_tool 已有）。
用法：python search_server.py [--port 8710] [--pack-dir DIR]
"""
import argparse
import json
import os
import re
import struct
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb_vec

PACK_DIR = Path(kb_vec.MODEL_DIR.parent)  # vector_tool/models -> vector_tool；下方被 args 覆盖
ALIAS_FILE = Path(__file__).resolve().parent / "aliases.json"
TOP_K_DEFAULT = 8
LOCK = threading.Lock()
STATE = {"mtime": None, "packs": [], "meta": None, "loading": False,
         "alias_mtime": None, "aliases": {}}  # packs: [{domain, n, dim, vecs, paths, heads, texts}]

_PATH_BOOST = 0.12         # 查询 token 命中 doc_path（多字，目录名强信号）
_PATH_SINGLE = 0.08        # 命中 doc_path（单字，弱信号——防"着至教论"式误报）
_HEAD_BOOST = 0.12         # 命中 heading（多字 token）
_HEAD_SINGLE = 0.06        # 命中 heading（单字 token，弱信号，防"着至教论"式误报）
_TEXT_BOOST = 0.03         # 命中正文（仅多字 token）
_BOOST_CAP = 0.6
_MAX_BODY = 4096           # POST body 上限（防恶意大包）
_MAX_CONCURRENT = 16       # 并发上限（超出 429，防打爆）
_SEM = threading.BoundedSemaphore(_MAX_CONCURRENT)
_CACHE = {}                # (q, k) -> results（热点查询缓存）
_CACHE_ORDER = []          # LRU 顺序
_CACHE_MAX = 128
_STOP = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}
_QUERIES = 0               # 累计查询数
_CACHE_HITS = 0            # 缓存命中数
_QUERY_LOCK = threading.Lock()


def load_pack(path: Path):
    data = path.read_bytes()
    assert data[:4] == b"QPK1", path
    dim, n = struct.unpack("<HI", data[4:10])
    off, segs = 10, []
    for _ in range(5):
        (ln,) = struct.unpack("<I", data[off:off + 4]); off += 4
        segs.append((off, ln)); off += ln
    vo, vl = segs[0]; io, il = segs[1]; po, pl = segs[2]; ho, hl = segs[3]; to, tl = segs[4]
    import numpy as np
    vecs = np.frombuffer(data[vo:vo + vl], dtype=np.int8).reshape(n, dim)
    idx = np.frombuffer(data[io:io + il], dtype="<u4")
    paths = data[po:po + pl]
    heads = data[ho:ho + hl]
    texts = []
    o = to
    while o < to + tl:
        (ln,) = struct.unpack("<I", data[o:o + 4]); o += 4
        texts.append(zlib.decompress(data[o:o + ln]).decode("utf-8", "ignore"))
        o += ln
    assert len(texts) == n, (path, len(texts), n)
    return {"domain": path.stem, "n": n, "dim": dim, "vecs": vecs,
            "idx": idx, "paths": paths, "heads": heads, "texts": texts}


def reload_if_changed(pack_dir: Path, force=False):
    """检测 pack 变化 → 后台线程加载（原子替换），请求不卡 1s 重载。
    加载失败不抛：保留旧 packs 继续服务。"""
    files = sorted(pack_dir.glob("*.pack.bin"))
    if not files:
        return False
    sig = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    with LOCK:
        if not force and sig == STATE["mtime"]:
            return False
        if STATE.get("loading"):
            return True                      # 已在后台加载，避免重复线程
        STATE["loading"] = True

    def do_load():
        try:
            packs = [load_pack(p) for p in files]
            with LOCK:
                STATE["packs"] = packs
                STATE["mtime"] = sig
                cache_clear()
                STATE["loading"] = False
        except Exception as e:
            with LOCK:
                STATE["loading"] = False
            print(f"[search_server] 重载失败(保留旧数据): {e}", file=sys.stderr)

    threading.Thread(target=do_load, daemon=True).start()
    return True


def q_tokens(q: str):
    """查询词切分：按原始空格分段 → 段内 tokenizer 分词（合并 ## 续词）→
    段内连续中文单字合成词。保留词边界（\"量化 游资\"→[\"量化\",\"游资\"]，
    不是\"量化游资\"）；合成词让 boost/高亮按词命中（\"教你\"→一个词），
    避免单字误报（\"着至教论\"路径含\"教\"字不再命中\"教你\"）。"""
    _, tok = kb_vec.get_model()
    out = []
    for seg in re.split(r"\s+", q.strip()):
        if not seg:
            continue
        toks = []
        for t in tok.encode(seg).tokens:
            if t in _STOP:
                continue
            if t.startswith("##"):
                if toks:
                    toks[-1] += t[2:]
                continue
            toks.append(t)
        merged, buf = [], ""
        for t in toks:
            if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
                buf += t
            else:
                if buf:
                    merged.append(buf)
                    buf = ""
                merged.append(t)
        if buf:
            merged.append(buf)
        out.extend(m for m in merged if m)
    return out


def load_aliases_if_changed(force=False):
    """加载/重载 aliases.json（mtime 检测，改别名无需重启服务）。"""
    try:
        sig = ALIAS_FILE.stat().st_mtime_ns
    except OSError:
        return False
    with LOCK:
        if not force and sig == STATE["alias_mtime"]:
            return False
        try:
            data = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
            data.pop("_comment", None)
            STATE["aliases"] = data
            STATE["alias_mtime"] = sig
            return True
        except Exception as e:
            print(f"[search_server] 别名表加载失败(保留旧表): {e}", file=sys.stderr)
            return False


def expand_terms(q: str, terms):
    """别名扩展：原始整串精确匹配 + 词级匹配，扩展词追加参与 boost/高亮。"""
    extra = []
    aliases = STATE["aliases"]
    if not aliases:
        return terms
    qs = q.strip().lower()
    aliases_l = {k.lower(): v for k, v in aliases.items()}
    # 整串精确匹配（"教你炒股"→"教你炒股票"）
    if qs in aliases_l:
        extra.extend(aliases_l[qs])
    # 词级匹配
    for t in terms:
        tl = t.lower()
        if tl in aliases_l:
            extra.extend(aliases_l[tl])
    if not extra:
        return terms
    merged = list(terms)
    for x in extra:
        if x not in merged:
            merged.append(x)
    return merged


def search(q: str, k: int):
    import numpy as np
    qv = kb_vec.embed([q], is_query=True)[0]
    q8 = np.round(qv * 127).astype(np.int32)
    toks = expand_terms(q, q_tokens(q))
    hits = []
    for pk in STATE["packs"]:
        sims = pk["vecs"] @ q8
        kk = min(k * 4, pk["n"])          # 先取每库 4k，避免 boost 后漏掉
        if kk <= 0:
            continue
        top = np.argpartition(-sims, kk - 1)[:kk]
        order = np.argsort(-sims[top])
        for i in top[order]:
            b = int(i) * 4
            ps, plen = int(pk["idx"][b]), int(pk["idx"][b + 1])
            hs, hlen = int(pk["idx"][b + 2]), int(pk["idx"][b + 3])
            doc = pk["paths"][ps:ps + plen].decode("utf-8", "ignore")
            head = pk["heads"][hs:hs + hlen].decode("utf-8", "ignore")
            text = pk["texts"][int(i)]
            vec_score = float(sims[int(i)]) / (127 * 127)
            boost = 0.0
            for t in toks:
                if not t:
                    continue
                if t in doc:
                    boost += _PATH_BOOST if len(t) > 1 else _PATH_SINGLE
                elif head and t in head:
                    boost += _HEAD_BOOST if len(t) > 1 else _HEAD_SINGLE
                # 单字（如"教""你"）命中正文不再加分——避免"着至教论"等
                # 仅含单字的无关文档被 text boost 拉高（2026-09-24 实测
                # 「教你」top3 混入中医「着至教论」）
                elif len(t) > 1 and t in text:
                    boost += _TEXT_BOOST
            score = vec_score + min(boost, _BOOST_CAP)
            hits.append((score, pk["domain"], doc, head, text, vec_score))
    hits.sort(key=lambda r: -r[0])
    # 同文档去重：同一 doc_path 只保留最高分条目（避免一篇占多卡）。
    seen, out = set(), []
    for h in hits:
        if h[2] in seen:
            continue
        seen.add(h[2])
        out.append(h)
        if len(out) >= k:
            break
    return out


def suggest(q: str, limit: int = 8):
    """搜索建议：别名表 key 匹配 + 库内文件名包含匹配。"""
    if not q or not STATE["packs"]:
        return []
    qs = q.strip().lower()
    out, seen = [], set()
    # 别名表 key
    for k in STATE["aliases"]:
        if qs in k.lower() or k.lower() in qs:
            if k not in seen:
                seen.add(k)
                out.append(k)
        if len(out) >= limit:
            return out
    # 库内文件名（通过 idx 段提取 doc_path；paths blob 无分隔符，不能 split）
    for pk in STATE["packs"]:
        idx = pk["idx"]
        for b in range(0, pk["n"] * 4, 4):
            ps, plen = int(idx[b]), int(idx[b + 1])
            doc = pk["paths"][ps:ps + plen].decode("utf-8", "ignore")
            base = doc.rsplit("/", 1)[-1].replace(".md", "")
            base = re.sub(r"^\d{3}-", "", base)
            if base and qs in base.lower() and base not in seen:
                seen.add(base)
                out.append(base)
                if len(out) >= limit:
                    return out
    return out


def cache_get(key):
    global _CACHE_HITS
    if key in _CACHE:
        _CACHE_HITS += 1
        _CACHE_ORDER.remove(key)
        _CACHE_ORDER.append(key)
        return _CACHE[key]
    return None


def cache_put(key, val):
    if key in _CACHE:
        _CACHE_ORDER.remove(key)
    _CACHE[key] = val
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _CACHE.pop(old, None)


def cache_clear():
    _CACHE.clear()
    _CACHE_ORDER.clear()


def search_cached(q: str, k: int):
    global _QUERIES
    key = (q, k)
    with _QUERY_LOCK:
        _QUERIES += 1
    got = cache_get(key)
    if got is not None:
        return got
    out = search(q, k)
    cache_put(key, out)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/healthz":
            with LOCK:
                n = sum(p["n"] for p in STATE["packs"])
                body = json.dumps({"ok": True, "packs": len(STATE["packs"]),
                                   "chunks": n, "cache": len(_CACHE),
                                   "loading": STATE.get("loading", False),
                                   "queries": _QUERIES, "cache_hits": _CACHE_HITS}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/suggest"):
            try:
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
                if not qs:
                    body = b'{"suggestions":[]}'
                else:
                    body = json.dumps({"suggestions": suggest(qs)},
                                      ensure_ascii=False).encode()
            except Exception:
                body = b'{"suggestions":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/search":
            self.send_response(404); self.end_headers(); return
        ln = int(self.headers.get("Content-Length") or 0)
        if ln > _MAX_BODY:
            self.send_response(413); self.end_headers(); return
        try:
            body = json.loads(self.rfile.read(ln) or b"{}")
            q = str(body.get("q") or "").strip()
            k = min(int(body.get("k") or TOP_K_DEFAULT), 30)
        except Exception:
            self.send_response(400); self.end_headers(); return
        if not q:
            resp = {"results": [], "error": "empty query"}
        else:
            if not _SEM.acquire(blocking=False):
                out = b'{"error":"busy","results":[]}'
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            try:
                changed = reload_if_changed(pack_dir)
                if changed:
                    cache_clear()
                load_aliases_if_changed()
                t0 = time.time()
                hits = search_cached(q, k)
                dt_ms = int((time.time() - t0) * 1000)
                if dt_ms > 200:                     # 慢查询日志（systemd journal）
                    print(f"[search_server] slow q={q!r} k={k} {dt_ms}ms", file=sys.stderr)
                resp = {
                    "terms": expand_terms(q, q_tokens(q)),
                    "results": [
                        {"score": round(h[0], 4), "vec": round(h[5], 4),
                         "domain": h[1], "doc_path": h[2], "heading": h[3],
                         "text": h[4][:800]}
                        for h in hits
                    ],
                    "ms": dt_ms,
                }
            finally:
                _SEM.release()
        out = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


pack_dir = None


def main():
    global pack_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--pack-dir", default=str(PACK_DIR))
    args = ap.parse_args()
    pack_dir = Path(args.pack_dir)
    ok = reload_if_changed(pack_dir, force=True)
    load_aliases_if_changed(force=True)
    print(f"[search_server] 别名表 {len(STATE['aliases'])} 条")
    if not ok or not STATE["packs"]:
        # 启动容错：sync_public.sh 用 git reset --hard，有窗口期 pack 不存在；
        # 保持运行，首个请求时再尝试加载（reload_if_changed 会重试）。
        print(f"[search_server] 警告: 暂无 pack 可加载: {pack_dir}（请求时将重试）",
              file=sys.stderr)
    else:
        n = sum(p["n"] for p in STATE["packs"])
        print(f"[search_server] 加载 {len(STATE['packs'])} 个 pack, {n} chunks, 端口 {args.port}")
    # 预热模型/分词器：首查不必背 onnxruntime 初始化（实测首查 259ms → 预热后稳定 ~25ms）
    t0 = time.time()
    kb_vec.embed(["预热"], is_query=True)
    print(f"[search_server] 模型预热完成 {time.time()-t0:.1f}s，就绪")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
