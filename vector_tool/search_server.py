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
TOP_K_DEFAULT = 8
LOCK = threading.Lock()
STATE = {"mtime": None, "packs": [], "meta": None}  # packs: [{domain, n, dim, vecs, paths, heads, texts}]

_PATH_HEAD_BOOST = 0.12    # 查询 token 命中 doc_path/heading
_TEXT_BOOST = 0.03         # 命中正文
_BOOST_CAP = 0.6
_MAX_BODY = 4096           # POST body 上限（防恶意大包）
_CACHE = {}                # (q, k) -> results（热点查询缓存）
_CACHE_ORDER = []          # LRU 顺序
_CACHE_MAX = 128
_STOP = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}


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
    files = sorted(pack_dir.glob("*.pack.bin"))
    if not files:
        return False
    sig = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    with LOCK:
        if not force and sig == STATE["mtime"]:
            return False
        STATE["packs"] = [load_pack(p) for p in files]
        STATE["mtime"] = sig
    return True


def q_tokens(q: str):
    """查询词切分：tokenizer tokens，合并 ## 续词。"""
    _, tok = kb_vec.get_model()
    toks = []
    for t in tok.encode(q).tokens:
        if t in _STOP:
            continue
        if t.startswith("##"):
            if toks:
                toks[-1] += t[2:]
            continue
        toks.append(t)
    return toks


def search(q: str, k: int):
    import numpy as np
    qv = kb_vec.embed([q], is_query=True)[0]
    q8 = np.round(qv * 127).astype(np.int32)
    toks = q_tokens(q)
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
                if t in doc or (head and t in head):
                    boost += _PATH_HEAD_BOOST
                elif t in text:
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


def cache_get(key):
    if key in _CACHE:
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
    key = (q, k)
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
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
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
            changed = reload_if_changed(pack_dir)
            if changed:
                cache_clear()
            t0 = time.time()
            hits = search_cached(q, k)
            resp = {
                "results": [
                    {"score": round(h[0], 4), "vec": round(h[5], 4),
                     "domain": h[1], "doc_path": h[2], "heading": h[3],
                     "text": h[4][:4000]}
                    for h in hits
                ],
                "ms": int((time.time() - t0) * 1000),
            }
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
    if not ok or not STATE["packs"]:
        print(f"FATAL: 无 pack 可加载: {pack_dir}", file=sys.stderr)
        sys.exit(1)
    n = sum(p["n"] for p in STATE["packs"])
    print(f"[search_server] 加载 {len(STATE['packs'])} 个 pack, {n} chunks, 端口 {args.port}")
    # 预热模型/分词器：首查不必背 onnxruntime 初始化（实测首查 259ms → 预热后稳定 ~25ms）
    t0 = time.time()
    kb_vec.embed(["预热"], is_query=True)
    print(f"[search_server] 模型预热完成 {time.time()-t0:.1f}s，就绪")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
