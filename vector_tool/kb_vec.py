#!/usr/bin/env python3
"""obsdata 知识库向量索引：Markdown 分块 + bge-small-zh-v1.5(ONNX) + sqlite-vec。

用法:
  build   --vault PATH --db PATH [--limit N]   增量构建（按内容 hash 跳过未变文件）
  query   "检索文本" [--db PATH] [-k 8] [--domain 理财]   语义检索
  status  [--db PATH]                           统计

设计:
  - 按标题分块（保留 heading 路径做上下文），小块合并、大块按段落再切
  - CLS pooling + L2 归一（bge 语义），查询侧加 bge 中文指令前缀
  - chunks 文本与元数据存普通表，向量存 vec0 虚表，rowid 对齐
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

VAULT_DEFAULT = os.environ.get("KB_VAULT", "/home/ai/ai_runner/obsdata")
VAULT = VAULT_DEFAULT  # 各命令入口会按 --vault 重设；rel_domain 依赖它
DB_DEFAULT = os.environ.get("KB_DB", "/home/ai/ai_runner/kb_vec/kb.db")
# 模型目录：默认取脚本同级的 models/（可整体搬到 CI 仓库），KB_MODEL_DIR 可覆盖
MODEL_DIR = Path(os.environ.get("KB_MODEL_DIR") or (Path(__file__).resolve().parent / "models"))
EMBED_DIM = 512
BATCH = 24
MAX_TOKENS = 480
CHUNK_MIN, CHUNK_SOFT, CHUNK_HARD = 200, 900, 1400
QUERY_INSTRUCT = "为这个句子生成表示以用于检索相关文章："
SKIP_DIRS = {".git", ".obsidian", ".claude", ".cache", "assets", "Templates",
             "node_modules", "__pycache__"}
SKIP_MD_WORDS = ("health_report", "emergency_log")

# ---------------- markdown 解析 ----------------

FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def parse_frontmatter(text):
    m = FM_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and v:
            meta[k] = v
    return meta, text[m.end():]


def split_sections(body):
    """按 ATX 标题切段，返回 [(heading_path, text)]；开头无标题段落 heading=''。"""
    lines = body.splitlines()
    sections, cur_head, cur = [], [], []
    stack = []  # [(level, title)]
    fence = False
    for ln in lines:
        if ln.lstrip().startswith(("```", "~~~")):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,4})\s+(.+?)\s*$", ln)
        if m:
            if "".join(cur).strip():
                sections.append((" > ".join(stack and [t for _, t in stack] or []),
                                 "\n".join(cur).strip()))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [(l, t) for l, t in stack if l < level] + [(level, title)]
            cur_head, cur = title, [ln]
        else:
            cur.append(ln)
    if "".join(cur).strip():
        sections.append((" > ".join([t for _, t in stack] or []), "\n".join(cur).strip()))
    return sections


def long_parts(text, limit=CHUNK_HARD):
    """大块按空行段落聚合成 <=limit 的片。超长的单段硬切。"""
    paras = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for p in paras:
        if len(p) > limit:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(p), limit):
                out.append(p[i:i + limit])
            continue
        cand = (buf + "\n\n" + p).strip() if buf else p.strip()
        if len(cand) > limit:
            out.append(buf)
            buf = p.strip()
        else:
            buf = cand
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_file(path, text):
    path = Path(path)
    meta, body = parse_frontmatter(text)
    raw_secs = split_sections(body)
    # 合并过小相邻块
    merged = []
    for head, sec in raw_secs:
        if merged and len(sec) < CHUNK_MIN and len(merged[-1][1]) < CHUNK_SOFT:
            h0, s0 = merged[-1]
            merged[-1] = (h0 or head, s0 + "\n\n" + sec)
        else:
            merged.append((head, sec))
    title = meta.get("title") or meta.get("h1") or path.stem
    domain = rel_domain(path)
    chunks = []
    for head, sec in merged:
        parts = [sec] if len(sec) <= CHUNK_HARD else long_parts(sec)
        for p in parts:
            if len(p.strip()) < 30:
                continue
            prefix = f"[{head}]\n" if head else ""
            chunks.append((title, domain, head, (title + "\n" + prefix + p)[:4000]))
    return meta, chunks


def rel_domain(path):
    parts = Path(path).relative_to(VAULT).parts
    if len(parts) < 2:  # 根级散落文件
        return "_root"
    return parts[0] if parts[0] not in SKIP_DIRS else "_root"


# ---------------- embedding ----------------

_model = _tok = None


def get_model():
    global _model, _tok
    if _model is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        _tok.enable_truncation(max_length=MAX_TOKENS)
        _tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        # 确定性说明：向量字节漂移的实测根因是 onnxruntime 版本变化（跨天
        # CI 19-23% 字节差异），已由 requirements.txt 锁版本解决；同版本下
        # 多线程/单线程结果一致（实测 0 差异），故保留多线程以维持 CI 速度
        # （单线程会把 shard 步骤从 ~5min 拖到 ~34min）。
        # 若未来再次观察到 pack 漂移，先查依赖版本，再考虑改 intra_op=1。
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        _model = ort.InferenceSession(
            str(MODEL_DIR / "model_quantized.onnx"), opts,
            providers=["CPUExecutionProvider"])
    return _model, _tok


def embed(texts, is_query=False):
    model, tok = get_model()
    if is_query:
        texts = [QUERY_INSTRUCT + t for t in texts]
    embs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok.encode_batch(batch)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        tids = np.array([e.type_ids for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        inp_names = {x.name for x in model.get_inputs()}
        if "token_type_ids" in inp_names:
            feeds["token_type_ids"] = tids
        out = model.run(None, feeds)[0]  # (B, S, H)
        cls = out[:, 0, :]
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        embs.append(cls.astype(np.float32))
    return np.vstack(embs)


# ---------------- 存储 ----------------

def open_db(db_path):
    import sqlite_vec
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(f"""
    CREATE TABLE IF NOT EXISTS docs(
      path TEXT PRIMARY KEY, domain TEXT, title TEXT,
      hash TEXT, mtime REAL, n_chunks INTEGER);
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY, doc_path TEXT, seq INTEGER,
      domain TEXT, heading TEXT, text TEXT, tags TEXT, summary TEXT);
    CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(
      id INTEGER PRIMARY KEY, embedding float[{EMBED_DIM}] distance_metric=cosine);
    """)
    return db


def iter_md(vault):
    root = Path(vault)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".md") and not any(w in fn for w in SKIP_MD_WORDS):
                yield os.path.join(dirpath, fn)


def build_docs(db, files):
    """增量嵌入 files（绝对路径列表），幂等：按内容 hash 跳过未变，
    内容重复仅挂链接。返回 (n_docs, n_chunks, linked)。

    注意：**只看 hash，不看 mtime**。CI 每次全新 checkout，文件 mtime 全是
    checkout 时刻，用 mtime 做快速路径会让增量永远失效（每轮全量重嵌）。
    """
    have = {p: (h, m, n) for p, h, m, n in
            db.execute("SELECT path, hash, mtime, n_chunks FROM docs")}
    # 清理已从 vault 删除的文档：复用缓存 db 时必须清，否则 pack 里残留
    # 死链（笔记已删、检索仍命中 404）
    fileset = set(files)
    stale = [p for p in have if p not in fileset]
    for p in stale:
        db.execute("DELETE FROM vec WHERE id IN (SELECT id FROM chunks WHERE doc_path=?)", (p,))
        db.execute("DELETE FROM chunks WHERE doc_path=?", (p,))
        db.execute("DELETE FROM docs WHERE path=?", (p,))
        have.pop(p, None)
    if stale:
        db.commit()
        print(f"[build] 清理已删除文档 {len(stale)} 个", flush=True)
    todo = []       # (path, text, hash, canonical_or_None)
    linked = 0      # 内容重复、仅挂链接不重嵌
    # canonical 按 files 顺序（iter_md 排序）确定：同 hash 的首个文件为
    # canonical 并嵌入，其余只挂链接。**同轮内也要去重**——否则"新增重复
    # 文件"在增量路径（旧 db 已有 canonical → 挂链接）与全量路径（同轮都
    # 嵌入）下产出不同 pack；顺序由 files 决定，两条路径结果一致。
    canon_of = {}   # hash -> 本轮首个 path
    for f in files:
        mt = os.path.getmtime(f)
        cur = have.get(f)
        try:
            text = Path(f).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        h = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        if h in canon_of:
            if cur is None or cur[2] != 0:
                # 非 canonical 却带 chunks（历史同轮双份）→ 收敛为挂链接
                todo.append((f, "", h, canon_of[h]))
                linked += 1
            continue
        canon_of[h] = f
        if cur and cur[0] == h and cur[2] > 0:
            # 已是 canonical 且内容未变 → 跳过（仅刷新 mtime）
            if cur[1] != mt:
                db.execute("UPDATE docs SET mtime=? WHERE path=?", (mt, f))
            continue
        # 新增 / 内容变化 / 原为挂链接（canonical 消失或顺序变了）→ 嵌入
        todo.append((f, text, h, None))
    # 幂等修正：早期版本把根级散落文件的 domain 记成了文件名
    db.execute("UPDATE chunks SET domain='_root' WHERE domain LIKE '%.md'")
    db.execute("UPDATE docs SET domain='_root' WHERE domain LIKE '%.md'")
    db.commit()
    print(f"[build] md={len(files)} 需处理={len(todo)} (其中内容重复仅链接={linked})", flush=True)
    t0 = time.time()
    n_docs = n_chunks = 0
    for fi, (f, text, h, canon) in enumerate(todo):
        if canon:  # 纯去重：docs 记 hash，不产 chunks
            db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,0)",
                       (f, rel_domain(f), Path(f).stem, h,
                        os.path.getmtime(f)))
            db.commit()
            continue
        meta, chunks = chunk_file(f, text)
        db.execute("DELETE FROM vec WHERE id IN (SELECT id FROM chunks WHERE doc_path=?)", (f,))
        db.execute("DELETE FROM chunks WHERE doc_path=?", (f,))
        if not chunks:
            db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,0)",
                       (f, rel_domain(f), meta.get("title") or Path(f).stem, h,
                        os.path.getmtime(f)))
            db.commit()
            continue
        vec = embed([c[3] for c in chunks])
        tags = meta.get("tags", "")
        for seq, (title, domain, head, chunk_text) in enumerate(chunks):
            cur = db.execute(
                "INSERT INTO chunks(doc_path,seq,domain,heading,text,tags,summary) VALUES(?,?,?,?,?,?,?)",
                (f, seq, domain, head, chunk_text, tags, meta.get("summary", "")))
            db.execute("INSERT INTO vec(id, embedding) VALUES(?,?)",
                       (cur.lastrowid, vec[seq].tobytes()))
        db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,?)",
                   (f, rel_domain(f), meta.get("title") or Path(f).stem, h,
                    os.path.getmtime(f), len(chunks)))
        db.commit()
        n_docs += 1
        n_chunks += len(chunks)
        if (fi + 1) % 100 == 0 or fi + 1 == len(todo):
            el = time.time() - t0
            rate = (n_docs) / el if el else 0
            eta = (len(todo) - linked - n_docs) / rate if rate else 0
            print(f"[build] {fi+1}/{len(todo)} docs={n_docs} chunks={n_chunks} "
                  f"{el:.0f}s rate={rate:.2f}doc/s eta={eta/60:.0f}min", flush=True)
    print(f"[done] 嵌入 docs={n_docs} chunks={n_chunks} 去重链接={linked} "
          f"用时 {time.time()-t0:.0f}s")
    return n_docs, n_chunks, linked


def cmd_build(args):
    global VAULT
    VAULT = args.vault
    db = open_db(args.db)
    files = sorted(iter_md(args.vault))
    if args.limit:
        files = files[: args.limit]
    build_docs(db, files)


def shard_open(db_path):
    """分库专用 schema：普通表，无 vec0 的 2MB chunk 预分配地板。
    emb.v = int8[512] blob（float32 归一向量 ×127）；chunks.text = gzip blob。"""
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS docs(
      path TEXT PRIMARY KEY, domain TEXT, title TEXT,
      hash TEXT, mtime REAL, n_chunks INTEGER);
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY, doc_path TEXT, seq INTEGER,
      domain TEXT, heading TEXT, text BLOB, tags TEXT, summary TEXT);
    CREATE TABLE IF NOT EXISTS emb(id INTEGER PRIMARY KEY, v BLOB NOT NULL);
    """)
    db.execute("INSERT OR IGNORE INTO meta VALUES('format','int8+gzip-v1')")
    db.execute("INSERT OR IGNORE INTO meta VALUES('dim',?)", (str(EMBED_DIM),))
    db.commit()
    return db


def shard_build_one(db, f, text, h):
    """单文件嵌入写入分库（int8 向量 + gzip 文本）。返回 chunk 数。"""
    import zlib
    meta, chunks = chunk_file(f, text)
    db.execute("DELETE FROM emb WHERE id IN (SELECT id FROM chunks WHERE doc_path=?)", (f,))
    db.execute("DELETE FROM chunks WHERE doc_path=?", (f,))
    if not chunks:
        db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,0)",
                   (f, rel_domain(f), meta.get("title") or Path(f).stem, h,
                    os.path.getmtime(f)))
        db.commit()
        return 0
    vec = embed([c[3] for c in chunks])
    q8 = np.clip(np.round(vec * 127), -127, 127).astype(np.int8)
    tags = meta.get("tags", "")
    for seq, (title, domain, head, chunk_text) in enumerate(chunks):
        cur = db.execute(
            "INSERT INTO chunks(doc_path,seq,domain,heading,text,tags,summary) VALUES(?,?,?,?,?,?,?)",
            (f, seq, domain, head, zlib.compress(chunk_text.encode("utf-8"), 6),
             tags, meta.get("summary", "")))
        db.execute("INSERT INTO emb VALUES(?,?)", (cur.lastrowid, q8[seq].tobytes()))
    db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,?)",
               (f, rel_domain(f), meta.get("title") or Path(f).stem, h,
                os.path.getmtime(f), len(chunks)))
    db.commit()
    return len(chunks)


def shard_build_files(db, files):
    """分库增量嵌入：与 build_docs 同语义（只看内容 hash、不看 mtime，
    重复内容仅挂链接、已删文档清理）。"""
    have = {p: (hh, m, n) for p, hh, m, n in
            db.execute("SELECT path, hash, mtime, n_chunks FROM docs")}
    fileset = set(files)
    stale = [p for p in have if p not in fileset]
    for p in stale:
        db.execute("DELETE FROM emb WHERE id IN (SELECT id FROM chunks WHERE doc_path=?)", (p,))
        db.execute("DELETE FROM chunks WHERE doc_path=?", (p,))
        db.execute("DELETE FROM docs WHERE path=?", (p,))
        have.pop(p, None)
    if stale:
        db.commit()
        print(f"[shard] 清理已删除文档 {len(stale)} 个", flush=True)
    todo, linked = [], 0
    canon_of = {}   # hash -> 本轮首个 path（canonical 按 files 顺序确定）
    for f in files:
        mt = os.path.getmtime(f)
        cur = have.get(f)
        try:
            text = Path(f).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        h = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        if h in canon_of:
            if cur is None or cur[2] != 0:
                db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,0)",
                           (f, rel_domain(f), Path(f).stem, h, mt))
                linked += 1
            continue
        canon_of[h] = f
        if cur and cur[0] == h and cur[2] > 0:
            if cur[1] != mt:
                db.execute("UPDATE docs SET mtime=? WHERE path=?", (mt, f))
            continue
        todo.append((f, text, h))
    db.commit()
    print(f"[shard] md={len(files)} 需嵌入={len(todo)} 去重链接={linked}", flush=True)
    t0 = time.time()
    n_docs = n_chunks = 0
    for fi, (f, text, h) in enumerate(todo):
        n = shard_build_one(db, f, text, h)
        n_docs += 1
        n_chunks += n
        if (fi + 1) % 100 == 0 or fi + 1 == len(todo):
            el = time.time() - t0
            rate = n_docs / el if el else 0
            eta = (len(todo) - n_docs) / rate if rate else 0
            print(f"[shard] {fi+1}/{len(todo)} chunks={n_chunks} "
                  f"eta={eta/60:.0f}min", flush=True)
    return n_docs, n_chunks, linked


def domain_groups(vault):
    """vault 内全部 md 按一级目录(领域)分组，返回 {domain: [abs paths]}"""
    groups = {}
    for f in iter_md(vault):
        groups.setdefault(rel_domain(f), []).append(f)
    return {k: sorted(v) for k, v in groups.items()}


def cmd_shard(args):
    """按领域拆分向量库：每域一个自包含小 .db + manifest.json（幂等增量）。"""
    global VAULT
    VAULT = args.vault
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = domain_groups(args.vault)
    manifest = {"vault": os.path.basename(os.path.abspath(args.vault)),
                "dim": EMBED_DIM, "model": "bge-small-zh-v1.5",
                "metric": "cosine-int8", "format": "int8+gzip-v1",
                "domains": []}
    for dom in sorted(groups):
        db_path = out / f"{dom}.db"
        db = shard_open(db_path)
        shard_build_files(db, groups[dom])
        n = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db.close()
        if not args.no_vacuum and n:
            vdb = sqlite3.connect(db_path)
            vdb.execute("VACUUM")
            vdb.close()
        manifest["domains"].append({
            "domain": dom, "file": db_path.name, "docs": len(groups[dom]),
            "chunks": n, "bytes": db_path.stat().st_size})
    tot = sum(d["bytes"] for d in manifest["domains"])
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[shard] domains={len(manifest['domains'])} "
          f"total={tot/1e6:.1f}MB -> {out}", flush=True)


def cmd_qshards(args):
    """跨分库统一检索：numpy int8 暴力点积（无 vec0 依赖）。"""
    global VAULT
    VAULT = args.vault
    import zlib
    out = Path(args.out)
    mf = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    qv = embed([args.text], is_query=True)[0]
    q8 = np.round(qv * 127).astype(np.int32)
    hits = []
    for d in mf["domains"]:
        if args.domain and d["domain"] != args.domain:
            continue
        db = shard_open(out / d["file"])
        ids = [r[0] for r in db.execute("SELECT id FROM emb ORDER BY id")]
        if not ids:
            db.close()
            continue
        M = np.frombuffer(b"".join(
            r[0] for r in db.execute("SELECT v FROM emb ORDER BY id")),
            dtype=np.int8).reshape(len(ids), EMBED_DIM)
        sims = M @ q8
        kk = min(args.k, len(ids))
        top = np.argpartition(-sims, kk - 1)[:kk] if kk < len(ids) else np.arange(len(ids))
        for i in top[np.argsort(-sims[top])]:
            row = db.execute(
                "SELECT doc_path, heading, text FROM chunks WHERE id=?",
                (ids[i],)).fetchone()
            if row:
                hits.append((float(sims[i]) / (127*127), row[0], row[1],
                             zlib.decompress(row[2]).decode("utf-8", "ignore")))
        db.close()
    hits.sort(key=lambda r: -r[0])
    for score, p, head, txt in hits[: args.k]:
        try:
            rp = Path(p).relative_to(VAULT)
        except ValueError:
            rp = Path(p).name
        head = head or Path(p).stem
        snip = re.sub(r"\s+", " ", txt)[:160]
        print(f"{score:.4f}  {rp}  «{head}»\n        {snip}")


def cmd_query(args):
    global VAULT
    VAULT = args.vault
    db = open_db(args.db)
    qv = embed([args.text], is_query=True)[0]
    sql = """
      SELECT c.doc_path, c.heading, substr(c.text,1,300), v.distance
      FROM vec AS v JOIN chunks AS c ON c.id = v.id
      WHERE v.embedding MATCH ? AND k = ?
    """
    params = [qv.tobytes(), args.k * 4]
    if args.domain:
        sql += " AND c.domain = ?"
        params.append(args.domain)
    sql += " ORDER BY v.distance LIMIT ?"
    params.append(args.k)
    for p, head, snip, dist in db.execute(sql, params):
        head = head or Path(p).stem
        snip = re.sub(r"\s+", " ", snip)[:160]
        print(f"{1-dist:.4f}  {Path(p).relative_to(VAULT)}  «{head}»\n        {snip}")


def cmd_status(args):
    db = open_db(args.db)
    print("docs:", db.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    print("chunks:", db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    for d, n in db.execute("SELECT domain, COUNT(*) FROM chunks GROUP BY domain ORDER BY 2 DESC"):
        print(f"  {d}: {n}")
    print("db size:", f"{os.path.getsize(args.db)/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "query", "status", "shard", "qshards"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--vault", default=VAULT_DEFAULT)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--out", default=os.environ.get("KB_SHARD_DIR", "kb_shards"))
    ap.add_argument("--no-vacuum", action="store_true")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domain", default="")
    a = ap.parse_args()
    {"build": cmd_build, "query": cmd_query, "status": cmd_status,
     "shard": cmd_shard, "qshards": cmd_qshards}[a.cmd](a)


if __name__ == "__main__":
    main()
