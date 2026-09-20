#!/usr/bin/env python3
"""把向量分库导出成纯浏览器可用的静态检索包（无服务端、无 wasm-sqlite）。

用法:
  python vector_tool/emit_web.py --shards <shard目录> \
      --models vector_tool/models --out quartz-app/public/vector-web

产物 (--out):
  manifest.json      domains + 尺寸统计（前端读）
  model.onnx         量化模型原样拷贝（ort-wasm 支持 QOperator int8 算子）
  tokenizer.json     HF tokenizer 原样拷贝（前端 js 实现 WordPiece）
  <domain>.pack.bin  单领域检索包，自定小端格式：

  magic 'QPK1'(4B)
  u16 dim | u32 n_chunks
  然后 5 段，每段 [u32 len][bytes]：
    vectors  int8[dim*n]，行主序，第 i 行 = chunk i（按 emb.id 升序）
    index    n × [u32 path_off][u32 path_len][u32 head_off][u32 head_len]
             （offset 以各自 blob 段起点计）
    paths    utf8 blob，doc_path 去重
    headings utf8 blob
    texts    n × [u32 gz_len][gzip bytes]（chunk 原文，utf8 解压）
"""
import argparse
import json
import shutil
import sqlite3
import struct
from pathlib import Path


def build_pack(db_path: Path, out_path: Path, vault_prefix: str = "") -> dict:
    db = sqlite3.connect(db_path)
    dim = int(db.execute("SELECT v FROM meta WHERE k='dim'").fetchone()[0])
    vault = str(Path(vault_prefix).resolve()) if vault_prefix else None
    # chunk i = emb 按 id 升序的第 i 条；chunks 行一一对应
    rows = list(db.execute(
        "SELECT c.doc_path, c.heading, c.text FROM emb v "
        "JOIN chunks c ON c.id=v.id ORDER BY v.id"))
    vecs = [v for (v,) in db.execute("SELECT v FROM emb ORDER BY id")]
    db.close()
    if vault:
        # 绝对路径归一为 vault 相对路径（防把构建机路径烤进公开站）
        rows = [(r[0][len(vault) + 1:] if r[0].startswith(vault + "/")
                 else Path(r[0]).name, *r[1:]) for r in rows]
    assert len(rows) == len(vecs)

    vec_blob = b"".join(vecs)
    assert len(vec_blob) == len(rows) * dim, (
        len(vec_blob), len(rows), dim)

    paths: dict[bytes, tuple[int, int]] = {}
    path_blob = bytearray()
    head_blob = bytearray()
    idx_bin = bytearray()
    text_blob = bytearray()
    for doc_path, head, gz_text in rows:
        pb = doc_path.encode("utf-8")
        po, pl = paths.get(pb, (len(path_blob), len(pb)))
        if pb not in paths:
            paths[pb] = (po, pl)
            path_blob += pb
        hb = (head or "").encode("utf-8")
        idx_bin += struct.pack("<IIII", po, pl, len(head_blob), len(hb))
        head_blob += hb
        tb = gz_text if isinstance(gz_text, (bytes, bytearray)) \
            else __import__("zlib").compress(str(gz_text).encode("utf-8"), 6)
        text_blob += struct.pack("<I", len(tb)) + tb

    with out_path.open("wb") as f:
        f.write(b"QPK1")
        f.write(struct.pack("<HI", dim, len(rows)))
        for blob in (vec_blob, bytes(idx_bin), bytes(path_blob),
                     bytes(head_blob), bytes(text_blob)):
            f.write(struct.pack("<I", len(blob)))
            f.write(blob)
    st = out_path.stat().st_size
    return {"chunks": len(rows), "pack_bytes": st,
            "vec_bytes": len(vec_blob), "text_bytes": len(text_blob)}


def copy_ort_runtime(out: Path, ort_dist: Path):
    """拷贝 onnxruntime-web ESM bundle 运行时到 out/ort/（前端引擎运行时
    import ort.bundle.min.mjs，它再拉 jsep wasm）。
    numThreads=1 不依赖 SharedArrayBuffer，无需 COOP/COEP 头。
    注意：不能用 ort.wasm.min.js(UMD)——它加载 -threaded.wasm 在无
    crossOriginIsolation 页面会死锁悬挂。"""
    dst = out / "ort"
    dst.mkdir(parents=True, exist_ok=True)
    names = ["ort.bundle.min.mjs",
             "ort-wasm-simd-threaded.jsep.wasm",
             "ort-wasm-simd-threaded.jsep.mjs"]
    for n in names:
        src = ort_dist / n
        if not src.exists():
            raise SystemExit(f"[emit] 缺少 ort 运行时文件: {src}")
        shutil.copy2(src, dst / n)
    tot = sum((dst / n).stat().st_size for n in names)
    print(f"[emit] ort runtime -> {dst}/ ({tot/1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="cmd_shard 产物目录")
    ap.add_argument("--models", required=True, help="vector_tool/models 目录")
    ap.add_argument("--out", required=True, help="输出目录(public/vector-web)")
    ap.add_argument("--vault-prefix", default="",
                    help="shard 时 --vault 的路径（用于把 doc_path 归一为相对路径）")
    ap.add_argument("--ort-dist", default="",
                    help="onnxruntime-web/dist 路径（默认自动找 ./ 与 ./quartz-app/）")
    a = ap.parse_args()
    shards, models, out = Path(a.shards), Path(a.models), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mf = json.loads((shards / "manifest.json").read_text(encoding="utf-8"))
    web = {"format": "web-v1", "dim": mf["dim"], "model": mf["model"],
           "metric": "cosine-int8", "updated": mf["updated"],
           "files": {"model": "model.onnx", "tokenizer": "tokenizer.json"},
           "domains": []}
    for d in mf["domains"]:
        if not d["chunks"]:
            continue
        name = d["domain"] + ".pack.bin"
        stats = build_pack(shards / d["file"], out / name, a.vault_prefix)
        web["domains"].append({"domain": d["domain"], "file": name,
                               "docs": d["docs"], **stats})
        print(f"[emit] {name}: chunks={stats['chunks']} "
              f"pack={stats['pack_bytes']/1e6:.2f}MB", flush=True)
    (out / "manifest.json").write_text(
        json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(models / "model_quantized.onnx", out / "model.onnx")
    shutil.copy2(models / "tokenizer.json", out / "tokenizer.json")
    # ort wasm 运行时随向量包同源分发（插件 JS 内联的是 UMD loader 本体）
    ort_dist = Path(a.ort_dist) if a.ort_dist else next(
        (p for p in [
            Path("node_modules/onnxruntime-web/dist"),
            Path("quartz-app/node_modules/onnxruntime-web/dist"),
        ] if p.exists()), None)
    if ort_dist:
        copy_ort_runtime(out, ort_dist)
    else:
        print("[emit] 警告: 未找到 onnxruntime-web/dist，跳过 ort 运行时拷贝"
              "（用 --ort-dist 指定）")
    tot = sum(d["pack_bytes"] for d in web["domains"])
    print(f"[emit] domains={len(web['domains'])} packs={tot/1e6:.1f}MB -> {out}")


if __name__ == "__main__":
    main()
