/* vector-search 浏览器引擎：以字符串内联进站点页面脚本运行。
 * 依赖仅：onnxruntime-web ESM bundle（ensureReady 时动态 import）、
 *         fetch + DecompressionStream("gzip")。
 * 流程：懒加载 manifest/tokenizer/model/packs → 输入防抖 →
 *       全域 int8 点积 → 跨域 top-k → 解压 chunk 文本 → 渲染卡片。
 * pack 二进制格式与打分公式（round(f32*127) 点积 / 127^2）对齐
 * vector_tool/emit_web.py 与 kb_vec.cmd_qshards。
 */
(() => {
  let ort = null; // ensureReady 时动态 import 的 ort 模块命名空间
  const VEC_BASE = (() => {
    const base = (typeof document !== "undefined" &&
      document.body?.dataset?.basepath) || "";
    return base + "/vector-web/";
  })();
  const QUERY_INSTRUCT = "为这个句子生成表示以用于检索相关文章：";
  const MAX_TOKENS = 480;
  const TOP_K = 8;

  const state = {
    ready: false, loading: null,
    tok: null, sess: null, manifest: null,
  };

  // ort 运行时（UMD 前置内联，见 build.mjs）；wasm 文件随向量库同源分发
  const ORT_THREADS = 1;
  function configureOrt() {
    if (!ort) throw new Error('onnxruntime 未加载');
    ort.env.wasm.wasmPaths = VEC_BASE + "ort/";
    ort.env.wasm.numThreads = ORT_THREADS;
    ort.env.wasm.proxy = false;
  }

  // ---------- BERT WordPiece 分词（与 HF Rust 实现对分验证一致） ----------
  function makeBertTokenizer(vocabObj) {
    const isPunct = (cp) => /[\s\p{P}]/u.test(String.fromCodePoint(cp));
    const isChinese = (cp) =>
      (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
      (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x20000 && cp <= 0x2a6df) ||
      (cp >= 0x2a700 && cp <= 0x2ceaf) || (cp >= 0x2ceb0 && cp <= 0x2ebe0) ||
      cp === 0x3005;

    function normalize(text) {
      let out = "";
      for (const ch of text) {
        const cp = ch.codePointAt(0);
        if (cp === 0 || (cp >= 1 && cp <= 8) || cp === 11 || cp === 12 ||
            (cp >= 14 && cp <= 31) || cp === 127) continue;
        out += (ch === "\n" || ch === "\r" || ch === "\t") ? " " : ch;
      }
      let res = "";
      for (const ch of out) res += isChinese(ch.codePointAt(0)) ? " " + ch + " " : ch;
      return res;
    }

    function preTokenize(text) {
      const words = [];
      for (const ws of text.split(/\s+/)) {
        if (!ws) continue;
        let cur = "", prevPunct = null;
        for (const ch of ws) {
          const p = isPunct(ch.codePointAt(0));
          if (prevPunct !== null && p !== prevPunct && cur) { words.push(cur); cur = ""; }
          cur += ch;
          prevPunct = p;
        }
        if (cur) words.push(cur);
      }
      return words;
    }

    function wordPiece(word) {
      const chars = Array.from(word);
      if (chars.length > 100) return [vocabObj["[UNK]"]];
      const ids = [];
      let s = 0;
      while (s < chars.length) {
        let e = chars.length, cur = null;
        while (s < e) {
          const tok = (s > 0 ? "##" : "") + chars.slice(s, e).join("");
          if (tok in vocabObj) { cur = tok; break; }
          e -= 1;
        }
        if (cur === null) return [vocabObj["[UNK]"]];
        ids.push(vocabObj[cur]);
        s = e;
      }
      return ids;
    }

    return function tokenize(text) {
      const ids = [vocabObj["[CLS]"]];
      for (const w of preTokenize(normalize(text))) {
        for (const t of wordPiece(w)) ids.push(t);
        if (ids.length >= MAX_TOKENS - 1) break;
      }
      ids.push(vocabObj["[SEP]"]);
      return ids;
    };
  }

  // ---------- 资源 ----------
  // 弹性加载：分片 Range 请求 + 失败重试（指数退避）+ Cache API 持久缓存
  // + 实时进度回调。动机（2026-09-21 用户实测）：公网隧道下 24MB model.onnx
  // / 26MB pack 多次在 11~18MB 处断流，普通 fetch 一次失败即全盘报废，
  // UI 永远停在"正在加载语义模型…"。nginx 已支持 Range（默认 on），
  // 分片越小单次失败重传的代价越小。
  const CHUNK = 2 * 1024 * 1024        // 2MB 分片（隧道 RTT 高时仍可控）
  const RETRIES = 8                    // 单分片最多重试次数（指数退避 0.5s 起）
  const fetchTimers = {}               // path -> {received, total}
  let progressCb = null                // ensureReady 设置，驱动 UI 进度条

  function reportProgress(path, received, total) {
    fetchTimers[path] = { received, total }
    if (progressCb) progressCb()
  }

  async function fetchWithRetry(url, headers, attempt = 0) {
    try {
      const ctl = new AbortController()
      const kill = () => ctl.abort()
      const t = setTimeout(kill, 60000)           // 单请求 60s 硬超时（防永久悬挂）
      let resp
      try {
        resp = await fetch(url, { ...headers, signal: ctl.signal })
        if (!resp.ok) throw new Error("HTTP " + resp.status)
        if (!resp.body) throw new Error("no body")
        const reader = resp.body.getReader()
        const chunks = []
        let got = 0
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value); got += value.byteLength
        }
        const buf = new Uint8Array(got)
        let o = 0
        for (const c of chunks) { buf.set(c, o); o += c.byteLength }
        return buf
      } finally { clearTimeout(t) }
    } catch (e) {
      if (attempt >= RETRIES) throw e
      await new Promise(r2 => setTimeout(r2, Math.min(500 * 2 ** attempt, 8000)))
      return fetchWithRetry(url, headers, attempt + 1)
    }
  }

  async function fetchBuf(p) {
    const url = VEC_BASE + p
    // 1) Cache API 命中直接返回（持久缓存，浏览器重启后仍在）
    const cache = "caches" in window ? await caches.open("vector-web-v1") : null
    if (cache) {
      const hit = await cache.match(url)
      if (hit) {
        const ab = await hit.arrayBuffer()
        reportProgress(p, ab.byteLength, ab.byteLength)
        return ab
      }
    }
    // 2) HEAD 探测大小；不支持 Range 的服务器/代理回退整文件 fetch
    let total = 0, accepts = false
    try {
      const h = await fetch(url, { method: "HEAD" })
      if (h.ok) {
        total = +(h.headers.get("content-length") || 0)
        accepts = (h.headers.get("accept-ranges") || "").includes("bytes")
      }
    } catch (e) { /* HEAD 失败不阻塞，走整文件回退 */ }
    if (!accepts || !total) {
      const buf = await fetchWithRetry(url, {})
      if (cache) { try { await cache.put(url, new Response(buf)) } catch (e) {} }
      reportProgress(p, buf.length, buf.length)
      return buf.buffer
    }
    // 3) 分片 Range 下载（每片独立重试；中断后下次仅补缺失分片）
    const parts = new Array(Math.ceil(total / CHUNK))
    const loaded = new Uint8Array(total)
    let done = 0
    reportProgress(p, 0, total)
    for (let i = 0; i < parts.length; i++) {
      const start = i * CHUNK
      const end = Math.min(start + CHUNK, total) - 1
      const b = await fetchWithRetry(url, {
        method: "GET",
        headers: { Range: `bytes=${start}-${end}` },
      })
      loaded.set(b, start)
      done += b.length
      reportProgress(p, done, total)
    }
    const out = loaded.buffer
    if (cache) { try { await cache.put(url, new Response(out)) } catch (e) {} }
    return out
  }
  const fetchJson = async (p) => JSON.parse(new TextDecoder().decode(await fetchBuf(p)));

  // pack 解码（格式：'QPK1' u16 dim u32 n + 5 段 [u32 len][bytes] 交替）
  function decodePack(buf) {
    const dv = new DataView(buf);
    const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
    if (magic !== "QPK1") throw new Error("bad pack magic: " + magic);
    const dim = dv.getUint16(4, true);
    const n = dv.getUint32(6, true);
    // 逐段读 [u32 len]，记录每段数据起点：vec, index, paths, heads, texts
    const seg = [], offs = [];
    let o = 10;
    for (let i = 0; i < 5; i++) {
      const len = dv.getUint32(o, true);
      offs.push(o + 4); seg.push(len); o += 4 + len;
    }
    const p = {
      dim, n, buf, dv,
      vecs: new Int8Array(buf, offs[0], seg[0]),
      // index 段起点不保证 4 字节对齐，只能按 u32 元素经 DataView 读
      indexOff: offs[1], indexLen: seg[1],
      paths: new Uint8Array(buf, offs[2], seg[2]),
      heads: new Uint8Array(buf, offs[3], seg[3]),
      textOff: offs[4],
      _textIdx: null,
    };
    p.indexAt = (k) => dv.getUint32(p.indexOff + k * 4, true);
    return p;
  }

  function textIndexOf(pack) {
    if (!pack._textIdx) {
      const offs = new Uint32Array(pack.n);
      let o = pack.textOff;
      for (let i = 0; i < pack.n; i++) { offs[i] = o; o += 4 + pack.dv.getUint32(o, true); }
      pack._textIdx = offs;
    }
    return pack._textIdx;
  }

  async function inflate(bytes, format) {
    // 不用 Blob.stream()/Response（headless shell 下 Blob.stream 抛
    // "Failed to fetch"），手工喂 DecompressionStream 双 pipe 读干。
    const ds = new DecompressionStream(format);
    const writer = ds.writable.getWriter();
    const chunks = [];
    const reader = ds.readable.getReader();
    const pump = (async () => {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
    })();
    writer.write(bytes);
    writer.close();
    await pump;
    let len = 0;
    for (const c of chunks) len += c.length;
    const out = new Uint8Array(len);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return new TextDecoder("utf-8").decode(out);
  }

  async function ensureReady(onProgress) {
    if (state.ready) return;
    if (state.loading) return state.loading;
    progressCb = typeof onProgress === "function" ? onProgress : null
    state.loading = (async () => {
      if (!ort) ort = await import(VEC_BASE + "ort/ort.bundle.min.mjs");
      configureOrt();
      state.manifest = await fetchJson("manifest.json");
      const raw = await fetchJson("tokenizer.json");
      state.tok = makeBertTokenizer(raw.model.vocab);
      const modelBuf = await fetchBuf("model.onnx");
      state.sess = await ort.InferenceSession.create(modelBuf, {
        graphOptimizationLevel: "all", executionMode: "sequential",
      });
      state.packs = [];
      for (const d of state.manifest.domains) {
        state.packs.push({ domain: d.domain, pack: decodePack(await fetchBuf(d.file)) });
      }
      state.ready = true;
      progressCb = null
    })();
    try {
      await state.loading;
    } catch (e) {
      state.loading = null;
      progressCb = null
      throw e;
    }
  }

  // ---------- 推理：query embedding（CLS pooling + L2 + int8） ----------
  async function embedQuery(text) {
    const ids = state.tok(QUERY_INSTRUCT + text);
    const S = ids.length;
    const i64 = new BigInt64Array(S);
    for (let i = 0; i < S; i++) i64[i] = BigInt(ids[i]);
    const mask = new BigInt64Array(S).fill(1n);
    const feeds = {
      input_ids: new ort.Tensor("int64", i64, [1, S]),
      attention_mask: new ort.Tensor("int64", mask, [1, S]),
    };
    if (state.sess.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(S), [1, S]);
    }
    const out = await state.sess.run(feeds);
    const t = out[state.sess.outputNames[0]];
    const H = t.dims[2];
    const cls = t.data.subarray(0, H);
    let norm = 0;
    for (let i = 0; i < H; i++) norm += cls[i] * cls[i];
    norm = Math.sqrt(norm) || 1;
    const q8 = new Int32Array(H);
    for (let i = 0; i < H; i++) q8[i] = Math.round((cls[i] / norm) * 127);
    return q8;
  }

  // ---------- doc_path -> 站点 URL（对齐 slugifyFilePath/simplifySlug） ----------
  function slugifyPath(s) {
    return s.split("/")
      .map((seg) => seg.replace(/\s/g, "-").replace(/&/g, "-and-")
        .replace(/%/g, "-percent").replace(/\?/g, "").replace(/#/g, "")
        .replace(/[<>:"|*]/g, "").toLowerCase())
      .join("/").replace(/\/$/, "");
  }

  function docPathToUrl(p) {
    let fp = p.replace(/^\//, "").replace(/\.(md|html)$/i, "");
    let slug = slugifyPath(fp);
    const segs = slug.split("/");
    if (segs.length >= 2 && segs[segs.length - 1] === segs[segs.length - 2]) {
      segs[segs.length - 1] = "index";
      slug = segs.join("/");
    }
    if (slug === "index") slug = "";
    else if (slug.endsWith("/index")) slug = slug.slice(0, -"index".length);
    else if (slug.endsWith("index")) slug = slug.slice(0, -"index".length);
    const base = document.body?.dataset?.basepath || "";
    return base + "/" + slug;
  }

  // ---------- 检索 ----------
  let seq = 0;
  async function runSearch(query, box) {
    const mySeq = ++seq;
    if (!query || !query.trim()) { box.innerHTML = ""; return; }
    box.innerHTML = '<div class="vs-status">正在加载语义模型（首次 ~70MB，之后浏览器缓存）…</div>';
    try {
      await ensureReady(() => {
        if (mySeq !== seq) return;
        let got = 0, tot = 0;
        for (const k in fetchTimers) { got += fetchTimers[k].received; tot += fetchTimers[k].total }
        if (tot > 0) {
          const pct = Math.round((got / tot) * 100)
          box.innerHTML = '<div class="vs-status">正在加载语义模型 ' + pct + '%（' +
            (got / 1048576).toFixed(1) + ' / ' + (tot / 1048576).toFixed(1) + ' MB）…</div>'
        }
      });
    } catch (e) {
      if (mySeq === seq) box.innerHTML = '<div class="vs-status vs-error">向量库加载失败：' + e.message + "；请重试（弱网下会自动断点续传）</div>";
      return;
    }
    if (mySeq !== seq) return;
    box.innerHTML = '<div class="vs-status">检索中…</div>';

    try {
      const q8 = await embedQuery(query.trim());
      if (mySeq !== seq) return;

      const hits = [];
    for (const { domain, pack } of state.packs) {
      if (pack.dim !== q8.length) continue;
      const sims = new Float64Array(pack.n);
      for (let i = 0; i < pack.n; i++) {
        let acc = 0;
        const row = i * pack.dim;
        for (let j = 0; j < pack.dim; j++) acc += pack.vecs[row + j] * q8[j];
        sims[i] = acc;
      }
      const kk = Math.min(TOP_K, pack.n);
      let top = [];
      for (let i = 0; i < pack.n; i++) {
        if (top.length < kk) { top.push(i); }
        else {
          let worst = 0;
          for (const t of top) if (sims[t] < sims[worst]) worst = t;
          if (sims[i] > sims[worst]) top[top.indexOf(worst)] = i;
        }
      }
      top.sort((a, b) => sims[b] - sims[a]);
      for (const i of top) hits.push({ score: sims[i] / 16129, domain, pack, i });
    }
    hits.sort((a, b) => b.score - a.score);
    if (mySeq !== seq) return;
    const top = hits.slice(0, TOP_K);
    if (!top.length) { box.innerHTML = '<div class="vs-status">无结果。</div>'; return; }

    const dec = new TextDecoder();
    const html = [];
    for (const h of top) {
      const b = h.i * 4;
      const idx = (k) => h.pack.indexAt(k);
      const docPath = dec.decode(h.pack.paths.subarray(idx(b), idx(b) + idx(b + 1)));
      const heading = dec.decode(h.pack.heads.subarray(idx(b + 2), idx(b + 2) + idx(b + 3)));
      const o = textIndexOf(h.pack)[h.i];
      const text = await inflate(new Uint8Array(h.pack.buf, o + 4, h.pack.dv.getUint32(o, true)), "deflate");
      if (mySeq !== seq) return;
      const url = docPathToUrl(docPath);
      const title = heading || docPath.split("/").pop().replace(/\.md$/i, "");
      const snip = text.replace(/[#>*`\[\]!]/g, "").replace(/\s+/g, " ").trim().slice(0, 150);
      html.push(
        '<a class="result-card" href="' + url + '">' +
        '<h3 class="card-title">' + esc(title) + "</h3>" +
        '<p class="card-description">' + esc(snip) + "</p>" +
        '<p class="vs-meta">' + esc(h.domain) + " · " + esc(docPath) +
        " · 相似度 " + h.score.toFixed(3) + "</p></a>");
    }
    if (mySeq !== seq) return;
    box.innerHTML = html.join("");
    } catch (e) {
      if (mySeq === seq) box.innerHTML = '<div class="vs-status vs-error">检索失败：' + (e && e.message ? e.message : String(e)) + "</div>";
    }
  }

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // ---------- 移动端 TOC 置顶（站点级补丁） ----------
  // Quartz 在 <1200px 下 `.sidebar.right>.toc{display:none}` 隐藏目录，且右栏被
  // 排在正文之后。此处把 TOC 节点搬进正文顶部的 <details> 折叠块（桌面自动还原）。
  // 放在本插件是因为它是本站唯一的站点级 afterDOMLoaded 钩子（custom.scss 只能改
  // 样式，搬不动 DOM）。
  const tocWide = window.matchMedia("(min-width: 1200px)");
  function placeToc() {
    const article = document.querySelector(".center article") || document.querySelector("article");
    if (!article) return;
    const moved = document.querySelector(".vs-mobile-toc");
    const toc = (moved && moved.querySelector(".toc")) ||
      document.querySelector(".sidebar.right > .toc");
    if (!toc || !toc.querySelector("a")) return;  // 无标题的笔记不显示空目录框
    if (!tocWide.matches) {
      if (moved) return;
      const wrap = document.createElement("details");
      wrap.className = "vs-mobile-toc";
      wrap.open = true;
      const sum = document.createElement("summary");
      sum.textContent = "目录";
      wrap.appendChild(sum);
      wrap.appendChild(toc);
      article.insertBefore(wrap, article.firstChild);
    } else if (moved) {
      const sb = document.querySelector(".sidebar.right");
      const graph = sb && sb.querySelector(".graph");
      if (graph) graph.after(toc);
      else if (sb) sb.appendChild(toc);
      moved.remove();
    }
  }
  if (tocWide.addEventListener) tocWide.addEventListener("change", placeToc);
  else if (tocWide.addListener) tocWide.addListener(placeToc);

  // ---------- UI 挂载 ----------
  function mount() {
    document.querySelectorAll(".vector-search").forEach((root) => {
      if (root.dataset.vsMounted) return;
      root.dataset.vsMounted = "1";
      const input = root.querySelector(".search-bar");
      const box = root.querySelector(".vector-results");
      if (!input || !box) return;
      let timer = null;
      input.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(() => runSearch(input.value, box), 450);
      });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { input.value = ""; box.innerHTML = ""; input.blur(); }
      });
    });
    placeToc();
  }

  // 点击面板外关闭结果（全屏浮层形态下必须有，模块级只注册一次）
  document.addEventListener("click", (e) => {
    document.querySelectorAll(".vector-search").forEach((root) => {
      if (root.contains(e.target)) return;
      const b = root.querySelector(".vector-results");
      if (b && b.innerHTML) b.innerHTML = "";
    });
  });

  window.__vectorSearch = { runSearch, mount, state };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
  document.addEventListener("nav", mount);
})();
