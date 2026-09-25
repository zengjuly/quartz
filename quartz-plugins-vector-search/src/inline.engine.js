/* vector-search 浏览器引擎：以字符串内联进站点页面脚本运行。
 * 依赖仅：fetch（服务端检索 API）。
 * 架构（2026-09-22 用户要求）：检索改在服务器端执行（vector_tool/search_server.py），
 *   客户端不再下载 ~70MB 模型/pack/wasm，也不再本地推理；本引擎只负责
 *   UI（输入防抖 → POST /api/search → 渲染结果卡）+ 移动端 TOC 置顶补丁。
 */
(() => {
  const TOP_K = 8;

  // ---------- API ----------
  const API_BASE = (() => {
    const base = (typeof document !== "undefined" &&
      document.body?.dataset?.basepath) || "";
    return base + "/api/search";
  })();

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

  // ---------- 搜索历史（localStorage） ----------
  const HIST_KEY = "vs-search-history";
  const HIST_MAX = 10;
  function getHist() {
    try { const h = JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); return Array.isArray(h) ? h : []; }
    catch (e) { return []; }
  }
  function addHist(q) {
    try {
      const h = getHist().filter((x) => x !== q);
      h.unshift(q);
      localStorage.setItem(HIST_KEY, JSON.stringify(h.slice(0, HIST_MAX)));
    } catch (e) { /* localStorage 不可用时静默 */ }
  }
  function renderHist(box, input) {
    const h = getHist();
    if (!h.length) { box.innerHTML = ""; return; }
    const chips = h.map((x) =>
      '<button class="vs-chip" type="button" data-q="' + esc(x) + '">' + esc(x) + "</button>").join("");
    box.innerHTML = '<div class="vs-history">最近搜索：' + chips +
      '<button class="vs-clear" type="button" title="清除历史">✕</button></div>';
    box.querySelectorAll(".vs-chip").forEach((b) => b.addEventListener("click", () => {
      const q2 = b.dataset.q;
      input.value = q2;
      runSearch(q2, box);
    }));
    const cl = box.querySelector(".vs-clear");
    if (cl) cl.addEventListener("click", () => {
      try { localStorage.removeItem(HIST_KEY); } catch (e) {}
      box.innerHTML = "";
      input.focus();
    });
  }

  // ---------- 检索 ----------
  let seq = 0;
  async function runSearch(query, box) {
    const mySeq = ++seq;
    if (!query || !query.trim()) { renderHist(box, document.querySelector(".vector-search > .search-bar")); return; }
    addHist(query.trim());
    box.innerHTML = '<div class="vs-status">正在检索…</div>';
    try {
      const r = await fetch(API_BASE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: query.trim(), k: TOP_K }),
      });
      if (!r.ok) {
        if (mySeq !== seq) return;
        box.innerHTML = '<div class="vs-status vs-error">检索失败（HTTP ' + r.status + '）。' +
          '<button class="vs-retry" type="button">重试</button></div>';
        const btn = box.querySelector(".vs-retry");
        if (btn) btn.addEventListener("click", () => runSearch(query, box));
        return;
      }
      const data = await r.json();
      if (mySeq !== seq) return;
      const hits = data.results || [];
      if (!hits.length) { box.innerHTML = '<div class="vs-status">无结果。试试换个词。</div>'; return; }
      const terms = (data.terms || []).filter((t) => t.length >= 1);
      const hl = (s) => {                       // 查询词高亮（s 已 esc；terms 逐词高亮）
        let out = s;
        for (const t of terms) {
          const qt = esc(t);
          if (!qt || !out.includes(qt)) continue;
          out = out.split(qt).join('<mark class="vs-hl">' + qt + "</mark>");
        }
        return out;
      };
      const near = (text, toks, w) => {          // 围绕首个命中词截 snippet
        let at = -1;
        for (const t of toks) {
          if (!t) continue;
          const i = text.indexOf(t);
          if (i >= 0 && (at < 0 || i < at)) at = i;
        }
        if (at < 0) return text.slice(0, 150);
        const s = Math.max(0, at - w);
        return (s > 0 ? "…" : "") + text.slice(s, Math.min(text.length, at + w + 90));
      };

      const html = [];
      const seen = new Set();
      for (const h of hits) {
        if (seen.has(h.doc_path)) continue;      // 双保险：同文档只出一卡
        seen.add(h.doc_path);
        const url = docPathToUrl(h.doc_path);
        // 标题优先文档文件名（短、稳定）；heading 太长时不用
        const fileTitle = h.doc_path.split("/").pop().replace(/\.md$/i, "")
          .replace(/^\d{3}-/, "");
        const title = (h.heading && h.heading.length <= 24) ? h.heading : fileTitle;
        const raw = (h.text || "").replace(/[#>*`\[\]!]/g, "").replace(/\s+/g, " ").trim();
        const snip = near(raw, terms.length ? terms : [query.trim()], 60);
        html.push(
          '<a class="result-card" href="' + url + '">' +
          '<h3 class="card-title">' + hl(esc(title)) + "</h3>" +
          '<p class="card-description">' + hl(esc(snip)) + "</p>" +
          '<p class="vs-meta">' + esc((h.domain || "").replace(/\.pack$/, "")) + " · " + esc(h.doc_path) +
          " · 相似度 " + h.score.toFixed(3) + "</p></a>");
      }
      if (mySeq !== seq) return;
      box.innerHTML = '<div class="vs-count">找到 ' + hits.length + ' 条相关结果</div>' + html.join("");
    } catch (e) {
      if (mySeq === seq) box.innerHTML = '<div class="vs-status vs-error">检索失败：' + (e && e.message ? e.message : String(e)) + "</div>";
    }
  }

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // ---------- 移动端 TOC 置顶（站点级补丁） ----------
  // Quartz 在 <1200px 下 `.sidebar.right>.toc{display:none}` 隐藏目录，且右栏被
  // 排在正文之后。此处把 TOC 节点搬进正文顶部的 <details> 折叠块（桌面自动还原）。
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
        timer = setTimeout(() => runSearch(input.value, box), 250);
      });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { input.value = ""; box.innerHTML = ""; input.blur(); }
      });
      input.addEventListener("focus", () => {
        if (!input.value.trim() && !box.querySelector(".vs-history")) renderHist(box, input);
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

  window.__vectorSearch = { runSearch, mount, state: { ready: true } };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
  document.addEventListener("nav", mount);
})();
