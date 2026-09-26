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
  const API_SUGGEST = (() => {
    const base = (typeof document !== "undefined" &&
      document.body?.dataset?.basepath) || "";
    return base + "/api/suggest?q=";
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
  // 常用词引导（服务端 /api/suggest 空查询返回别名表术语，内存缓存一次）
  let HOT_CACHE = null;
  function loadHot(cb) {
    if (HOT_CACHE) { cb(HOT_CACHE); return; }
    fetch(API_SUGGEST, { method: "GET" })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => { HOT_CACHE = (j && j.suggestions) || []; cb(HOT_CACHE); })
      .catch(() => { HOT_CACHE = []; cb([]); });
  }
  function renderHist(box, input) {
    const h = getHist();
    const bind = (sel, fn) => box.querySelectorAll(sel).forEach(fn);
    const render = (hot) => {
      if (!h.length && !hot.length) { box.innerHTML = ""; return; }
      const histHtml = h.length
        ? '<div class="vs-history">最近搜索：' + h.map((x) =>
            '<button class="vs-chip" type="button" data-q="' + esc(x) + '">' + esc(x) + "</button>").join("") +
          '<button class="vs-clear" type="button" title="清除历史">✕</button></div>'
        : "";
      const hotHtml = hot.length
        ? '<div class="vs-hot">常用：' + hot.map((x) =>
            '<button class="vs-chip" type="button" data-q="' + esc(x) + '">' + esc(x) + "</button>").join("") + "</div>"
        : "";
      box.innerHTML = histHtml + hotHtml;
      bind(".vs-chip", (b) => b.addEventListener("click", () => {
        const q2 = b.dataset.q;
        input.value = q2;
        runSearch(q2, box);
      }));
      const cl = box.querySelector(".vs-clear");
      if (cl) cl.addEventListener("click", () => {
        try { localStorage.removeItem(HIST_KEY); } catch (e) {}
        box.innerHTML = "";
        input.focus();
        renderHist(box, input);
      });
    };
    // 历史先渲染（即时），常用词到达后合并渲染
    if (h.length) render([]);
    loadHot((hot) => { if (!input.value.trim()) render(hot); });
  }

  // ---------- 检索 ----------
  let seq = 0;
  function renderError(box, msg, query) {
    box.innerHTML = '<div class="vs-status vs-error">检索失败：' + msg +
      '<button class="vs-retry" type="button">重试</button></div>';
    const btn = box.querySelector(".vs-retry");
    if (btn) btn.addEventListener("click", () => runSearch(query, box));
  }
  function closePanel(input, box) {
    input.value = "";
    box.innerHTML = "";
    input.blur();
    document.body.style.removeProperty("overflow");   // 解锁背景滚动
    document.body.style.removeProperty("padding-right");
  }
  function lockScroll() {                             // 面板有内容时锁背景滚动
    const sw = window.innerWidth - document.documentElement.clientWidth;
    document.body.style.overflow = "hidden";
    if (sw > 0) document.body.style.paddingRight = sw + "px";  // 补滚动条宽度防跳动
  }
  async function runSearch(query, box, k) {
    const mySeq = ++seq;
    const kk = k || TOP_K;
    // 「查看更多」（k > TOP_K）时保留面板滚动位置；新查询/输入变化回顶
    const keepScroll = kk > TOP_K;
    const prevScroll = box.scrollTop;
    if (!query || !query.trim()) { renderHist(box, document.querySelector(".vector-search > .search-bar")); return; }
    addHist(query.trim());
    box.innerHTML = '<div class="vs-status">正在检索…</div>';
    try {
      const q2 = query.trim();
      const [r, sugR] = await Promise.all([
        fetch(API_BASE, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ q: q2, k: kk }),
        }),
        fetch(API_SUGGEST + encodeURIComponent(q2)).catch(() => null),
      ]);
      if (!r.ok) {
        if (mySeq !== seq) return;
        renderError(box, "HTTP " + r.status, query);
        return;
      }
      const data = await r.json();
      if (mySeq !== seq) return;
      const hits = data.results || [];
      const terms = (data.terms || []).filter((t) => t.length >= 1);
      // 猜你想搜（suggest 中非当前查询的词）
      let sug = [];
      try {
        if (sugR && sugR.ok) {
          const sj = await sugR.json();
          sug = (sj.suggestions || []).filter((s) => s !== q2);
        }
      } catch (e) { sug = []; }
      const sugHtml = !hits.length && sug.length
        ? '<div class="vs-suggest">猜你想搜：' + sug.slice(0, 5).map((s) =>
            '<button class="vs-chip" type="button" data-q="' + esc(s) + '">' + esc(s) + "</button>").join("") + "</div>"
        : "";
      if (!hits.length) {
        box.innerHTML = '<div class="vs-status">无结果。试试换个词。</div>' + sugHtml;
        box.querySelectorAll(".vs-suggest .vs-chip").forEach((b) => b.addEventListener("click", () => {
          const inp = document.querySelector(".vector-search > .search-bar");
          if (inp) { inp.value = b.dataset.q; runSearch(b.dataset.q, box); }
        }));
        return;
      }
      const hl = (s) => {                       // 查询词高亮（s 已 esc；terms 逐词高亮）
        let out = s;
        for (const t of terms) {
          const qt = esc(t);
          if (!qt || !out.includes(qt)) continue;
          out = out.split(qt).join('<mark class="vs-hl">' + qt + "</mark>");
        }
        return out;
      };
      const near = (text, toks, w) => {          // snippet：覆盖最多命中词的最短窗口
        const lower = text.toLowerCase();
        const pos = [];
        for (const t of toks) {                  // 收集所有命中位置（多词多位置）
          if (!t) continue;
          const tl = t.toLowerCase();
          let i = -1;
          while ((i = lower.indexOf(tl, i + 1)) !== -1) pos.push(i);
        }
        if (!pos.length) return text.slice(0, 150);
        // 暴力枚举起止命中点，选覆盖词数最多、其次最短的窗口
        let best = null;
        for (const s of pos) {
          for (const e of pos) {
            if (e < s) continue;
            const covered = pos.filter((p) => p >= s && p <= e).length;
            if (!best || covered > best.covered ||
                (covered === best.covered && e - s < best.len)) {
              best = { s, e, covered, len: e - s };
            }
          }
        }
        const start = Math.max(0, best.s - w);
        const end = Math.min(text.length, best.e + w);
        return (start > 0 ? "…" : "") + text.slice(start, end) +
          (end < text.length ? "…" : "");
      };

      const html = [];
      const seen = new Set();
      for (const h of hits) {
        if (seen.has(h.doc_path)) continue;      // 双保险：同文档只出一卡
        seen.add(h.doc_path);
        const url = docPathToUrl(h.doc_path) +
          (h.heading ? "#" + encodeURIComponent(h.heading) : "");   // 锚点直达命中章节
        // 标题优先文档文件名（短、稳定）；heading 太长时不用
        const fileTitle = h.doc_path.split("/").pop().replace(/\.md$/i, "")
          .replace(/^\d{3}-/, "");
        const title = (h.heading && h.heading.length <= 24) ? h.heading : fileTitle;
        const raw = (h.text || "").replace(/[#>*`\[\]!]/g, "").replace(/\s+/g, " ").trim();
        const snip = near(raw, terms.length ? terms : [q2], 60);
        const pct = Math.max(0, Math.min(100, Math.round(h.score * 100)));
        const whyTxt = h.why === "path" ? "路径命中" : h.why === "head" ? "标题命中"
          : h.why === "text" ? "正文命中" : "";
        const whyHtml = whyTxt
          ? '<span class="vs-why" title="词汇加权来源">' + whyTxt +
            (h.boost ? " +" + h.boost.toFixed(2) : "") + "</span>"
          : "";
        html.push(
          '<a class="result-card" href="' + url + '">' +
          '<h3 class="card-title">' + hl(esc(title)) + "</h3>" +
          '<p class="card-description">' + hl(esc(snip)) + "</p>" +
          '<p class="vs-meta"><span class="vs-bar"><span class="vs-bar-fill" style="width:' + pct + '%"></span></span>' +
          whyHtml +
          esc((h.domain || "").replace(/\.pack$/, "")) + " · " + esc(h.doc_path) +
          " · " + h.score.toFixed(3) + "</p></a>");
      }
      if (mySeq !== seq) return;
      lockScroll();                        // 面板有内容 → 锁背景滚动
      const moreHtml = (hits.length >= kk && kk < 30)
        ? '<button class="vs-more" type="button">查看更多</button>' : "";
      box.innerHTML = sugHtml +
        '<div class="vs-count"><span class="vs-n">找到 ' + hits.length + ' 条相关结果</span>' +
        '<button class="vs-close" type="button" title="关闭">✕</button></div>' + html.join("") + moreHtml;
      if (keepScroll) box.scrollTop = prevScroll;   // 查看更多后不丢阅读位置
      const mb = box.querySelector(".vs-more");
      if (mb) mb.addEventListener("click", () => runSearch(q2, box, kk + 22));
      const ic = box.querySelector(".vs-close");
      if (ic) ic.addEventListener("click", () => {
        const inp = document.querySelector(".vector-search > .search-bar");
        if (inp) closePanel(inp, box);
      });
      box.querySelectorAll(".vs-suggest .vs-chip").forEach((b) => b.addEventListener("click", () => {
        const q2b = b.dataset.q;
        const inp = document.querySelector(".vector-search > .search-bar");
        if (inp) { inp.value = q2b; runSearch(q2b, box); }
      }));
    } catch (e) {
      if (mySeq === seq) renderError(box, (e && e.message ? e.message : String(e)), query);
    }
  }

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // ---------- 移动端 TOC 置顶（站点级补丁） ----------
  // Quartz 在 <1200px 下 `.sidebar.right>.toc{display:none}` 隐藏目录，且右栏被
  // 排在正文之后。此处把 TOC 节点搬进正文顶部的 <details> 折叠块（桌面自动还原）。
  const tocWide = window.matchMedia("(min-width: 1200px)");
  const TOC_OPEN_KEY = "vs-toc-open";
  function tocOpenState() {
    try { return localStorage.getItem(TOC_OPEN_KEY) !== "0"; }
    catch (e) { return true; }               // 默认展开
  }
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
      wrap.open = tocOpenState();            // 记住上次展开/收起
      const sum = document.createElement("summary");
      sum.textContent = "目录";
      wrap.appendChild(sum);
      wrap.appendChild(toc);
      wrap.addEventListener("toggle", () => {
        try { localStorage.setItem(TOC_OPEN_KEY, wrap.open ? "1" : "0"); } catch (e) {}
      });
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
      const input = root.querySelector(".search-bar");
      const box = root.querySelector(".vector-results");
      if (!input || !box) return;
      // 按 input 元素判断是否已绑定：Quartz 二次渲染会替换 input 节点，
      // 若用 root 级标记则新 input 永不绑定（输入不搜索、?q= 直达失效）
      if (input.dataset.vsBound) return;
      input.dataset.vsBound = "1";
      // 输入框清空按钮（✕，有内容时显示）——先清掉可能残留的旧节点
      const stale = root.querySelector(".vs-clear-input");
      if (stale) stale.remove();
      const clearBtn = document.createElement("button");
      clearBtn.type = "button";
      clearBtn.className = "vs-clear-input";
      clearBtn.title = "清空";
      clearBtn.textContent = "✕";
      clearBtn.style.display = "none";
      input.parentNode.insertBefore(clearBtn, input.nextSibling);
      clearBtn.addEventListener("click", () => {
        input.value = "";
        box.innerHTML = "";
        clearBtn.style.display = "none";
        input.focus();
        renderHist(box, input);
      });
      let timer = null;
      input.addEventListener("input", () => {
        clearBtn.style.display = input.value ? "block" : "none";
        clearTimeout(timer);
        timer = setTimeout(() => runSearch(input.value, box), 250);
      });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { closePanel(input, box); return; }
        const cards = Array.from(box.querySelectorAll(".result-card"));
        if (e.key === "Enter") {
          // IME 组合中（中文输入法选字）不拦截
          if (e.isComposing || e.keyCode === 229) return;
          e.preventDefault();
          let idx = cards.findIndex((c) => c.classList.contains("vs-active"));
          if (idx < 0) idx = 0;                 // 无选中 → 直达第一个
          if (cards[idx]) cards[idx].click();
          return;
        }
        if (!cards.length) return;
        let idx = cards.findIndex((c) => c.classList.contains("vs-active"));
        if (e.key === "ArrowDown") {
          e.preventDefault();
          idx = idx < 0 ? 0 : (idx + 1) % cards.length;   // 循环：末尾回到第一个
          activate(cards, idx);
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          idx = idx < 0 ? cards.length - 1 : (idx - 1 + cards.length) % cards.length; // 循环
          activate(cards, idx);
        } else if (e.key === "Home") {
          e.preventDefault();
          activate(cards, 0);
        } else if (e.key === "End") {
          e.preventDefault();
          activate(cards, cards.length - 1);
        }
      });
      function activate(cards, idx) {
        cards.forEach((c, i) => c.classList.toggle("vs-active", i === idx));
        cards[idx].scrollIntoView({ block: "nearest" });
      }
      input.addEventListener("focus", () => {
        if (!input.value.trim() && !box.querySelector(".vs-history")) renderHist(box, input);
      });
      // URL 直达：?q=关键词 自动搜索（便于从外部链接分享/跳转）
      try {
        const qp = new URLSearchParams(location.search).get("q");
        if (qp && qp.trim()) {
          input.value = qp;
          clearBtn.style.display = "block";
          runSearch(qp, box);
        }
      } catch (e) { /* ignore */ }
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
  // 兜底：Quartz 会在 nav/渲染过程中重建导航栏 DOM（input 节点被替换），
  // 仅靠 DOMContentLoaded/nav 可能绑不上新 input（表现为输入不搜索、?q= 失效）。
  // 用 MutationObserver 节流（非防抖——页面持续变动时防抖会饿死定时器，
  // 永远不触发），保证变化后 200ms 内必有一次 mount。
  try {
    let moLast = 0;
    let moTimer = null;
    const runMount = () => { moLast = Date.now(); mount(); };
    const mo = new MutationObserver(() => {
      if (Date.now() - moLast > 200) runMount();
      else {
        clearTimeout(moTimer);
        moTimer = setTimeout(runMount, 200);
      }
    });
    mo.observe(document.body, { childList: true, subtree: true });
  } catch (e) { /* MutationObserver 不可用时忽略 */ }
})();
