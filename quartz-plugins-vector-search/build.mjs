// 预构建脚本：node build.mjs
// 产物 dist/components/index.js 完全自包含（ort UMD + 引擎 + 样式内联为字符串），
// Quartz loader 检测到预构建 dist/ 后免 npm install/build，直接 dynamic import。
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.dirname(fileURLToPath(import.meta.url))
const read = (p) => fs.readFileSync(path.join(root, p), "utf-8")

// 1) onnxruntime-web 不再内联：引擎运行时用动态 import() 加载
//    ort.bundle.min.mjs（ESM bundle，自包含 wasm+webgl+webgpu；
//    旧 UMD 版会加载 -threaded.wasm 在非 crossOriginIsolated 页面死锁悬挂）
// 2) 引擎源码（IIFE，自行 import ort 模块）
const engine = read("src/inline.engine.js")
// 3) 样式
const css = read("src/style.css")

const runtime = engine

// 4) 组件模板 → 替换占位符
let component = read("src/component.js")
component = component
  .replace("__ENGINE_SRC__", JSON.stringify(runtime))
  .replace("__CSS_SRC__", JSON.stringify(css))

fs.mkdirSync(path.join(root, "dist/components"), { recursive: true })
fs.writeFileSync(path.join(root, "dist/components/index.js"), component)
// 插件主入口：component-only 插件无处理工厂，仅需可 import
fs.writeFileSync(path.join(root, "dist/index.js"), 'export {}\n')

console.log(
  "built dist/components/index.js:",
  (component.length / 1024).toFixed(1) + "KB (engine",
  (engine.length / 1024).toFixed(1) + "KB)",
)
