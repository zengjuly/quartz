// VectorSearch 组件源码 —— 由 build.mjs 注入运行时字符串后生成 dist/components/index.js
// 占位符（引擎源码与样式的注入点）会被 build.mjs 替换为 JSON 字符串字面量。
import { h } from "preact"

const classNames = (...classes) => classes.filter(Boolean).join(" ")

export const VectorSearch = (userOpts = {}) => {
  const Component = ({ displayClass }) => {
    const placeholder = userOpts.placeholder || "语义搜索…"
    return h("div", { class: classNames(displayClass, "vector-search") }, [
      h("input", {
        class: "search-bar",
        type: "text",
        autocomplete: "off",
        "aria-label": placeholder,
        placeholder,
      }),
      h("div", { class: "vector-results" }),
    ])
  }
  Component.displayName = "VectorSearch"
  Component.afterDOMLoaded = __ENGINE_SRC__
  Component.css = __CSS_SRC__
  return Component
}
