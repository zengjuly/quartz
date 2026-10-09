import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import Component from "./quartz/components" 

const config = await loadQuartzConfig()

// 注入高级硬切除过滤回调函数，将海量碎文件剔除出 DOM 树以解决加载慢问题
if (config.plugins && config.plugins.emitters) {
  // 寻找到配置中的 Explorer  emitter 并注入 filterFn
  const explorerPlugin = config.plugins.emitters.find(p => p.name === "Explorer")
  if (explorerPlugin) {
    explorerPlugin.cfg = {
      ...explorerPlugin.cfg,
      filterFn: (node) => {
        const exclude = ["templates", "assets", "archive", "pasted"]
        return !exclude.includes(node.name)
      }
    }
  }
}


export default config
export const layout = await loadQuartzLayout()
