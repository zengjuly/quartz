import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

// Advanced: pass callback functions that can't be expressed in YAML
ExternalPlugin.Explorer({
  filterFn: (node) => {
    // 排除掉不希望渲染的、含有海量无意义碎文件的文件夹名称
    const exclude = ["templates", "assets", "archive", "pasted"]
    return !exclude.includes(node.name)
  },

})

const config = await loadQuartzConfig()
export default config
export const layout = await loadQuartzLayout()
