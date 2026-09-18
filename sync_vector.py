import os
import subprocess
import sqlite3
import numpy as np
import onnxruntime as ort

# ================= 路径配置 =================
MODEL_PATH = "bge-small-zh-v1.5.onnx"  # 建议将模型文件直接上传到您的 quartz 私有仓库根目录
DB_PATH = "public/vector.db"           # 自动输出到发布目录，作为静态资源随网页一起下发
CONTENT_DIR = "content"
# ============================================

def get_git_changes():
    """利用 Git 差异历史，精准获取本次提交中发生变动的文件"""
    try:
        # 切换到 content 目录执行 git diff
        cd_cmd = "cd content && "
        changed = subprocess.check_output(cd_cmd + "git diff --name-only --diff-filter=AM HEAD~1 HEAD", shell=True).decode('utf-8').splitlines()
        deleted = subprocess.check_output(cd_cmd + "git diff --name-only --diff-filter=D HEAD~1 HEAD", shell=True).decode('utf-8').splitlines()
        return [os.path.join(CONTENT_DIR, f) for f in changed if f.endswith('.md')], [os.path.join(CONTENT_DIR, f) for f in deleted if f.endswith('.md')]
    except Exception:
        print("无法获取 Git 历史差异（可能是首次运行工作流），自动切换为全量扫描...")
        all_files = []
        for root, _, files in os.walk(CONTENT_DIR):
            for file in files:
                if file.endswith('.md'):
                    all_files.append(os.path.join(root, file))
        return all_files, []

def init_sqlite_vec():
    """初始化数据库基础存储表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        path TEXT PRIMARY KEY,
        content TEXT
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vec_articles (
        path TEXT PRIMARY KEY,
        embedding BLOB
    );
    """)
    conn.commit()
    return conn

def main():
    changed_files, deleted_files = get_git_changes()
    if not changed_files and not deleted_files:
        print("没有检测到任何 Markdown 文本变动，跳过向量库更新。")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = init_sqlite_vec()
    cursor = conn.cursor()

    # 1. 增量删除：处理在 Obsidian 中被您删掉的笔记
    for p in deleted_files:
        print(f"[-] 正在从向量库中移除删除的文件: {p}")
        cursor.execute("DELETE FROM articles WHERE path = ?", (p,))
        cursor.execute("DELETE FROM vec_articles WHERE path = ?", (p,))
    
    # 2. 增量更新：处理您新写或修改的笔记
    if changed_files:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ 警告: 未在框架根目录下找到 ONNX 模型文件 {MODEL_PATH}，跳过向量计算。")
            conn.close()
            return
            
        print(f"[+] 正在初始化 onnxruntime 引擎...")
        ort_session = ort.InferenceSession(MODEL_PATH)
        
        for p in changed_files:
            if not os.path.exists(p): continue
            print(f"[+] 正在针对变动文件进行本地向量化: {p}")
            with open(p, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 【此处后续可填入 Tokenizer 编码逻辑，将文本转化为 input_ids 喂给 ort_session】
            # 这里先转为标准的 512 维二进制 Blob 占位结构存入 SQLite
            mock_vector = np.random.randn(512).astype(np.float32).tobytes() 
            
            cursor.execute("INSERT OR REPLACE INTO articles (path, content) VALUES (?, ?)", (p, text))
            cursor.execute("INSERT OR REPLACE INTO vec_articles (path, embedding) VALUES (?, ?)", (p, mock_vector))
            
    conn.commit()
    conn.close()
    print("✨ 向量库增量更新成功！")

if __name__ == "__main__":
    main()
