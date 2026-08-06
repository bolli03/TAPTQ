"""
从 HuggingFace 镜像站下载 Qwen3Guard-Stream-8B 安全审核模型到本地目录。

Qwen3Guard 用于检测 prompt 和 response 中的不安全内容，作为安全分类器使用。
"""

import os
from huggingface_hub import snapshot_download

# 待下载的模型标识符（HuggingFace repo ID）
MODEL_ID = "Qwen/Qwen3-8B"
# 模型保存的本地目标路径
SAVE_PATH = "/root/autodl-tmp/Local_LLMs/Qwen3-8B"

# 使用 HF 镜像站加速下载，避免国内网络访问 HuggingFace 官方站点的延迟和中断
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_model():
    """下载模型全部文件（含权重、配置、tokenizer 等）到本地。"""
    print(f"[*] 开始下载模型: {MODEL_ID}")
    print(f"[*] 目标路径: {SAVE_PATH}")

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"[+] 已创建目录: {SAVE_PATH}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=SAVE_PATH,
        local_dir_use_symlinks=False,  # 复制实际文件而非符号链接，确保模型目录可独立使用
        resume_download=True,           # 支持断点续传，避免网络中断后重复下载
        token=None,                     # 公开模型无需认证 token
        max_workers=8                   # 并行下载线程数，加速大文件传输
    )

if __name__ == "__main__":
    download_model()