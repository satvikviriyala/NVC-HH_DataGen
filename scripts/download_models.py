import os
from huggingface_hub import snapshot_download

# SOTA Native 2026 Model List
MODELS = {
    "GLM-4.7": "zai-org/GLM-4.7",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "Qwen3-235B": "Qwen/Qwen3-235B-A22B",
    "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
}

CACHE_DIR = "/root/.cache/huggingface"

def download_all():
    print(f"💾 Downloading Native SOTA Models to {CACHE_DIR}...")
    for name, repo in MODELS.items():
        print(f"⬇️  Fetching {name} ({repo})...")
        try:
            snapshot_download(
                repo_id=repo, 
                cache_dir=CACHE_DIR,
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"], 
                local_files_only=False
            )
            print(f"✅ {name} Downloaded.")
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")

if __name__ == "__main__":
    download_all()
