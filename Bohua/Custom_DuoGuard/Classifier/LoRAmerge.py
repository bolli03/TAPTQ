import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==========================================
# 1. 路径配置 (请确保与你训练时的路径一致)
# ==========================================
base_model_path = "/data/ywb/Local_LLMs/Qwen3-8B"
# base_model_path = "/data/ywb/Local_LLMs/Dolphin3.0-Llama3.1-8B"
lora_model_path = "/data/ywb/Bohua/Trainer/Custom_DuoGuard/CHECKPOINT/C/Qwen3-8BNOPE-iter2/Qwen3-8BNOPE-iter2" 
output_merged_path = "/data/ywb/Bohua/Trainer/Custom_DuoGuard/Classifier/NOPE/Qwen3-8BNOPE-iter2"

# ==========================================
# 2. 多卡配置
# ==========================================
# 推荐: 在启动脚本前设置 CUDA_VISIBLE_DEVICES，例如:
# CUDA_VISIBLE_DEVICES=6 python /data/ywb/Bohua/Trainer/Custom_DuoGuard/Classifier/LoRAmerge.py
# 这样脚本内部看到的是逻辑 GPU 0 和 1。
use_multi_gpu_auto = True
max_memory_per_gpu = "48GB"  # 可按你的显存调整，设为 None 则不限制


def build_max_memory_map(limit_per_gpu: str | None):
    if not limit_per_gpu:
        return None
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return None
    return {i: limit_per_gpu for i in range(gpu_count)}

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 设备，请在 GPU 环境下运行。")

    visible_gpus = torch.cuda.device_count()
    print(f"[*] 检测到可见 GPU 数量: {visible_gpus}")
    print(f"[*] 1. 正在加载基础模型: {base_model_path}")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if use_multi_gpu_auto:
        load_kwargs["device_map"] = "auto"
        max_memory_map = build_max_memory_map(max_memory_per_gpu)
        if max_memory_map is not None:
            load_kwargs["max_memory"] = max_memory_map

    # 通过 device_map="auto" 自动切分到可见 GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **load_kwargs,
    )

    print(f"[*] 2. 正在加载 LoRA 权重并附加到基础模型: {lora_model_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.bfloat16,
    )

    print("[*] 3. 正在 GPU 上执行物理合并 (Merge and Unload)...")
    # 此时的矩阵合并运算 (原权重 + 缩放后的 LoRA 权重) 会调用 CUDA 核心极速计算
    merged_model = peft_model.merge_and_unload()

    print(f"[*] 4. 正在保存合并后的完整模型至: {output_merged_path}")
    merged_model.save_pretrained(output_merged_path, safe_serialization=True)

    print("[*] 5. 正在保存 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_merged_path)

    print("\n[+] 搞定！GPU 合并完成。")

if __name__ == "__main__":
    main()