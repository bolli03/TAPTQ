import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig  
from peft import LoraConfig, TaskType

# 单卡示例:
# accelerate launch --num_processes=1 --gpu_ids 6 LoRATrainer.py --config ./sft_config.json 2>&1 | tee lora_single.log
# 多卡数据并行示例:
# export PATH=/data/ywb/miniconda3/envs/SGuard/bin:$PATH
# accelerate launch --num_processes=2 --gpu_ids 2,5 LoRATrainer.py --config ./sft_config.json --multi_gpu 2>&1 | tee lora4.log
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT Trainer (supports multi-GPU data parallel)")
    parser.add_argument("--config", type=str, default="./sft_config.json", help="训练配置文件路径")
    parser.add_argument("--multi_gpu", action="store_true", help="启用多卡数据并行(DDP)，需配合 accelerate/torchrun")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def build_sft_prompt(tokenizer, user_text, assistant_text):
    messages = [
        {"role": "system", "content": "你是一个安全审核助手。请只输出'安全'或'不安全'。"},
        {"role": "user", "content": f"请判断以下文本是否安全：\n{user_text}"},
        {"role": "assistant", "content": assistant_text}
    ]
    # 全量/SFT训练时不需要 add_generation_prompt=True
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        enable_thinking=False
    )

def prepare_dataset(train_file, eval_file, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    train_raw = load_dataset("json", data_files=train_file, split="train")
    eval_raw = load_dataset("json", data_files=eval_file, split="train")
    
    def map_sft_format(example):
        full_text = build_sft_prompt(tokenizer, example["prompt"], example["label"])
        return {"text": full_text}
    
    train_dataset = train_raw.map(map_sft_format, remove_columns=train_raw.column_names)
    eval_dataset = eval_raw.map(map_sft_format, remove_columns=eval_raw.column_names)
    
    return train_dataset, eval_dataset

def main():
    cli_args = parse_args()
    config = load_config(cli_args.config)
    torch.set_float32_matmul_precision('high')

    enable_multi_gpu = cli_args.multi_gpu or config.get("multi_gpu", False)
    world_size = get_world_size()
    rank = get_rank()

    if enable_multi_gpu and world_size == 1:
        print("[!] 警告: 配置文件或参数开启了 multi_gpu，但当前环境 WORLD_SIZE=1，已自动回退至单卡训练模式。")
        enable_multi_gpu = False

    if rank == 0:
        mode = "DDP多卡" if enable_multi_gpu else "单卡"
        print(f"[*] 训练模式: {mode} | world_size={world_size}")
    
    model_id = config["model_id"]
    dtype = getattr(torch, config["dtype"])

    # 1. 加载基础模型
    # 注意：不要在这里手动调用 .gradient_checkpointing_enable()
    print("[*] 正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    # 显式关闭 KV Cache，这是开启梯度检查点的前置要求
    model.config.use_cache = False

    # 2. 配置 LoRA
    # 我们不在这里用 get_peft_model 包装，而是把配置准备好
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),               # 推荐 r=16
        lora_alpha=config.get("lora_alpha", 32),  # 通常 alpha 是 r 的 2 倍
        target_modules="all-linear",              # 推荐全部线性层，效果最好
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 3. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 4. 准备数据
    train_file = config.get("train_file")
    eval_file = config.get("eval_file")
    train_dataset, eval_dataset = prepare_dataset(train_file, eval_file, model_id)
    
    if rank == 0:
        print(f"[*] SFT 数据加载完成:")
        print(f"    - 训练集数量: {len(train_dataset)}")
        print(f"    - 评估集数量: {len(eval_dataset)}")

    # 5. 配置 SFT 训练参数
    sft_kwargs = dict(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        optim=config["optim"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        tf32=config["tf32"],
        eval_strategy="steps",
        eval_steps=config.get("save_steps", 500),
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        dataset_text_field="text",
        report_to=config["report_to"],
        gradient_checkpointing=True,        # SFTTrainer 会通过这里自动接管并安全地开启梯度检查点
        packing=False,
    )

    # 预留并实现 DDP 相关接口，默认兼容单卡
    if enable_multi_gpu:
        sft_kwargs["ddp_find_unused_parameters"] = config.get("ddp_find_unused_parameters", False)
        ddp_backend = config.get("ddp_backend", "nccl")
        if ddp_backend:
            sft_kwargs["ddp_backend"] = ddp_backend

    sft_args = SFTConfig(**sft_kwargs)

    # 6. 初始化 Trainer
    if rank == 0:
        print("[*] 初始化 Trainer (自动包装 LoRA)...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config             # <--- 关键点：直接把 lora_config 传给 Trainer
    )

    # 打印可训练参数量，确认 LoRA 成功注入
    if rank == 0:
        trainer.model.print_trainable_parameters()

    # 7. 开始训练
    if rank == 0:
        print("[*] 启动 SFT 训练并评估...")
    trainer.train()

    # 8. 保存模型
    # 注意：因为使用了 LoRA，这里保存的仅仅是 Adapter 权重，而不是完整的 4B 模型
    final_save_path = os.path.join(config["output_dir"], config["output_model_name"])
    trainer.save_model(final_save_path)
    if rank == 0:
        print(f"[*] SFT 训练完成，LoRA 权重已保存至: {final_save_path}")

if __name__ == "__main__":
    main()