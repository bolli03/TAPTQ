"""
W4A4 部署：使用 ``param/vggt/w4a4.txt`` 时，先用 ``build_and_save_quant_model.py`` 导出
``outputs/w4a4_deploy/``（int4_weights / qwt / non_quant / deploy_config），再用本脚本加载推理。

默认策略：
  - **默认开启 QwT**（与 PTQ+补偿一致）；需要更快速度时用 ``--no-qwt``。
  - **默认开启 torchao**，且默认 ``torchao_int4_packing='tile'``（``TILE_PACKED_TO_4D`` /
    tinygemm）：在 Ampere（sm>=8）上**固定走 INT4 权重 GEMM**，不先尝试依赖 ``mslk`` 的 PLAIN。
    日志里若出现 bfloat16，仅指 **打包阶段** 把 Linear 权重转成 bf16 再交给 ``quantize_``；
    **前向仍是 INT4 权重矩阵乘**，不是整层用 FP 权重 matmul。若需 PLAIN/mslk 路径，传
    ``--torchao-packing plain`` 或 ``auto``。
  - 默认 **禁止** torchao 失败后静默用反量化 FP 权重；旧卡或调试可加 ``--allow-fp-weight-fallback``。

**不显式做 4bit 激活伪量化**（仓库里 ``quant_forward`` 仍是 round 后 ``F.linear``）。

1) 导出权重（与 W4A16 相同，需 hydra 数据配置做 QwT；bit 改为 4+4）::

    cd /path/to/Pi3-evaluation
    python deployment/build_and_save_quant_model.py \\
      ptq.quant_param_file=param/vggt/w4a4.txt \\
      ptq.output_dir=outputs/w4a4_deploy \\
      ptq.bit_setting=[4,4] \\
      ptq.deploy_mode=w4a4

2) 推理::

    python deployment/w4a4_deploy.py --dummy
    python deployment/w4a4_deploy.py --compare-fp --num-views 4
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from deployment.w4a16_deploy import (  # noqa: E402
    TorchaoInt4Packing,
    _print_compare_report,
    forward_inference,
    inference,
    load_w4a16_deployed_model,
    run_fp_vs_deploy_compare,
)


def load_w4a4_deployed_model(
    pretrained_model_path: str,
    deploy_dir: str | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    use_torchao: bool = True,
    use_qwt: bool = True,
    torchao_int4_packing: TorchaoInt4Packing = "tile",
    allow_fp_weight_fallback: bool = False,
    torchao_group_size: int = 32,
):
    """加载 W4A4 导出目录；默认 **QwT 开**、**torchao INT4（默认 tile / tinygemm）**。"""
    if deploy_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        deploy_dir = os.path.join(repo_root, "outputs", "w4a4_deploy")
    return load_w4a16_deployed_model(
        pretrained_model_path,
        deploy_dir,
        device=device,
        dtype=dtype,
        use_torchao=use_torchao,
        torchao_group_size=torchao_group_size,
        torchao_int4_packing=torchao_int4_packing,
        allow_fp_weight_fallback=allow_fp_weight_fallback,
        use_qwt=use_qwt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VGGT W4A4 部署（默认 QwT + torchao INT4 tile / tinygemm）")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VGGT_MODEL_PATH", "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"),
    )
    parser.add_argument(
        "--deploy-dir",
        type=str,
        default=None,
        help="默认: <repo>/outputs/w4a4_deploy",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-torchao", action="store_true")
    parser.add_argument(
        "--no-qwt",
        action="store_true",
        help="不挂载 QwT（更快，与 PTQ+QwT 数值差异更大；默认启用 QwT）",
    )
    parser.add_argument(
        "--torchao-packing",
        type=str,
        choices=["auto", "plain", "tile"],
        default="tile",
        help="INT4 打包：tile=仅 tinygemm(sm>=8)，plain=仅 mslk，auto=先 plain 再 tile（默认 tile）",
    )
    parser.add_argument(
        "--allow-fp-weight-fallback",
        action="store_true",
        help="torchao 打包失败时允许回退为反量化 FP 权重（默认不允许，以保证 INT4 GEMM 或显式失败）",
    )
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--compare-fp", action="store_true")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--img-h", type=int, default=518)
    parser.add_argument("--img-w", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="--compare-fp 时整网与各 head；默认仅主干 aggregator（与 w4a16_deploy 一致）",
    )
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = args.deploy_dir or os.path.join(repo_root, "outputs", "w4a4_deploy")

    if args.compare_fp:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用")
        dev = torch.device(args.device or "cuda")
        report = run_fp_vs_deploy_compare(
            args.model,
            deploy_dir,
            dev,
            dtype=torch.float32,
            use_torchao=not args.no_torchao,
            torchao_int4_packing=args.torchao_packing,
            allow_fp_weight_fallback=args.allow_fp_weight_fallback,
            use_amp_benchmark=args.amp,
            use_qwt=not args.no_qwt,
            num_views=args.num_views,
            img_h=args.img_h,
            img_w=args.img_w,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            aggregator_only=not args.full_model,
        )
        _print_compare_report(report)
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"已写入 {args.save_json}")
        return

    model, meta = load_w4a4_deployed_model(
        args.model,
        deploy_dir,
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
        use_torchao=not args.no_torchao,
        use_qwt=not args.no_qwt,
        torchao_int4_packing=args.torchao_packing,
        allow_fp_weight_fallback=args.allow_fp_weight_fallback,
    )
    print("W4A4 加载:", json.dumps({k: v for k, v in meta.items() if k != "deploy_config"}, indent=2))

    if args.dummy:
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        x = torch.rand(1, 2, 3, 518, 518, device=dev, dtype=dt)
        out = inference(model, x)
        print("keys:", list(out.keys()))


if __name__ == "__main__":
    main()

# cd /root/Pi3-evaluation
# python deployment/w4a4_deploy.py --dummy
# # 或
# python deployment/w4a4_deploy.py --compare-fp --num-views 50