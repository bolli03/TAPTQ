#!/usr/bin/env python3
"""
VGGT W4A8：wrap + 从 JSON 加载量化参数（不跑 calibration），导出整型权重。
整型计算逻辑写在本脚本内，不修改 quant_layers/linear.py 与 PTQ/utils/integer.py。

  cd /root/Pi3-evaluation
  python mv_recon/save_vggt_w4_int_weights.py --quant-json param/vggt/channelwise_8scan_w4a8.json
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT

import mv_recon.benchmark_common as bc


def int_weights_from_wrapped(wrapped_modules: dict) -> dict[str, torch.Tensor]:
    """与 PTQSLQuantLinear.quant_weight_bias 一致的整型档位；MinMax 层用平面 weight。"""

    out: dict[str, torch.Tensor] = {}
    for name, m in wrapped_modules.items():
        if not hasattr(m, "weight") or not hasattr(m, "w_interval"):
            continue
        qmax = m.w_qmax
        try:
            if all(hasattr(m, a) for a in ("n_V", "crb_rows", "n_H", "crb_cols")):
                w = (
                    m.weight.view(m.n_V, m.crb_rows, m.n_H, m.crb_cols) / m.w_interval
                ).round_().clamp_(-qmax, qmax - 1)
                w_int = w.view_as(m.weight)
            else:
                w_int = (m.weight / m.w_interval).round_().clamp_(-qmax, qmax - 1)
            out[name] = w_int.cpu().detach().to(torch.int8)
        except Exception:
            continue
    return out


def main() -> None:
    proj = Path(__file__).resolve().parents[1]
    os.chdir(proj)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant-json",
        type=str,
        default=str(proj / "param/vggt/channelwise_8scan_w4a8.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(proj / "param/vggt/channelwise_8scan_w4a8_int_weights.pth"),
    )
    parser.add_argument("--config-name", type=str, default="PTQ4ViT_channelwise")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float32", "float16", "bfloat16"),
        help="加载后模型浮点类型，可降低显存占用",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]

    model_path = args.model_path or bc.default_model_path()
    quant_json = str(Path(args.quant_json).resolve())
    out_path = Path(args.out).resolve()

    ptq = bc.load_ptq_module()
    logger = logging.getLogger("save_vggt_w4_int")
    logging.basicConfig(level=logging.INFO)

    model = VGGT.from_pretrained(model_path).to(device=device, dtype=model_dtype).eval()

    quant_cfg = ptq.init_config(args.config_name)
    mod = ptq.cfg_modifier(
        linear_ptq_setting=(1, 1, 1),
        metric="hessian",
        bit_setting=(4, 8),
        linear_channelwise=True,
    )
    quant_cfg = mod(quant_cfg)

    wrapped_modules = ptq.wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True)
    ptq.model_load(model, quant_json, logger=logger)

    int_weights = int_weights_from_wrapped(wrapped_modules)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "int_weights": int_weights,
            "quant_json": quant_json,
            "config_name": args.config_name,
            "w_bit": 4,
            "a_bit": 8,
            "model_path": model_path,
            "model_dtype": args.dtype,
            "note": "int8 tensor 存 4bit 对称档位，取值约 [-8, 7]。",
        },
        out_path,
    )
    logger.info("Saved %d tensors -> %s", len(int_weights), out_path)


if __name__ == "__main__":
    main()
