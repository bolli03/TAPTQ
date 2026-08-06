#!/usr/bin/env python3
"""
Skeleton demo: PyTorch fake-quant path vs (future) real INT deployment — mostly TODOs.

This file is intentionally non-runnable as a full pipeline; copy sections into your own
scripts when you wire real paths, TensorRT, etc.

Repo context:
  - VGGT + wrap_modules_in_net(quantize_aggregator=True) + model_load(json) + enable_quant
  - Saved int weights: see save_vggt_w4_int_weights.py output (.pth)
  - TensorRT / ONNX: mv_recon/tensorrt/README.md, export_vggt_onnx.py
"""
from __future__ import annotations

# region imports (TODO: uncomment when you run for real)
# import os
# import time
# import statistics
# from pathlib import Path
#
# import torch
#
# import rootutils
# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
#
# from PTQ.vggt.models.vggt import VGGT
# import mv_recon.benchmark_common as bc
# import mv_recon.benchmark_common as benchmark_common  # run_deploy_benchmark, build_dummy_batch
# endregion


def todo_load_fp_vggt():
    """
    TODO: set VGGT_MODEL_PATH or pass --model-path to local HF snapshot.
    TODO: choose device cuda:0 and dtype (float32 vs float16 for memory).
    TODO: model.eval() before timing.
    """
    # model = VGGT.from_pretrained(model_path).to(device, dtype=dtype).eval()
    # return model
    raise NotImplementedError("TODO: implement load")


def todo_wrap_and_load_quant_json(model, quant_cfg_name: str, quant_json: str):
    """
    TODO: os.chdir(repo_root) so init_config finds ./configs.
    TODO: ptq = bc.load_ptq_module()
    TODO: quant_cfg = ptq.init_config(quant_cfg_name)  # e.g. PTQ4ViT_channelwise
    TODO: quant_cfg = ptq.cfg_modifier(
        linear_ptq_setting=(1, 1, 1),
        metric="hessian",
        bit_setting=(4, 8),
        linear_channelwise=True,
    )(quant_cfg)
    TODO: wrapped = ptq.wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True)
    TODO: ptq.model_load(model, quant_json, logger=...)
    TODO: return wrapped  # dict name -> quant module (for int weight export etc.)
    """
    raise NotImplementedError("TODO")


def todo_enable_quant_forward(model, images: "torch.Tensor"):
    """
    TODO: ptq.enable_quant(model)  # or project-specific helper that sets mode='quant_forward'
    TODO: with torch.inference_mode():
    TODO:     out = model(images)              # full VGGT
    TODO:     # or: out_list, idx = model.aggregator(images)  # aggregator-only latency
    """
    raise NotImplementedError("TODO")


def todo_benchmark_latency_pytorch(
    model,
    images: "torch.Tensor",
    warmup: int,
    iters: int,
    forward_fn,  # callable: () -> None, e.g. lambda: model.aggregator(images)
):
    """
    TODO: model.eval(); torch.cuda.synchronize() before/after each timed region.
    TODO: warmup loop (no timing) to stabilize CUDA/cudnn autotune.
    TODO: timed loop: sync -> perf_counter -> forward_fn() -> sync -> perf_counter
    TODO: collect list of deltas, report mean_ms, stdev_ms, min_ms (see benchmark_common.benchmark_forward).
    TODO: or use existing CLI: benchmark_quant_deploy.py --aggregator-only (and optional --quant-json ...).
    TODO: optional: torch.compile / cudnn.benchmark effects — document in your report.
    """
    # times = []
    # for _ in range(warmup):
    #     forward_fn()
    # torch.cuda.synchronize()
    # for _ in range(iters):
    #     torch.cuda.synchronize()
    #     t0 = time.perf_counter()
    #     forward_fn()
    #     torch.cuda.synchronize()
    #     times.append(time.perf_counter() - t0)
    # return {"mean_ms": statistics.mean(times) * 1e3, ...}
    raise NotImplementedError("TODO")


def todo_peak_memory_cuda(tag: str):
    """
    TODO: torch.cuda.reset_peak_memory_stats(device)
    # run forward once or iters
    TODO: peak = torch.cuda.max_memory_allocated(device)
    TODO: log peak bytes / MiB with tag (e.g. 'after_load', 'after_forward').
    """
    raise NotImplementedError("TODO")


def todo_load_saved_int_weights_pth(path: str):
    """
    TODO: payload = torch.load(path, map_location='cpu', weights_only=False)
    TODO: int_weights = payload['int_weights']  # dict[str, Tensor int8]
    TODO: quant_json = payload.get('quant_json')  # intervals still needed for dequant / TRT scales
    TODO: NOTE: these tensors alone do NOT plug into VGGT.forward(); you need a deployment path
          that consumes packed int4 + scales (TensorRT, custom kernel, etc.).
    """
    raise NotImplementedError("TODO")


def todo_export_onnx_aggregator(out_onnx: str):
    """
    TODO: from repo root: python mv_recon/export_vggt_onnx.py --target aggregator --out ...
    TODO: fix num-views / H / W to match production trace shape.
    TODO: ONNX from this script is FP; embedding PTQ JSON requires separate Q/DQ pipeline
          (see mv_recon/tensorrt/README.md).
    """
    raise NotImplementedError("TODO")


def todo_tensorrt_build_engine(onnx_path: str, engine_path: str):
    """
    TODO: install TensorRT wheel matching CUDA/driver.
    TODO: trtexec --onnx=... --saveEngine=... --fp16  (validate first)
    TODO: INT8/INT4: add calibration or Q/DQ ONNX; map block-wise w_interval to per-tensor/channel scales
          or custom plugin — large TODO, see README section 3.
    """
    raise NotImplementedError("TODO")


def todo_tensorrt_benchmark(engine_path: str):
    """
    TODO: trtexec --loadEngine=... --iterations=N --avgRuns=M  (read TRT docs for flags)
    TODO: or Python: tensorrt.Runtime + execute_async_v3 + CUDA events for latency
    TODO: memory: engine size + activation scratch (workspace); use nvidia-smi / Nsight for ground truth
    """
    raise NotImplementedError("TODO")


def todo_end_to_end_checklist():
    """
    Checklist (all TODO for you to tick):

    [ ] Input layout: B x S x 3 x H x W float in [0,1] (match VGGT forward contract).
    [ ] Same shape/dtype/amp policy for FP baseline vs quant vs TRT.
    [ ] Warmup iterations >= 5 (GPU); sync before timer.
    [ ] Report: mean/stdev/min latency; GPU name; batch/S/H/W; PyTorch vs TRT version.
    [ ] For "real INT": benchmark the *engine* or *ORT session*, not PyTorch fake-quant only.
    [ ] If comparing accuracy: same random seed or same saved tensor inputs.
    """
    pass


def main():
    """Prints skeleton pointers only; does not run a model."""
    print(__doc__)
    print("\nFunctions to fill in (search for TODO in this file):")
    for name in globals():
        if name.startswith("todo_"):
            print(f"  - {name}()")
    todo_end_to_end_checklist()


if __name__ == "__main__":
    main()
