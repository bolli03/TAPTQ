#!/usr/bin/env python3
"""Evaluate the deployed VGGT W8A8 backend with the shared E1 protocol."""
from __future__ import annotations

import logging
import os

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from deployment.w8a8_triton_deploy import load_model, resolve_dtype  # noqa: E402
from mv_recon.taptq import evaluate  # noqa: E402


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("w8a8_triton_eval")
    model_path = os.environ["VGGT_MODEL_PATH"]
    artifact = os.environ["VGGT_W8A8_ARTIFACT"]
    post_gelu_mode = os.environ.get("VGGT_W8A8_POST_GELU_MODE", "dual")
    non_quant_dtype = resolve_dtype(os.environ.get("VGGT_NON_QUANT_DTYPE", "bfloat16"))
    model, metadata = load_model(
        model_path,
        artifact,
        torch.device(cfg.device),
        non_quant_dtype,
        post_gelu_mode,
    )
    if non_quant_dtype != torch.float32:
        model.register_forward_pre_hook(
            lambda _module, args: (args[0].to(non_quant_dtype), *args[1:])
        )
    logger.info("Loaded deployed W8A8 model: %s", metadata)
    evaluate(cfg, model, logger)


if __name__ == "__main__":
    main()
