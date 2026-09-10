"""Run the SVDQuant-style VGGT adapter with the unified evaluator."""
from __future__ import annotations

import json
import logging
import os
import os.path as osp

import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from PTQ.quant_layers.svdquant import SVDQuantLinear
from PTQ.vggt.models.vggt import VGGT
from PTQ.utils.baseline_calib import collect_inputs
from mv_recon.taptq import evaluate
from utils.messages import set_default_arg

LEAF_NAMES = {"qkv", "proj", "fc1", "fc2"}


def resolve(root_module, path):
    current = root_module
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def parent(root_module, path):
    prefix, leaf = path.rsplit(".", 1)
    return resolve(root_module, prefix), leaf


def replace_linears(model, w_bit, a_bit, rank, alpha, group_size, num_grids):
    wrapped = {}
    for name, module in list(model.named_modules()):
        if not name.startswith("aggregator.") or name.rsplit(".", 1)[-1] not in LEAF_NAMES:
            continue
        if not isinstance(module, nn.Linear):
            continue
        owner, leaf = parent(model, name)
        quant = SVDQuantLinear.from_float(module, w_bit, a_bit, rank, alpha, group_size, num_grids)
        setattr(owner, leaf, quant)
        wrapped[name] = quant
    if len(wrapped) != 288:
        raise RuntimeError(f"Expected 288 VGGT Linear modules, found {len(wrapped)}")
    return wrapped


def calibration_data(cfg):
    names = cfg.optim_datasets or cfg.test_datasets
    name = str(names[0])
    info = cfg.data[name]
    dataset = hydra.utils.instantiate(info.cfg)
    with open(info.seq_id_map, "r", encoding="utf-8") as handle:
        seq_id_map = json.load(handle)
    return name, dataset, seq_id_map


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    logger = logging.getLogger("svdquant-vggt")
    w_bit = int(OmegaConf.select(cfg, "svdquant.w_bit") or 4)
    a_bit = int(OmegaConf.select(cfg, "svdquant.a_bit") or 4)
    rank = int(OmegaConf.select(cfg, "svdquant.rank") or 32)
    alpha = float(OmegaConf.select(cfg, "svdquant.alpha") or 0.5)
    max_tokens = int(OmegaConf.select(cfg, "svdquant.max_tokens") or 4096)
    candidates = int(OmegaConf.select(cfg, "svdquant.scale_candidates") or 20)
    group_size = int(OmegaConf.select(cfg, "svdquant.group_size") or 64)
    num_grids = int(OmegaConf.select(cfg, "svdquant.num_grids") or 20)
    model_path = os.environ.get("VGGT_MODEL_PATH", osp.join(root, "..", "models", "hf_hub", "models--facebook--VGGT-1B"))
    model = VGGT.from_pretrained(model_path).to(cfg.device).eval()
    modules = replace_linears(model, w_bit, a_bit, rank, alpha, group_size, num_grids)
    dataset_name, dataset, seq_id_map = calibration_data(cfg)
    inputs = collect_inputs(model, modules, dataset, seq_id_map, max_tokens, cfg.device, logger)
    for index, (name, module) in enumerate(modules.items(), 1):
        module.calibrate(inputs[name], candidates)
        logger.info("[%d/%d] calibrated %s", index, len(modules), name)
    logger.info("DeepCompressor-aligned SVDQuant VGGT calibration complete: W%dA%d rank=%d group=%d grids=%d", w_bit, a_bit, rank, group_size, num_grids)
    evaluate(cfg, model, logger)


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
