"""Unified VGGT entrypoint for RTN, RepQ and ERQ baselines.

Examples:
  python mv_recon/baseline_quant.py ++baseline.method=erq ++mode=calib
  python mv_recon/baseline_quant.py ++baseline.method=repq ++mode=e2e
  python mv_recon/baseline_quant.py ++baseline.method=gptq ++mode=e2e ++baseline.bit=[4,8]
"""
import json
import logging
import os
import os.path as osp
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from PTQ.utils.baseline_calib import calibrate_baseline, calibrate_smoothquant, collect_inputs, load_checkpoint, save_checkpoint, wrap_vggt_gptq_linears, wrap_vggt_linears, wrap_vggt_smoothquant_linears, write_metadata
from PTQ.vggt.models.vggt import VGGT
from mv_recon.taptq import evaluate
from utils.messages import set_default_arg

def _value(cfg, key, default):
    value = OmegaConf.select(cfg, key)
    return default if value is None else value

def _calibration_data(cfg):
    names = cfg.optim_datasets or cfg.test_datasets
    if not names: raise ValueError("optim_datasets/test_datasets is empty")
    info = cfg.data[str(names[0])]
    dataset = hydra.utils.instantiate(info.cfg)
    with open(info.seq_id_map, "r", encoding="utf-8") as handle:
        seq_id_map = json.load(handle)
    return str(names[0]), dataset, seq_id_map

def _build(cfg, settings):
    device = torch.device(str(cfg.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT unified evaluation currently requires an available CUDA device")
    if str(_value(cfg, "model_name", "vggt")).lower() != "vggt":
        raise ValueError("This baseline adapter currently supports model_name=vggt only")
    model_path = os.environ.get("VGGT_MODEL_PATH", osp.join(root, "..", "models", "hf_hub", "models--facebook--VGGT-1B"))
    model = VGGT.from_pretrained(model_path).to(cfg.device).eval()
    if settings["method"] == "gptq":
        modules = wrap_vggt_gptq_linears(model, settings["w_bit"], settings["a_bit"], expected_count=288)
    elif settings["method"] == "smoothquant":
        modules = wrap_vggt_smoothquant_linears(model, settings["a_bit"], expected_count=288)
    else:
        modules = wrap_vggt_linears(model, settings["a_bit"], expected_count=288)
    return model, modules, model_path

def _settings(cfg):
    method = str(_value(cfg, "baseline.method", "rtn")).lower()
    bits = list(_value(cfg, "baseline.bit", [4, 8]))
    settings = {
        "method": method, "w_bit": int(bits[0]) if len(bits) == 2 else 0,
        "a_bit": int(bits[1]) if len(bits) == 2 else 0,
        "ridge": float(_value(cfg, "baseline.ridge", 1000.0)),
        "max_tokens": int(_value(cfg, "baseline.max_tokens", 512)),
        "scale_candidates": int(_value(cfg, "baseline.scale_candidates", 20)),
        "rounding_iterations": int(_value(cfg, "baseline.rounding_iterations", 100)),
        "row_batch_size": int(_value(cfg, "baseline.row_batch_size", 128)),
        "linear_group_size": int(_value(cfg, "baseline.linear_group_size", 256)),
        "erq_two_part": bool(_value(cfg, "baseline.erq_two_part", False)),
        "smoothquant_alpha": float(_value(cfg, "baseline.smoothquant_alpha", 0.5)),
        "smoothquant_eps": float(_value(cfg, "baseline.smoothquant_eps", 1e-5)),
        "damp_percent": float(_value(cfg, "baseline.damp_percent", 0.01)),
        "gptq_block_size": int(_value(cfg, "baseline.gptq_block_size", 128)),
        "gptq_group_size": int(_value(cfg, "baseline.gptq_group_size", -1)),
        "gptq_act_order": bool(_value(cfg, "baseline.gptq_act_order", False)),
    }
    if method not in {"rtn", "repq", "erq", "gptq", "smoothquant"}:
        raise ValueError(
            f"baseline.method must be rtn, repq, erq, gptq, or smoothquant; got {method}"
        )
    if len(bits) != 2 or not all(2 <= bit <= 16 for bit in (settings["w_bit"], settings["a_bit"])):
        raise ValueError(f"baseline.bit must contain two values in [2,16], got {bits}")
    positive = ("ridge", "max_tokens", "scale_candidates", "row_batch_size")
    if (any(settings[key] <= 0 for key in positive) or settings["rounding_iterations"] < 0
            or settings["linear_group_size"] < 0 or settings["damp_percent"] < 0
            or not 0 <= settings["smoothquant_alpha"] <= 1
            or settings["smoothquant_eps"] <= 0):
        raise ValueError(f"Invalid baseline settings: {settings}")
    return settings

def _checkpoint_path(cfg, settings):
    path = OmegaConf.select(cfg, "ckpt.path")
    if path: return str(path)
    filename = "vggt_{}_w{}a{}.pt".format(settings["method"], settings["w_bit"], settings["a_bit"])
    return osp.join(root, "param", "baselines", filename)

def run_calibration(cfg, evaluate_after=False):
    logger, settings = logging.getLogger("vggt-baseline"), _settings(cfg)
    torch.manual_seed(int(cfg.seed))
    model, modules, model_path = _build(cfg, settings)
    dataset_name, dataset, seq_id_map = _calibration_data(cfg)
    logger.info("Calibrating VGGT %s on %s with %d wrapped linears", settings["method"].upper(), dataset_name, len(modules))
    inputs = collect_inputs(model, modules, dataset, seq_id_map, settings["max_tokens"], cfg.device, logger)
    if settings["method"] == "gptq":
        import time
        started = time.time()
        stats = {}
        for index, (name, module) in enumerate(modules.items(), 1):
            calibration_inputs = inputs.pop(name)
            module.calibrate(
                calibration_inputs,
                settings["scale_candidates"],
                settings["damp_percent"],
                settings["gptq_block_size"],
                settings["gptq_group_size"],
                settings["gptq_act_order"],
            )
            stats[name] = {"tokens": int(calibration_inputs.reshape(-1, calibration_inputs.shape[-1]).shape[0])}
            logger.info("[%d/%d] GPTQ calibrated %s", index, len(modules), name)
            del calibration_inputs
        result = {
            "method": "gptq",
            "seconds": time.time() - started,
            "layers": stats,
            "module_names": list(modules),
            "reparameterized_norms": [],
            "damp_percent": settings["damp_percent"],
        }
    elif settings["method"] == "smoothquant":
        result = calibrate_smoothquant(
            model, modules, inputs, settings["w_bit"], settings["a_bit"],
            settings["smoothquant_alpha"], settings["smoothquant_eps"], logger,
        )
    else:
        result = calibrate_baseline(
            model, modules, inputs, settings["method"], settings["w_bit"], settings["a_bit"],
            settings["ridge"], settings["scale_candidates"], settings["rounding_iterations"],
            settings["row_batch_size"], settings["linear_group_size"], settings["erq_two_part"], logger,
        )
    metadata = {
        "adapter": "VGGT", "scope": "aggregator qkv/proj/fc1/fc2",
        "calibration_dataset": dataset_name, "model_path": model_path,
        "settings": settings, **result,
    }
    path = _checkpoint_path(cfg, settings)
    save_checkpoint(model, modules, metadata, path)
    write_metadata(metadata, path + ".json")
    logger.info("Saved %s checkpoint to %s", settings["method"].upper(), path)
    if evaluate_after: evaluate(cfg, model, logger)

def run_test(cfg):
    logger, settings = logging.getLogger("vggt-baseline"), _settings(cfg)
    path = _checkpoint_path(cfg, settings)
    model, modules, _ = _build(cfg, settings)
    metadata = load_checkpoint(model, modules, path)
    saved = metadata["settings"]
    if any(saved[key] != settings[key] for key in ("method", "w_bit", "a_bit")):
        raise ValueError(f"CLI settings do not match checkpoint metadata: {saved}")
    logger.info("Loaded VGGT %s checkpoint from %s", saved["method"].upper(), path)
    evaluate(cfg, model, logger)

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    mode = str(_value(cfg, "mode", "e2e"))
    if mode == "calib": run_calibration(cfg)
    elif mode == "test": run_test(cfg)
    elif mode == "e2e": run_calibration(cfg, evaluate_after=True)
    else: raise ValueError("baseline_quant mode must be calib, test, or e2e")

if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
