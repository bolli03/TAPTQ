"""VGGT model adapters and calibration orchestration for quant baselines."""
import json
import math
import os
import time
import torch
import torch.nn as nn
from PTQ.quant_layers.baseline import BaselineQuantLinear, AffineQParams, choose_qparams, fake_quant
from PTQ.quant_layers.erq import erq_linear, rtn_weight
from PTQ.quant_layers.gptq import GPTQQuantLinear
from PTQ.quant_layers.official_quant import affine_fake_quant, repq_qparams
from PTQ.quant_layers.smoothquant import SmoothQuantLinear, smooth_scale

LEAF_NAMES = {"qkv", "proj", "fc1", "fc2"}

def _resolve(root, path):
    current = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current

def _parent(root, path):
    parent, leaf = path.rsplit(".", 1)
    return _resolve(root, parent), leaf

def wrap_vggt_linears(model, a_bit, expected_count=None):
    wrapped = {}
    for name, module in list(model.named_modules()):
        if not name.startswith("aggregator.") or name.rsplit(".", 1)[-1] not in LEAF_NAMES:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, leaf = _parent(model, name)
        quant = BaselineQuantLinear.from_float(module, a_bit)
        setattr(parent, leaf, quant)
        wrapped[name] = quant
    if not wrapped:
        raise RuntimeError("No VGGT aggregator linear modules were wrapped")
    if expected_count is not None and len(wrapped) != expected_count:
        raise RuntimeError(f"VGGT scope must contain {expected_count} linears, got {len(wrapped)}")
    return wrapped


def wrap_vggt_gptq_linears(model, w_bit, a_bit, expected_count=None):
    wrapped = {}
    for name, module in list(model.named_modules()):
        if not name.startswith("aggregator.") or name.rsplit(".", 1)[-1] not in LEAF_NAMES:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, leaf = _parent(model, name)
        quant = GPTQQuantLinear.from_float(module, w_bit, a_bit)
        setattr(parent, leaf, quant)
        wrapped[name] = quant
    if not wrapped:
        raise RuntimeError("No VGGT aggregator linear modules were wrapped for GPTQ")
    if expected_count is not None and len(wrapped) != expected_count:
        raise RuntimeError(f"VGGT GPTQ scope must contain {expected_count} linears, got {len(wrapped)}")
    return wrapped


def wrap_vggt_smoothquant_linears(model, a_bit, expected_count=None):
    wrapped = {}
    for name, module in list(model.named_modules()):
        if not name.startswith("aggregator.") or name.rsplit(".", 1)[-1] not in LEAF_NAMES:
            continue
        if not isinstance(module, nn.Linear):
            continue
        runtime_smoothing = name.endswith((".attn.proj", ".mlp.fc2"))
        parent, leaf = _parent(model, name)
        quant = SmoothQuantLinear.from_float(module, a_bit, runtime_smoothing)
        setattr(parent, leaf, quant)
        wrapped[name] = quant
    if not wrapped:
        raise RuntimeError("No VGGT aggregator linears were wrapped for SmoothQuant")
    if expected_count is not None and len(wrapped) != expected_count:
        raise RuntimeError(
            f"VGGT SmoothQuant scope must contain {expected_count} linears, got {len(wrapped)}"
        )
    return wrapped

@torch.inference_mode()
def collect_inputs(model, modules, dataset, seq_id_map, max_tokens, device, logger):
    """Collect an even, deterministic token sample for every wrapped layer."""
    per_sequence = max(1, math.ceil(max_tokens / len(seq_id_map)))
    cached = {}
    counts = {name: 0 for name in modules}
    hooks = []
    for name, module in modules.items():
        def hook(_module, inputs, _output, key=name):
            remaining = max_tokens - counts[key]
            if remaining <= 0:
                return
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            take = min(values.shape[0], per_sequence, remaining)
            if values.shape[0] > take:
                index = torch.linspace(0, values.shape[0] - 1, take, device=values.device).long()
                values = values[index]
            values = values[:take].to(device="cpu", dtype=torch.float16)
            if key not in cached:
                cached[key] = torch.empty(max_tokens, values.shape[1], dtype=torch.float16)
            cached[key][counts[key]:counts[key] + take].copy_(values)
            counts[key] += take
        hooks.append(module.register_forward_hook(hook))
    try:
        for seq_name, ids in seq_id_map.items():
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            images = data["images"].to(device)
            model.aggregator(images.unsqueeze(0))
            del data, images
    finally:
        for hook in hooks: hook.remove()
    result = {}
    for name in modules:
        if counts[name] == 0:
            raise RuntimeError(f"No calibration input captured for {name}")
        result[name] = cached[name][:counts[name]]
    logger.info("Captured up to %d tokens for %d VGGT linear layers", max_tokens, len(result))
    return result

@torch.no_grad()
def apply_repq_reparameterization(model, modules, inputs, a_bit, candidates, logger):
    """Fold per-channel activation scales into LayerNorm/Linear, then use one scale."""
    changed_norms = []
    for name, module in modules.items():
        if name.endswith(".attn.qkv"):
            block_path, norm_leaf = name[:-len(".attn.qkv")], "norm1"
        elif name.endswith(".mlp.fc1"):
            block_path, norm_leaf = name[:-len(".mlp.fc1")], "norm2"
        else:
            continue
        norm_name = f"{block_path}.{norm_leaf}"
        norm = _resolve(model, norm_name)
        if not isinstance(norm, nn.LayerNorm) or norm.weight is None or norm.bias is None:
            raise RuntimeError(f"RepQ requires affine LayerNorm before {name}")
        x = inputs[name].float().to(module.weight.device)
        params = repq_qparams(x, a_bit, axis=1)
        target_scale = params.scale.mean().clamp_min(1e-8)
        target_zero = params.zero.mean()
        ratio = (params.scale / target_scale).clamp_min(1e-8)
        channel_min = -params.zero * params.scale
        target_min = -target_zero * target_scale
        shift = channel_min / ratio - target_min
        norm.weight.div_(ratio.to(norm.weight.device, norm.weight.dtype))
        norm.bias.div_(ratio.to(norm.bias.device, norm.bias.dtype)).sub_(shift.to(norm.bias.device, norm.bias.dtype))
        module.weight.mul_(ratio.to(module.weight.device, module.weight.dtype).unsqueeze(0))
        if module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features, device=module.weight.device, dtype=module.weight.dtype))
        module.bias.add_(module.weight @ shift.to(module.weight.device, module.weight.dtype))
        module.set_input_qparams(AffineQParams(target_scale, target_zero, a_bit))
        inputs[name] = (x / ratio - shift).to(device="cpu", dtype=torch.float16)
        changed_norms.append(norm_name)
    logger.info("Applied RepQ scale reparameterization to %d LayerNorm/Linear pairs", len(changed_norms))
    return changed_norms

@torch.no_grad()
def calibrate_smoothquant(model, modules, inputs, w_bit, a_bit, alpha, eps, logger):
    stats, norm_names, runtime_names, started = {}, [], [], time.time()
    for index, (name, module) in enumerate(modules.items(), 1):
        x = inputs.pop(name).float().to(module.weight.device)
        activation_absmax = x.abs().amax(dim=0)
        scale = smooth_scale(activation_absmax, module.weight, alpha, eps)
        module.set_smooth_scale(scale)
        if module.runtime_smoothing:
            runtime_names.append(name)
            smoothed_input = x / scale
            module.weight.mul_(scale.to(module.weight.device, module.weight.dtype).unsqueeze(0))
        else:
            if name.endswith(".attn.qkv"):
                block_path, norm_leaf = name[:-len(".attn.qkv")], "norm1"
            elif name.endswith(".mlp.fc1"):
                block_path, norm_leaf = name[:-len(".mlp.fc1")], "norm2"
            else:
                raise RuntimeError(f"SmoothQuant cannot fold scale for {name}")
            norm_name = f"{block_path}.{norm_leaf}"
            norm = _resolve(model, norm_name)
            if not isinstance(norm, nn.LayerNorm) or norm.weight is None or norm.bias is None:
                raise RuntimeError(f"SmoothQuant requires affine LayerNorm before {name}")
            norm.weight.div_(scale.to(norm.weight.device, norm.weight.dtype))
            norm.bias.div_(scale.to(norm.bias.device, norm.bias.dtype))
            module.weight.mul_(scale.to(module.weight.device, module.weight.dtype).unsqueeze(0))
            smoothed_input = x / scale
            norm_names.append(norm_name)
        activation = repq_qparams(smoothed_input, a_bit)
        module.set_input_qparams(AffineQParams(activation.scale, activation.zero, a_bit))
        weight_params = repq_qparams(module.weight, w_bit, axis=0)
        module.weight.copy_(affine_fake_quant(module.weight.float(), weight_params).to(module.weight.dtype))
        module.set_quant_state(True)
        stats[name] = {
            "tokens": int(x.shape[0]),
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
            "runtime_smoothing": bool(module.runtime_smoothing),
        }
        logger.info("[%d/%d] SmoothQuant calibrated %s", index, len(modules), name)
    return {
        "method": "smoothquant", "seconds": time.time() - started,
        "layers": stats, "module_names": list(modules),
        "reparameterized_norms": norm_names,
        "runtime_smoothed_modules": runtime_names,
        "alpha": float(alpha), "smooth_eps": float(eps),
    }


@torch.no_grad()
def calibrate_baseline(model, modules, inputs, method, w_bit, a_bit, ridge, candidates, iterations, row_batch, group_size, erq_two_part, logger):
    if method not in {"rtn", "repq", "erq"}: raise ValueError(f"Unsupported baseline: {method}")
    norm_names = []
    if method in {"repq", "erq"}:
        norm_names = apply_repq_reparameterization(model, modules, inputs, a_bit, candidates, logger)
    stats, started = {}, time.time()
    for index, (name, module) in enumerate(modules.items(), 1):
        x = inputs.pop(name).float().to(module.weight.device)
        if not bool(module.input_calibrated):
            activation = repq_qparams(x, a_bit)
            module.set_input_qparams(AffineQParams(activation.scale, activation.zero, a_bit))
        if method == "erq":
            use_two_part = erq_two_part and name.endswith((".attn.qkv", ".mlp.fc1"))
            stats[name] = erq_linear(
                module, x, w_bit, ridge, candidates, iterations, row_batch, group_size, use_two_part
            )
        elif method == "repq":
            weight_params = repq_qparams(module.weight, w_bit, axis=0)
            module.weight.copy_(affine_fake_quant(module.weight.float(), weight_params).to(module.weight.dtype))
            stats[name] = {"tokens": x.shape[0], "official_quantizer": "RepQ-ViT percentile grid"}
        else:
            module.weight.copy_(rtn_weight(module.weight, w_bit, candidates).to(module.weight.dtype))
            stats[name] = {"tokens": x.shape[0]}
        module.set_quant_state(True)
        logger.info("[%d/%d] calibrated %s", index, len(modules), name)
        del x
    return {"method": method, "seconds": time.time() - started, "layers": stats, "module_names": list(modules), "reparameterized_norms": norm_names}

def _required_state(model, modules, norm_names):
    prefixes = tuple(f"{name}." for name in modules) + tuple(f"{name}." for name in norm_names)
    return {key for key in model.state_dict() if key.startswith(prefixes)}

def save_checkpoint(model, modules, metadata, path):
    required = _required_state(model, modules, metadata["reparameterized_norms"])
    state = {}
    for key, value in model.state_dict().items():
        if key not in required:
            continue
        value = value.detach().cpu()
        state[key] = value.half() if value.is_floating_point() and key.endswith((".weight", ".bias")) else value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"version": 1, "metadata": metadata, "state_dict": state}, path)

def load_checkpoint(model, modules, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported baseline checkpoint: {path}")
    metadata = payload["metadata"]
    if metadata.get("module_names") != list(modules):
        raise RuntimeError("Checkpoint VGGT module scope does not match the current model")
    required = _required_state(model, modules, metadata["reparameterized_norms"])
    actual = set(payload["state_dict"])
    if actual != required:
        raise RuntimeError(f"Checkpoint key mismatch: missing={sorted(required-actual)[:8]}, extra={sorted(actual-required)[:8]}")
    _missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    del payload
    for module in modules.values():
        module.set_quant_state(True)
    return metadata

def write_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
