"""TAPTQ post-training quantization + QwT compensation for VGGT / Pi3.

Three modes, switched via Hydra `mode=...`:
    mode=calib   calibrate (+optional compensate) -> save checkpoint, no eval
    mode=test    load checkpoint -> eval only
    mode=e2e     calibrate + compensate + eval + save  (original pipeline)

Examples:
    # VGGT default behaviour
    python taptq.py
    # Pi3, using the same encoder/decoder/point_decoder linear scope as legacy ptq4pi3.py
    python taptq.py model_name=pi3 ptq.bit=[4,8] mode=calib \\
        ckpt.path=outputs/pi3_w4a8_ternary.pt ptq.search_mode=ternary
    # Pi3 W4A8 channel-wise calibration
    python taptq.py model_name=pi3 ptq.bit=[4,8] mode=calib \\
        ckpt.path=outputs/pi3_w4a8_channelwise.pt \\
        ptq.search_mode=ternary ptq.linear_channelwise=true
    # load and test
    python taptq.py ++mode=test ++ckpt.path=outputs/vggt_w4a8.pt
"""

import json
import logging
import os
import os.path as osp
import random
import re
import time
import ast
from importlib import import_module, reload
from itertools import product

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from PTQ.quant_layers.conv import MinMaxQuantConv2d
from PTQ.quant_layers.linear import MinMaxQuantLinear
from PTQ.quant_layers.matmul import MinMaxQuantMatMul
from PTQ.utils.integer import get_model_int_weight  # noqa: F401 (used in commented int8 save)
from PTQ.utils.net_wrap import wrap_certain_modules_in_net, wrap_modules_in_net  # noqa: F401
from PTQ.utils.quant_calib import HessianQuantCalibrator, QuantCalibrator  # noqa: F401
from PTQ.vggt.models.vggt import VGGT
from mv_recon.utils import accuracy, completion, umeyama
from utils.interfaces import infer_mv_pointclouds
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto


# ============================================================================
# Quant mode toggle
# ============================================================================

_QUANT_TYPES = (MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul)


def enable_quant(submodel):
    for _, m in submodel.named_modules():
        if isinstance(m, _QUANT_TYPES):
            m.mode = "quant_forward"


def disable_quant(submodel):
    for _, m in submodel.named_modules():
        if isinstance(m, _QUANT_TYPES):
            m.mode = "raw"


# ============================================================================
# Regression / SVD primitives for QwT compensation
# ============================================================================


def linear_regression(X, Y):
    """Closed-form OLS for Δ = X W + b. Returns (W, b, R2)."""
    X = X.reshape(-1, X.size(-1))
    Y = Y.reshape(-1, Y.size(-1))

    ones = torch.ones(X.size(0), 1, device=X.device)
    X1 = torch.cat([X, ones], dim=-1)

    W_all = torch.inverse(X1.t() @ X1) @ X1.t() @ Y
    W, b = W_all[:-1, :], W_all[-1, :]

    Y_pred = X @ W + b
    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2 = 1 - ss_res / ss_tot
    return W, b, r2


def svd_low_rank(W: torch.Tensor, rank: int):
    """Factor W ≈ A @ B with rank `rank`. Returns (A: [din, r], B: [r, dout])."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    s_half = torch.sqrt(S_r)
    return U_r * s_half.unsqueeze(0), s_half.unsqueeze(1) * Vh_r


# ============================================================================
# Output-quality diagnostics
# ============================================================================


# Runtime overrides (set by _apply_compensation from Hydra config; defaults applied here).
_RUNTIME_TAIL_RATIO = 0.01     # used by tail_relative_error()
# _QwTBase.rank (class attribute, default 32) is the third lever — overridden in-place.


@torch.no_grad()
def tail_relative_error(fp_out, quant_out, tail_ratio=None, eps=1e-8):
    """TRE: relative squared error on top-magnitude elements of FP output.

    tail_ratio=None -> use the runtime default `_RUNTIME_TAIL_RATIO`
    (set by `_apply_compensation` from `++compensate.tail_ratio`, defaults 0.01).
    """
    if tail_ratio is None:
        tail_ratio = _RUNTIME_TAIL_RATIO
    assert fp_out.shape == quant_out.shape
    y, y_q = fp_out.reshape(-1), quant_out.reshape(-1)
    k = max(1, int(tail_ratio * y.numel()))
    _, idx = torch.topk(y.abs(), k, largest=True, sorted=False)
    num = (y[idx] - y_q[idx]).pow(2).sum()
    den = y[idx].pow(2).sum() + eps
    return num / den


def compute_cosine_similarity_flat(fp_out, quant_out, dim=1, eps=1e-8):
    """Mean per-sample cosine similarity (double precision). Alternative metric to TRE."""
    if fp_out.shape != quant_out.shape:
        raise ValueError(f"Shape mismatch: {fp_out.shape} vs {quant_out.shape}")
    fp_out = fp_out.double() if fp_out.dtype != torch.float64 else fp_out
    quant_out = quant_out.double() if quant_out.dtype != torch.float64 else quant_out
    a = fp_out.view(fp_out.size(0), -1)
    b = quant_out.view(quant_out.size(0), -1)
    return F.cosine_similarity(a, b, dim=dim, eps=eps).mean().item()


def generate_binary_list(p, l):
    """Return (list, k) with exactly `k = round(l*p)` ones placed randomly."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if l <= 0 or not isinstance(l, int):
        raise ValueError("l must be a positive int")
    k = max(0, min(l, round(l * p)))
    out = [1] * k + [0] * (l - k)
    random.shuffle(out)
    return out, k


# ============================================================================
# Compensation modules
# ============================================================================


class CompensationBlock(nn.Module):
    """Wrap a block with an additive LoRA residual: out + x @ W + b (Δ-compensation).

    Used by `recon_vit` and `recon_blocks` (whole-block strategy).
    """

    def __init__(self, W, b, r2_score, block, linear_init=True):
        super().__init__()
        self.block = block
        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)

    def forward(self, x, pos=None):
        out = self.block(x, pos=pos) if pos is not None else self.block(x)
        if self.training:
            return out + x @ self.lora_weight.float() + self.lora_bias
        return out + (x @ self.lora_weight).float() + self.lora_bias


class _QwTBase(nn.Module):
    """Shared logic for AttnQwT / MlpQwT: SVD low-rank QwT residual on top of an inner module.

    Subclasses set `inner_modules` (list of attrs of `block` to chain) and an `_apply_inner` hook.
    """

    rank = 32

    def __init__(self, W, b, r2_score, block, linear_init=True):
        super().__init__()
        self.QwT_enabled = False
        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            A, B = svd_low_rank(W, self.rank)
            self.register_buffer("A", A)
            self.register_buffer("B", B)
            self.QwT_enabled = True
            del self.lora_weight
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)

    def _apply_inner(self, x, pos=None):
        raise NotImplementedError

    def forward(self, x, pos=None):
        out = self._apply_inner(x, pos=pos)
        if not self.QwT_enabled:
            return out
        A = self.A.half()
        B = self.B.half()
        return out + (x.half() @ A @ B).float() + self.lora_bias


class AttnQwT(_QwTBase):
    """QwT residual around attn submodule: ls1 ∘ attn ∘ norm1."""

    def __init__(self, W, b, r2_score, block, linear_init=True):
        super().__init__(W, b, r2_score, block, linear_init)
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1

    def _apply_inner(self, x, pos=None):
        return self.ls1(self.attn(self.norm1(x), pos=pos))


class MlpQwT(_QwTBase):
    """QwT residual around mlp submodule: ls2 ∘ mlp ∘ norm2."""

    def __init__(self, W, b, r2_score, block, linear_init=True):
        super().__init__(W, b, r2_score, block, linear_init)
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2

    def _apply_inner(self, x, pos=None):
        return self.ls2(self.mlp(self.norm2(x)))


# ============================================================================
# Layer-wise compensation (replaces individual nn.Linear/Conv/Matmul)
# ============================================================================


def convert_module_path(path):
    """'aggregator.patch_embed.blocks.0.attn.proj' -> 'aggregator.patch_embed.blocks[0].attn.proj'."""
    path = re.sub(r"\.(\d+)\.", r"[\1].", path)
    path = re.sub(r"\.(\d+)$", r"[\1]", path)
    return path


def safe_setattr(model, path, new_module):
    """Set nested attribute described by dotted path with [idx] segments."""
    try:
        parts = path.split(".")
        current = model
        for part in parts[:-1]:
            if "[" in part and "]" in part:
                name = part.split("[")[0]
                index = int(part.split("[")[1].split("]")[0])
                current = getattr(current, name)[index]
            else:
                current = getattr(current, part)
        setattr(current, parts[-1], new_module)
        return True
    except Exception as e:
        logging.error(f"safe_setattr failed for {path}: {e}")
        return False


def _linear_forward_hook(module, input, output):
    """Cache module input on the module itself (for layer-wise QwT)."""
    if module.raw_input is None:
        module.raw_input = []
    module.raw_input.append(input[0].cpu().detach())


def compensate_layerwise(q_model, wrapped_modules, calib_loader, seq_id_map):
    """Wrap each calibrated quant module with CompensationBlock fitted on (FP - Q) residual."""
    hooks = []
    for _, module in wrapped_modules.items():
        module.raw_input = None
        hooks.append(module.register_forward_hook(_linear_forward_hook))

    # 1. collect inputs for every wrapped module
    for seq_name, ids in seq_id_map.items():
        data = calib_loader.get_data(sequence_name=seq_name, ids=ids)
        with torch.no_grad():
            q_model(data["images"].cuda())

    for h in hooks:
        h.remove()

    # 2. fit residual per module
    # NOTE alternative ablation: uncomment to skip qkv/fc1/fc2/proj layers
    # SKIP_SUBSTRS = ("qkv", "fc1", "fc2", "proj")
    for name, module in tqdm(list(wrapped_modules.items()), desc="Layer-wise QwT"):
        # if any(s in name for s in SKIP_SUBSTRS): continue
        module.raw_input = torch.cat(module.raw_input, dim=0)

        if not hasattr(module, "raw_out"):
            disable_quant(module)
            with torch.no_grad():
                fp_out = module(module.raw_input.cuda()).detach().cpu()
        else:
            fp_out = module.raw_out

        if not hasattr(module, "quant_out"):
            enable_quant(module)
            with torch.no_grad():
                quant_out = module(module.raw_input.cuda()).detach().cpu()
        else:
            quant_out = module.quant_out

        target = fp_out - quant_out
        W, b, r2 = linear_regression(module.raw_input.cuda(), target.cuda())
        del module.raw_input, target, fp_out, quant_out

        log_fn = logging.info if r2 > 0 else logging.warning
        log_fn(f"R2 score for layer {name}: {r2.item():.6f}")

        comp = CompensationBlock(W=W, b=b, r2_score=r2, block=module, linear_init=True)
        wrapped_modules[name] = comp
        if not safe_setattr(q_model, convert_module_path(name), comp):
            logging.error(f"Failed to replace {name}")
    return q_model.cuda()


# ============================================================================
# Block-wise compensation: one CompensationBlock per transformer block
# ============================================================================


def _calib_one_block(block, cur_inp, *, pos=None, log_prefix=""):
    """Return (W, b, r2, fp_out) for residual Δ = FP - Q fitted on cur_inp."""
    disable_quant(block)
    fp_out = (block(cur_inp, pos=pos) if pos is not None else block(cur_inp)).detach().cpu()
    enable_quant(block)
    quant_out = (block(cur_inp, pos=pos) if pos is not None else block(cur_inp)).detach().cpu()
    target = fp_out - quant_out
    W, b, r2 = linear_regression(cur_inp.cuda(), target.cuda())
    logging.info(f"R2 score for {log_prefix}: {r2.item():.6f}")
    return W, b, r2, fp_out


def recon_vit(cur_inp, net):
    """Block-wise CompensationBlock on a ViT patch-embed-style net (no pos)."""
    cur_inp = net.prepare_tokens_with_masks(cur_inp, None).cuda()
    with torch.no_grad():
        for i in range(len(net.blocks)):
            block = net.blocks[i]
            W, b, r2, fp_out = _calib_one_block(block, cur_inp, log_prefix=f"Vit block {i}")
            net.blocks[i] = CompensationBlock(W, b, r2, block=block, linear_init=True)
            cur_inp = fp_out.cuda()
    return net


def recon_blocks(cur_inp, B, S, P, C, pos, net):
    """Block-wise CompensationBlock alternating frame_blocks (per-frame) and global_blocks (across frames)."""
    cur_inp, pos = cur_inp.cuda(), pos.cuda()
    with torch.no_grad():
        for i in range(net.aa_block_num):
            # ---- per-frame block ----
            if cur_inp.shape != (B * S, P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)
            if pos is not None and pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)
            block = net.frame_blocks[i]
            W, b, r2, fp_out = _calib_one_block(block, cur_inp, pos=pos, log_prefix=f"frame block {i}")
            net.frame_blocks[i] = CompensationBlock(W, b, r2, block=block, linear_init=True)
            cur_inp = fp_out.cuda()

            # ---- cross-frame (global) block ----
            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)
            if pos is not None and pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            block = net.global_blocks[i]
            W, b, r2, fp_out = _calib_one_block(block, cur_inp, pos=pos, log_prefix=f"global block {i}")
            net.global_blocks[i] = CompensationBlock(W, b, r2, block=block, linear_init=True)
            cur_inp = fp_out.cuda()
    return net


# ============================================================================
# Module-wise compensation: separate AttnQwT / MlpQwT per block
# ============================================================================


def _attn_inner(block, x, pos=None):
    return block.ls1(block.attn(block.norm1(x), pos=pos))


def _mlp_inner(block, x):
    return block.ls2(block.mlp(block.norm2(x)))


# ----- Ratio-based selection: score every module then pick top-K -----


def _module_score(fp_out, quant_out, metric):
    """Per-module quant-error score. Higher = worse, more in need of QwT.

    - tre:     tail relative error (top-1% magnitude elements)
    - mse:     plain mean squared error
    - hessian: Fisher-diagonal approximation — weight squared output error by
               fp_out magnitude squared. Equals diag(H) of L = ||y - y_target||²
               evaluated at y=fp_out, so modules whose high-magnitude outputs
               see large quant noise rank highest. Cheap (no backward pass).
               Switch to a real-gradient Hessian if needed.
    - random:  no ranking; _pick_selected samples K uniformly.
    """
    if metric == "tre":
        return tail_relative_error(fp_out, quant_out).item()
    if metric == "mse":
        return F.mse_loss(quant_out.float(), fp_out.float()).item()
    if metric == "hessian":
        fp = fp_out.float()
        err = (quant_out.float() - fp).pow(2)
        return (fp.pow(2) * err).mean().item()
    if metric == "random":
        return float("nan")
    raise ValueError(f"Unknown select_metric: {metric}")


def _score_pass_vit(cur_inp, net, metric):
    """Walk a vit-style net with all-quant + no compensation, return per-module scores.

    Keys: ('patch', i, kind) for kind in {'attn','mlp'}.
    cur_inp propagates through quant inner outputs (mimics 'no-QwT' trajectory).
    """
    scores = {}
    cur_inp = net.prepare_tokens_with_masks(cur_inp, None).cuda()
    with torch.no_grad():
        for i, block in enumerate(net.blocks):
            for kind, inner_fn in (("attn", _attn_inner), ("mlp", _mlp_inner)):
                disable_quant(block)
                fp_out = inner_fn(block, cur_inp).detach()
                enable_quant(block)
                quant_out = inner_fn(block, cur_inp).detach()
                scores[("patch", i, kind)] = _module_score(fp_out, quant_out, metric)
                cur_inp = cur_inp + quant_out
    return scores


def _score_pass_aggregator(cur_inp, B, S, P, C, pos, net, metric):
    """Walk frame_blocks + global_blocks (all-quant, no compensation), return scores.

    Keys: ('frame'|'global', i, kind).
    """
    scores = {}
    cur_inp, pos = cur_inp.cuda(), pos.cuda()
    with torch.no_grad():
        for i in range(net.aa_block_num):
            # ---- per-frame block ----
            if cur_inp.shape != (B * S, P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)
            if pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)
            block = net.frame_blocks[i]
            for kind, inner_fn, use_pos in (("attn", _attn_inner, True), ("mlp", _mlp_inner, False)):
                disable_quant(block)
                fp_out = (inner_fn(block, cur_inp, pos=pos) if use_pos else inner_fn(block, cur_inp)).detach()
                enable_quant(block)
                quant_out = (inner_fn(block, cur_inp, pos=pos) if use_pos else inner_fn(block, cur_inp)).detach()
                scores[("frame", i, kind)] = _module_score(fp_out, quant_out, metric)
                cur_inp = cur_inp + quant_out

            # ---- cross-frame (global) block ----
            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)
            if pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            block = net.global_blocks[i]
            for kind, inner_fn, use_pos in (("attn", _attn_inner, True), ("mlp", _mlp_inner, False)):
                disable_quant(block)
                fp_out = (inner_fn(block, cur_inp, pos=pos) if use_pos else inner_fn(block, cur_inp)).detach()
                enable_quant(block)
                quant_out = (inner_fn(block, cur_inp, pos=pos) if use_pos else inner_fn(block, cur_inp)).detach()
                scores[("global", i, kind)] = _module_score(fp_out, quant_out, metric)
                cur_inp = cur_inp + quant_out
    return scores


def _pick_selected(scores, keep_ratio, metric, seed=42):
    """Pick K = round(N * keep_ratio) module keys. Returns a set.

    metric='random': sample K uniformly. metric='tre'/'mse': top-K by score desc.
    """
    n = len(scores)
    k = max(0, min(n, round(n * float(keep_ratio))))
    keys = list(scores.keys())
    if metric == "random":
        rng = random.Random(seed)
        selected = set(rng.sample(keys, k))
    else:
        ranked = sorted(keys, key=lambda key: scores[key], reverse=True)
        selected = set(ranked[:k])
    logging.info(f"[select] metric={metric} keep_ratio={keep_ratio} -> {k}/{n} modules compensated")
    return selected


def _calib_module(block, inner_fn, cur_inp, *, pos, tau_list, log_prefix, tau_thr, rand_skip,
                  force_skip=False):
    """Calibrate one inner module (attn or mlp). Returns (W, b, r2 with optional gate).

    force_skip=True forces r2=-3 (no-op QwT) — used by ratio-based selection
    to disable compensation on modules outside the selected set.
    """
    disable_quant(block)
    if pos is not None:
        fp_out = inner_fn(block, cur_inp, pos=pos).detach().cpu()
    else:
        fp_out = inner_fn(block, cur_inp).detach().cpu()
    enable_quant(block)
    if pos is not None:
        quant_out = inner_fn(block, cur_inp, pos=pos).detach().cpu()
    else:
        quant_out = inner_fn(block, cur_inp).detach().cpu()

    target = fp_out - quant_out
    tau = tail_relative_error(fp_out, quant_out)
    logging.info(f"tail relative error for {log_prefix}: {tau.item():.6f}")
    tau_list.append(tau.item())

    # NOTE alternative metric: cosine sim instead of TRE
    # sim = compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
    # logging.info(f"cosine similarity for {log_prefix}: {sim:.6f}")

    W, b, r2 = linear_regression(cur_inp.cuda(), target.cuda())

    if force_skip:
        r2 = -3.0
        logging.info(f"skip {log_prefix} QwT (not in selected set)")
    elif tau < tau_thr and rand_skip:
        # Legacy gate: small quant error AND random gate fired -> drop QwT.
        r2 = -3.0
        logging.info(f"close {log_prefix} QwT (TRE below threshold and selected to skip)")
    return W, b, r2


def _log_tau_percentiles(tau_list):
    if not tau_list:
        return
    logging.info(f"TRE 25th pct: {np.percentile(tau_list, 25):.6f}")
    logging.info(f"TRE 50th pct: {np.percentile(tau_list, 50):.6f}")
    logging.info(f"TRE 75th pct: {np.percentile(tau_list, 75):.6f}")


def recon_vit_module_wise(cur_inp, net, p=0.0, tau_thr=0.007, selected=None):
    """Module-wise QwT on patch-embed-style net (per attn / per mlp).

    selected: optional set of ('patch', i, kind) keys to actually compensate.
              When None, falls back to legacy tau_thr + skip_p gate.
              When set, the legacy tau_thr gate is disabled — only `selected`
              decides who gets compensated (otherwise selected-but-low-TRE modules
              would be killed twice and violate the keep_ratio contract).
    """
    cur_inp = net.prepare_tokens_with_masks(cur_inp, None).cuda()
    n = len(net.blocks)
    if selected is None:
        rand_list, num_enabled = generate_binary_list(p, n * 2)
        logging.info(f"Total {num_enabled}/{n * 2} modules forced-skip by random gate.")
        logging.info(f"Random gate mask: {rand_list}")
    else:
        # Disable legacy gate: rand_skip = (rand_list[i] == 0), all 1s -> False everywhere.
        rand_list = [1] * (n * 2)
        n_sel = sum(1 for k in selected if k[0] == "patch")
        logging.info(f"Ratio selection: {n_sel}/{n * 2} patch_embed modules will be compensated "
                     f"(legacy tau_thr gate disabled)")
    tau_list = []
    with torch.no_grad():
        for i in range(n):
            block = net.blocks[i]
            # ---- attn ----
            force_skip = (selected is not None) and (("patch", i, "attn") not in selected)
            W, b, r2 = _calib_module(
                block, _attn_inner, cur_inp, pos=None, tau_list=tau_list,
                log_prefix=f"Vit block {i} attn", tau_thr=tau_thr, rand_skip=rand_list[i * 2] == 0,
                force_skip=force_skip,
            )
            comp = AttnQwT(W, b, r2, block=block, linear_init=True).cuda()
            cur_inp = (cur_inp + comp(cur_inp).cuda()).cuda()
            net.blocks[i].attn_QwT = comp
            # ---- mlp ----
            force_skip = (selected is not None) and (("patch", i, "mlp") not in selected)
            W, b, r2 = _calib_module(
                block, _mlp_inner, cur_inp, pos=None, tau_list=tau_list,
                log_prefix=f"Vit block {i} mlp", tau_thr=tau_thr, rand_skip=rand_list[i * 2 + 1] == 0,
                force_skip=force_skip,
            )
            comp = MlpQwT(W, b, r2, block=block, linear_init=True).cuda()
            cur_inp = (cur_inp + comp(cur_inp).cuda()).cuda()
            net.blocks[i].mlp_QwT = comp
            net.blocks[i].module_QwT = True
    _log_tau_percentiles(tau_list)
    return net


def recon_blocks_module_wise(cur_inp, B, S, P, C, pos, net, p=0.0, tau_thr=0.007, selected=None):
    """Module-wise QwT on frame_blocks + global_blocks (4 modules per idx: frame_attn, frame_mlp, global_attn, global_mlp).

    selected: optional set of ('frame'|'global', i, kind) keys to compensate.
              When None, falls back to legacy tau_thr + skip_p gate.
              When set, legacy tau_thr gate is disabled (see recon_vit_module_wise docstring).
    """
    cur_inp, pos = cur_inp.cuda(), pos.cuda()
    n = net.aa_block_num
    if selected is None:
        rand_list, num_enabled = generate_binary_list(p, n * 4)
        logging.info(f"Total {num_enabled}/{n * 4} modules forced-skip by random gate.")
        logging.info(f"Random gate mask: {rand_list}")
    else:
        rand_list = [1] * (n * 4)
        n_sel = sum(1 for k in selected if k[0] in ("frame", "global"))
        logging.info(f"Ratio selection: {n_sel}/{n * 4} aggregator modules will be compensated "
                     f"(legacy tau_thr gate disabled)")
    tau_list = []
    with torch.no_grad():
        for i in range(n):
            # ---- per-frame block: attn + mlp ----
            if cur_inp.shape != (B * S, P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)
            if pos is not None and pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)
            block = net.frame_blocks[i]

            force_skip = (selected is not None) and (("frame", i, "attn") not in selected)
            W, b, r2 = _calib_module(
                block, _attn_inner, cur_inp, pos=pos, tau_list=tau_list,
                log_prefix=f"frame block {i} attn", tau_thr=tau_thr,
                rand_skip=rand_list[i * 4] == 0, force_skip=force_skip,
            )
            comp = AttnQwT(W, b, r2, block=block, linear_init=True).cuda()
            cur_inp = (cur_inp + comp(cur_inp, pos=pos).cuda()).cuda()
            net.frame_blocks[i].attn_QwT = comp

            force_skip = (selected is not None) and (("frame", i, "mlp") not in selected)
            W, b, r2 = _calib_module(
                block, _mlp_inner, cur_inp, pos=None, tau_list=tau_list,
                log_prefix=f"frame block {i} mlp", tau_thr=tau_thr,
                rand_skip=rand_list[i * 4 + 1] == 0, force_skip=force_skip,
            )
            comp = MlpQwT(W, b, r2, block=block, linear_init=True).cuda()
            cur_inp = (cur_inp + comp(cur_inp).cuda()).cuda()
            net.frame_blocks[i].mlp_QwT = comp
            net.frame_blocks[i].module_QwT = True

            # ---- cross-frame (global) block: attn + mlp ----
            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)
            if pos is not None and pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            block = net.global_blocks[i]

            force_skip = (selected is not None) and (("global", i, "attn") not in selected)
            W, b, r2 = _calib_module(
                block, _attn_inner, cur_inp, pos=pos, tau_list=tau_list,
                log_prefix=f"global block {i} attn", tau_thr=tau_thr,
                rand_skip=rand_list[i * 4 + 2] == 0, force_skip=force_skip,
            )
            comp = AttnQwT(W, b, r2, block=block, linear_init=True).cuda()
            cur_inp = (cur_inp + comp(cur_inp, pos=pos).cuda()).cuda()
            net.global_blocks[i].attn_QwT = comp

            force_skip = (selected is not None) and (("global", i, "mlp") not in selected)
            W, b, r2 = _calib_module(
                block, _mlp_inner, cur_inp, pos=None, tau_list=tau_list,
                log_prefix=f"global block {i} mlp", tau_thr=tau_thr,
                rand_skip=rand_list[i * 4 + 3] == 0, force_skip=force_skip,
            )
            comp = MlpQwT(W, b, r2, block=block, linear_init=True).cuda()
            cur_inp = (cur_inp + comp(cur_inp).cuda()).cuda()
            net.global_blocks[i].mlp_QwT = comp
            net.global_blocks[i].module_QwT = True

    _log_tau_percentiles(tau_list)
    return net


# ============================================================================
# Strategy dispatchers
# ============================================================================


def compensate_blockwise(q_model, calib_loader, seq_id_map):
    """One CompensationBlock per block (patch_embed + aggregator)."""
    q_model.eval()
    with torch.no_grad():
        inputs = torch.stack(
            [calib_loader.get_data(sequence_name=n, ids=ids)["images"] for n, ids in seq_id_map.items()],
            dim=0,
        ).cuda()
        disable_quant(q_model)
        cur = q_model.aggregator.forward_before_patch_embed(inputs)
        q_model.aggregator.patch_embed = recon_vit(cur, q_model.aggregator.patch_embed)
        logging.info("VGGT patch_embed compensation done.")
        cur, B, S, P, C, pos = q_model.aggregator.forward_before_blocks(inputs)
        q_model.aggregator = recon_blocks(cur, B, S, P, C, pos, q_model.aggregator)
        logging.info("VGGT aggregator compensation done.")
        enable_quant(q_model)
    return q_model.cuda()


def compensate_modulewise(q_model, calib_loader, seq_id_map, p=0.0, tau_thr=0.007,
                          select_metric=None, keep_ratio=None, seed=42):
    """Separate AttnQwT/MlpQwT per block (patch_embed + aggregator).

    Two selection modes:
      - Legacy (select_metric is None): each module's QwT is dropped iff its TRE
        is below tau_thr AND the random gate (with prob p) fires.
      - Ratio (select_metric in {'tre','mse','random'}, keep_ratio in [0,1]):
        first do a no-compensation score pass over all (attn/mlp) modules in
        patch_embed + aggregator, then compensate the top K = round(N * keep_ratio)
        by descending score (for tre/mse) or a random K (for random).
        keep_ratio=1.0 compensates everything; 0.0 compensates nothing.
    """
    q_model.eval()
    with torch.no_grad():
        inputs = torch.stack(
            [calib_loader.get_data(sequence_name=n, ids=ids)["images"] for n, ids in seq_id_map.items()],
            dim=0,
        ).cuda()
        disable_quant(q_model)

        # Optional ratio-based selection: score every module then pick top-K.
        selected_patch, selected_agg = None, None
        if select_metric is not None:
            if keep_ratio is None:
                raise ValueError("compensate.select_metric requires compensate.keep_ratio")
            logging.info(f"Scoring all modules with metric={select_metric} ...")
            cur = q_model.aggregator.forward_before_patch_embed(inputs)
            scores_patch = _score_pass_vit(cur, q_model.aggregator.patch_embed, select_metric)
            cur, B, S, P, C, pos = q_model.aggregator.forward_before_blocks(inputs)
            scores_agg = _score_pass_aggregator(cur, B, S, P, C, pos, q_model.aggregator, select_metric)
            all_scores = {**scores_patch, **scores_agg}
            selected = _pick_selected(all_scores, keep_ratio, select_metric, seed=seed)
            selected_patch = {k for k in selected if k[0] == "patch"}
            selected_agg = {k for k in selected if k[0] in ("frame", "global")}
            logging.info(
                f"Selected {len(selected)}/{len(all_scores)} "
                f"({len(selected_patch)} patch_embed + {len(selected_agg)} aggregator)"
            )
            # Score pass leaves visited blocks in quant mode; reset for compensation pass.
            disable_quant(q_model)

        cur = q_model.aggregator.forward_before_patch_embed(inputs)
        q_model.aggregator.patch_embed = recon_vit_module_wise(
            cur, q_model.aggregator.patch_embed, p=p, tau_thr=tau_thr, selected=selected_patch,
        )
        logging.info("VGGT patch_embed compensation done.")
        cur, B, S, P, C, pos = q_model.aggregator.forward_before_blocks(inputs)
        q_model.aggregator = recon_blocks_module_wise(
            cur, B, S, P, C, pos, q_model.aggregator, p=p, tau_thr=tau_thr, selected=selected_agg,
        )
        logging.info("VGGT aggregator compensation done.")
        enable_quant(q_model)
    return q_model.cuda()


_COMPENSATE_STRATEGIES = {
    "layer": compensate_layerwise,
    "block": compensate_blockwise,
    "module": compensate_modulewise,
}


# ============================================================================
# Evaluation (multi-view point-cloud reconstruction)
# ============================================================================


def evaluate(hydra_cfg, model, logger):
    import open3d as o3d

    all_eval_datasets = hydra_cfg.eval_datasets
    all_data_info = hydra_cfg.data

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in data info: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating MV recon on {dataset_name}...")
        logger.info(f"Sampling strategy: {dataset_info.sampling.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map = json.load(f)

        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        acc_keys = ["Acc-mean", "Acc-med", "Comp-mean", "Comp-med",
                    "NC-mean", "NC-med", "NC1-mean", "NC1-med", "NC2-mean", "NC2-med"]
        agg = {k: 0.0 for k in acc_keys}

        samples_csv = osp.join(output_root, "_all_samples.csv")
        if osp.exists(samples_csv):
            os.remove(samples_csv)

        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            filelist = data["image_paths"]
            images = data["images"]
            gt_pts = data["pointclouds"]
            valid_mask = data["valid_mask"]

            data_h, data_w = images.shape[-2:]
            pred_pts = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))
            assert pred_pts.shape == gt_pts.shape

            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum("nhwj, ij -> nhwi", pred_pts, R) + t.T
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

            # ICP refine; DTU's metric scale is much larger
            threshold = 100 if "DTU" in dataset_name else 0.1
            reg = o3d.pipelines.registration.registration_icp(
                pcd, pcd_gt, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            pcd = pcd.transform(reg.transformation)
            pcd.estimate_normals()
            pcd_gt.estimate_normals()

            acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points,
                                                  np.asarray(pcd_gt.normals), np.asarray(pcd.normals))
            comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points,
                                                      np.asarray(pcd_gt.normals), np.asarray(pcd.normals))
            logger.info(
                f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, "
                f"Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )
            write_csv(samples_csv, {
                "seq": seq_name,
                "Acc-mean": acc, "Acc-med": acc_med,
                "Comp-mean": comp, "Comp-med": comp_med,
                "NC1-mean": nc1, "NC1-med": nc1_med,
                "NC2-mean": nc2, "NC2-med": nc2_med,
            })
            agg["Acc-mean"] += acc; agg["Acc-med"] += acc_med
            agg["Comp-mean"] += comp; agg["Comp-med"] += comp_med
            agg["NC-mean"] += (nc1 + nc2) / 2; agg["NC-med"] += (nc1_med + nc2_med) / 2
            agg["NC1-mean"] += nc1; agg["NC1-med"] += nc1_med
            agg["NC2-mean"] += nc2; agg["NC2-med"] += nc2_med
            torch.cuda.empty_cache()

        n = len(dataset)
        # 统一输出: 所有 run (ratio mode / legacy tau_thr / default) 都 append 到
        # ${output_dir}/tre_diff/all_metrics.csv, 用 dataset + tag + timestamp 列区分。
        # tag = save_suffix (例如 tre_50 / tau_007); 没指定时 fallback 到 "default"。
        tag = OmegaConf.select(hydra_cfg, "save_suffix") or "default"
        row = {"dataset": dataset_name, "tag": tag,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        row.update({k: v / n for k, v in agg.items()})
        stat_path = osp.join(hydra_cfg.output_dir, "tre_diff", "all_metrics.csv")
        os.makedirs(osp.dirname(stat_path), exist_ok=True)
        write_csv(stat_path, row)


# ============================================================================
# Checkpoint save / load (supports both .pt and .json)
# ============================================================================

# Per-module quantizer attributes that need persisting.
_QUANT_ATTRS = ("w_interval", "w_qmax", "a_interval", "a_qmax")


def _module_quant_dict(name, module):
    """Snapshot the quantizer state of a single module."""
    return {
        "name": name,
        "module": module.__class__.__name__,
        "mode": module.mode,
        **{attr: getattr(module, attr, None) for attr in _QUANT_ATTRS},
    }


def _to_jsonable(v):
    """Recursively convert tensors / numpy to lists for json serialisation."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().contiguous().float().tolist()
    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).tolist()
    if isinstance(v, (np.floating, np.integer)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _has_ellipsis(x):
    """Recursively check if a literal-eval result contains Ellipsis (i.e. '...' from truncated repr)."""
    if x is Ellipsis:
        return True
    if isinstance(x, (list, tuple)):
        return any(_has_ellipsis(v) for v in x)
    return False


def _parse_tensor_repr(s):
    """Parse a legacy-txt tensor repr like "tensor([...], device='cuda:0')" into a torch.Tensor.

    Returns None if the repr is truncated (contains '...') so the caller can fall back
    to the module's default attribute value rather than load broken data.
    """
    s = s.strip()
    m = re.match(r"^tensor\((.*)\)$", s, re.DOTALL)
    if not m:
        # Plain numeric string like "8" or "0.5".
        try:
            return torch.tensor(float(s))
        except (ValueError, TypeError):
            return None
    inner = m.group(1).strip()
    # Strip trailing kwargs that ast.literal_eval can't parse: device=, dtype=, requires_grad=
    inner = re.sub(r",\s*device\s*=\s*'[^']*'", "", inner)
    inner = re.sub(r",\s*dtype\s*=\s*[A-Za-z0-9_.]+", "", inner)
    inner = re.sub(r",\s*requires_grad\s*=\s*(True|False)", "", inner)
    try:
        data = ast.literal_eval(inner)
    except (ValueError, SyntaxError) as e:
        logging.warning(f"Cannot parse tensor repr: {s[:80]}... ({e})")
        return None
    if _has_ellipsis(data):
        logging.warning(f"Tensor repr contains '...' (truncated), skipping load: {s[:80]}...")
        return None
    return torch.tensor(data)


def _to_tensor(value, device=None):
    """Convert a (possibly nested-list / number / tensor-repr string) value back to tensor; return as-is otherwise."""
    if value is None or isinstance(value, torch.Tensor):
        t = value
    elif isinstance(value, list):
        t = torch.tensor(value)
    elif isinstance(value, (int, float)):
        t = torch.tensor(float(value))
    elif isinstance(value, str):
        t = _parse_tensor_repr(value)  # legacy .txt format stores tensors as repr strings
    else:
        return value
    return t.to(device) if (t is not None and device is not None) else t


def save_checkpoint(model, path, fmt="pt"):
    """Persist quantizer state (+ compensation LoRA weights if `fmt='pt'`).

    fmt='pt'   torch.save full model state_dict + per-module quantizer attrs.
               Loading restores both calibrated quantizer params AND any
               CompensationBlock / AttnQwT / MlpQwT parameters automatically.
    fmt='json' legacy format: only quantizer attrs (w_interval, etc.). Smaller
               and human-readable but does NOT capture LoRA compensation.
    """
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    if fmt == "pt":
        quant_records = [
            _module_quant_dict(name, m)
            for name, m in model.named_modules()
            if hasattr(m, "mode")
        ]
        payload = {
            "version": 1,
            "format": "pt",
            "state_dict": model.state_dict(),
            "quant_records": quant_records,
        }
        torch.save(payload, path)
    elif fmt == "json":
        quant_records = [
            {k: _to_jsonable(v) for k, v in _module_quant_dict(name, m).items()}
            for name, m in model.named_modules()
            if hasattr(m, "mode")
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(quant_records, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown ckpt fmt: {fmt}")
    logging.info(f"Saved checkpoint -> {path} (fmt={fmt})")


def _apply_quant_records(model, records, dev):
    """Restore quantizer attrs onto model from a list of dicts (json / pt / txt format).

    Modules and records are matched **positionally** (zip), since some legacy .txt
    formats store the class name in the "name" field instead of the dotted module
    path. The dotted-path check is kept only as a soft sanity warning, never as a
    gate on the attribute assignment.

    Raises RuntimeError if any visited module ends up with w_interval / a_interval
    still None — typical when a legacy .txt has '...'-truncated tensor reprs whose
    original data is unrecoverable.
    """
    target_modules = [(n, m) for n, m in model.named_modules() if hasattr(m, "mode")]
    if len(records) != len(target_modules):
        logging.warning(
            f"Quant record count mismatch: ckpt has {len(records)}, model has {len(target_modules)}"
        )

    name_warnings = 0
    skipped = []  # (name, attr, raw_repr) for diagnostics on truncated/unparseable values
    for (name, module), rec in zip(target_modules, records):
        module.calibrated = True
        module.mode = "quant_forward"
        rec_name = rec.get("name", "")
        if name not in rec_name and rec_name != name:
            name_warnings += 1
        for attr in _QUANT_ATTRS:
            raw = rec.get(attr)
            v = _to_tensor(raw, device=dev)
            if v is None and raw is not None:
                skipped.append((name, attr, repr(raw)[:60]))
            elif v is not None:
                if attr == "w_interval" and all(
                    hasattr(module, key) for key in ("n_V", "n_H", "crb_rows", "crb_cols", "weight")
                ):
                    expected_shape = (module.n_V, 1, module.n_H, 1)
                    actual_shape = tuple(v.shape)
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Quant checkpoint granularity mismatch for {name}: "
                            f"w_interval has shape {actual_shape}, but the wrapped module "
                            f"expects {expected_shape} (weight={tuple(module.weight.shape)}, "
                            f"n_V={module.n_V}, n_H={module.n_H}). "
                            "Recalibrate and save a checkpoint with the same "
                            "ptq.linear_channelwise setting."
                        )
                setattr(module, attr, v)
    if name_warnings:
        logging.info(
            f"{name_warnings}/{len(target_modules)} records had a 'name' field "
            f"that doesn't match the dotted module path (likely class name in legacy txt). "
            f"Records were applied positionally — verify ordering looks right."
        )

    # Validation: w_interval / a_interval cannot be None on a module that's now in
    # quant_forward mode, or the next forward pass will TypeError on tensor / None.
    broken = []
    for name, module in target_modules:
        for attr in ("w_interval", "a_interval"):
            if getattr(module, attr, None) is None:
                broken.append((name, attr))
    if broken:
        head = ", ".join(f"{n}.{a}" for n, a in broken[:5])
        more = f" (+{len(broken) - 5} more)" if len(broken) > 5 else ""
        skip_info = ""
        if skipped:
            samples = "; ".join(f"{n}.{a}={r}" for n, a, r in skipped[:3])
            skip_info = (f" Parsing returned None for {len(skipped)} entries — "
                         f"likely '...'-truncated tensor repr. Samples: {samples}")
        raise RuntimeError(
            f"{len(broken)} modules still have None for w_interval/a_interval after load: "
            f"{head}{more}.{skip_info} "
            f"For legacy .txt: re-save the source ckpt after "
            f"`torch.set_printoptions(threshold=float('inf'))`, or use the json/pt sibling."
        )


def load_checkpoint(model, path):
    """Auto-detect ckpt format from extension. .pt restores compensation modules too.

    Supported:
        .pt / .pth   torch payload with state_dict + quant_records (full restore)
        .json        list of quant records with tensors as nested lists (clean modern format)
        .txt         legacy: list of quant records with tensors as Python repr strings;
                     records with '...'-truncated tensors are skipped with a warning
    """
    dev = next(model.parameters()).device
    if path.endswith(".pt") or path.endswith(".pth"):
        payload = torch.load(path, map_location=dev, weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"Not a TAPTQ .pt ckpt: {path}")
        _apply_quant_records(model, payload["quant_records"], dev)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        logging.info(f"Loaded ckpt {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    elif path.endswith(".json"):
        with open(path, "r") as f:
            records = json.load(f)
        _apply_quant_records(model, records, dev)
        logging.info(f"Loaded json ckpt {path} (quantizer params only)")
    elif path.endswith(".txt"):
        # Legacy txt: outer JSON, tensor fields stored as 'tensor(...)' repr strings.
        # _to_tensor's string branch parses each via _parse_tensor_repr.
        with open(path, "r") as f:
            records = json.load(f)
        _apply_quant_records(model, records, dev)
        logging.info(f"Loaded legacy txt ckpt {path} (quantizer params only; "
                     f"truncated '...' tensors are skipped with a warning)")
    else:
        raise ValueError(f"Unknown ckpt extension: {path} (expected .pt/.pth/.json/.txt)")
    return model


# ============================================================================
# Bit-width accounting (utility, currently used in commented log lines)
# ============================================================================


def compute_quantized_params(model):
    """Approximate bit budget of the model (MB). Weights @ 4-bit, LoRA AB @ 8-bit, rest @ dtype bits."""
    total_bits = 0
    for _, module in model.named_modules():
        for k, p in module._parameters.items():
            if p is None:
                continue
            if k == "weight":
                n_bits = 4
            elif "lora_weight" in k:
                n_bits = 0
            elif k in ("A", "B"):
                n_bits = 8
            else:
                n_bits = torch.finfo(p.dtype).bits
            total_bits += p.numel() * n_bits
    return total_bits // 8 / 1e6  # MB


# ============================================================================
# Model registry and Hydra runtime config
# ============================================================================

MODEL_REGISTRY = {
    "vggt": os.environ.get(
        "VGGT_MODEL_PATH",
        osp.join(root, "..", "models", "hf_hub", "models--facebook--VGGT-1B"),
    ),
    "pi3": os.environ.get(
        "PI3_MODEL_PATH",
        osp.join(root, "..", "models", "hf_hub", "models--yyfz233--Pi3"),
    ),
    "vggt_omega": os.environ.get(
        "VGGT_OMEGA_MODEL_PATH",
        osp.join(root, "..", "models", "vggt_omega_1b_512.pt"),
    ),
    "d4rt": os.environ.get("D4RT_MODEL_PATH", ""),
}

# Keep this scope identical to the historical Pi3 PTQ entrypoint. Heads,
# patch embedding, and attention matmuls are intentionally outside this scope.
MODEL_QUANT_SCOPE = {
    "vggt": ("aggregator",),
    "pi3": ("encoder", "decoder", "point_decoder"),
    "vggt_omega": ("aggregator",),
    "d4rt": ("backbone",),
}
MODEL_QUANT_LEAF_NAMES = {
    "vggt": {"qkv", "proj", "fc1", "fc2", "matmul1", "matmul2", "reduction", "head"},
    "pi3": {"qkv", "proj", "fc1", "fc2"},
    "vggt_omega": {"qkv", "proj", "fc1", "fc2"},
    "d4rt": {"qkv", "proj", "q_proj", "k_proj", "v_proj", "fc1", "fc2"},
}
MODEL_QUANT_EXCLUDE_PATTERNS = {
    "vggt": (),
    "pi3": (),
    "vggt_omega": (),
    "d4rt": ("decoder.fourier_embed", "decoder.patch_embed"),
}


class RunConfig:
    def __init__(self, name, cfg_modifier, calib_size, config_name, linear_channelwise=None):
        self.name = name
        self.cfg_modifier = cfg_modifier
        self.calib_size = calib_size
        self.config_name = config_name
        self.linear_channelwise = linear_channelwise  # None: don't override; bool: set


# Global runtime config, set from main entry before Hydra dispatch.
current_run_config: RunConfig = None


def init_config(config_name):
    """Fresh-reload a quant config from ./configs/<name>.py."""
    _, _, files = next(os.walk("./configs"))
    if config_name + ".py" not in files:
        raise NotImplementedError(f"Invalid config name {config_name}")
    quant_cfg = import_module(f"configs.{config_name}")
    reload(quant_cfg)
    return quant_cfg


def _resolve_quant_cfg(hydra_cfg):
    """Materialise a fresh quant config + apply CLI/global overrides."""
    config_name = current_run_config.config_name
    cli_name = OmegaConf.select(hydra_cfg, "ptq.quant_config_name")
    if cli_name not in (None, ""):
        config_name = str(cli_name)

    bit_hydra = OmegaConf.select(hydra_cfg, "ptq.bit")
    if bit_hydra is not None:
        bits = tuple(int(b) for b in bit_hydra)
        if len(bits) != 2:
            raise ValueError(f"ptq.bit must be [w_bit, a_bit], got {bits}")
        current_run_config.cfg_modifier.bit_setting = bits

    quant_cfg = init_config(config_name)
    quant_cfg = current_run_config.cfg_modifier(quant_cfg)

    lc_run = getattr(current_run_config, "linear_channelwise", None)
    if lc_run is not None:
        quant_cfg.linear_channelwise = bool(lc_run)
    lc_hydra = OmegaConf.select(hydra_cfg, "ptq.linear_channelwise")
    if lc_hydra is not None:
        quant_cfg.linear_channelwise = bool(lc_hydra)
    search_mode = OmegaConf.select(hydra_cfg, "ptq.search_mode")
    if search_mode is not None:
        if search_mode not in ("exhaustive", "ternary"):
            raise ValueError(f"ptq.search_mode must be exhaustive or ternary, got {search_mode}")
        quant_cfg.ptqsl_linear_kwargs["search_mode"] = str(search_mode)
    return quant_cfg, config_name


def _validate_and_summarize_scope(model_name, wrapped_modules, logger):
    allowed_top_levels = MODEL_QUANT_SCOPE[model_name]
    allowed_leaf_names = MODEL_QUANT_LEAF_NAMES[model_name]
    invalid = []
    counts = {level: 0 for level in allowed_top_levels}
    for module_name in wrapped_modules:
        parts = module_name.split(".")
        top_level = parts[0]
        leaf = parts[-1]
        if top_level not in allowed_top_levels or leaf not in allowed_leaf_names:
            invalid.append(module_name)
        else:
            counts[top_level] += 1
    if invalid:
        raise RuntimeError(
            f"{model_name} quantization scope mismatch; unexpected modules: {invalid[:8]}"
        )
    logger.info("Quantization scope for %s: %s", model_name.upper(), counts)
    return counts


def build_model_and_wrap(hydra_cfg, logger):
    """Load VGGT/Pi3 and wrap the legacy-aligned linear quantization scope."""
    name = current_run_config.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}, supported: {list(MODEL_REGISTRY.keys())}")
    pretrained = MODEL_REGISTRY[name]
    if name == "vggt":
        model = VGGT.from_pretrained(pretrained).to(hydra_cfg.device).eval()
    elif name == "pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained(pretrained).to(hydra_cfg.device).eval()
    elif name in ("vggt_omega", "d4rt"):
        from mv_recon.foundation_models import load_foundation_model

        model = load_foundation_model(
            name,
            pretrained or None,
            hydra_cfg.device,
            image_resolution=int(OmegaConf.select(hydra_cfg, "foundation.image_resolution") or (512 if name == "vggt_omega" else 256)),
            temporal_size=int(OmegaConf.select(hydra_cfg, "foundation.temporal_size") or 10),
            query_grid=int(OmegaConf.select(hydra_cfg, "foundation.query_grid") or 16),
            variant=str(OmegaConf.select(hydra_cfg, "foundation.variant") or "base"),
            allow_random_init=bool(OmegaConf.select(hydra_cfg, "foundation.allow_random_init") or False),
        )
    else:
        raise ValueError(f"Unsupported model name={name}")
    logger.info("Loaded %s from %s", name.upper(), pretrained)

    quant_cfg, config_name = _resolve_quant_cfg(hydra_cfg)
    wrapped_modules = wrap_modules_in_net(
        model,
        quant_cfg,
        quantize_aggregator=(name in ("vggt", "vggt_omega")),
        quantize_scope=MODEL_QUANT_SCOPE[name],
        exclude_patterns=MODEL_QUANT_EXCLUDE_PATTERNS[name],
    )
    if not wrapped_modules:
        raise RuntimeError(f"No quantizable modules were wrapped for model={name}")
    counts = _validate_and_summarize_scope(name, wrapped_modules, logger)
    logger.info("Wrapped %d quantization modules for %s", len(wrapped_modules), name.upper())
    if name == "pi3" and counts != {"encoder": 96, "decoder": 144, "point_decoder": 20}:
        raise RuntimeError(
            "Pi3 legacy-aligned scope must contain 96 encoder + 144 decoder + "
            f"20 point_decoder linear modules, got {counts}"
        )
    if name == "vggt_omega" and counts != {"aggregator": 288}:
        raise RuntimeError(f"VGGT-Omega aggregator must contain 288 qkv/proj/fc1/fc2 modules, got {counts}")
    if name == "d4rt" and counts != {"backbone": 208}:
        raise RuntimeError(
            "OpenD4RT scope must contain 40 encoder blocks × 4 projections and "
            f"8 decoder blocks × 6 projections, got {counts}"
        )
    return model, wrapped_modules, quant_cfg, config_name


def _print_quant_summary(name, calib_size, config_name, quant_cfg, calib_secs):
    print(f"model: {name}")
    print(f"calibration size: {calib_size}")
    print(f"bit settings: {quant_cfg.bit}")
    print(f"config: {config_name}")
    print(f"linear_channelwise: {getattr(quant_cfg, 'linear_channelwise', False)}")
    print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs}")
    print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs}")
    print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs}")
    print(f"calibration time: {calib_secs / 60:.2f} min")


def _ckpt_default_path(name, quant_cfg, fmt):
    """Build a sensible default ckpt path matching the original naming scheme."""
    bit = quant_cfg.bit
    base = osp.join("param", name, f"channelwise_8scan_w{bit[0]}a{bit[1]}")
    return base + (".pt" if fmt == "pt" else ".json")


# ============================================================================
# Mode handlers
# ============================================================================


def _apply_compensation(hydra_cfg, model, wrapped_modules, dataset, seq_id_map, logger):
    """Dispatch QwT compensation per `compensate.strategy` with CLI param wiring."""
    strategy = OmegaConf.select(hydra_cfg, "compensate.strategy") or "module"
    if current_run_config.name in ("vggt_omega", "d4rt") and strategy != "layer":
        raise ValueError(
            f"{current_run_config.name} currently supports compensate.strategy=layer only; "
            "module/block QwT relies on VGGT-specific block internals"
        )
    if strategy not in _COMPENSATE_STRATEGIES:
        raise ValueError(f"Unknown compensate strategy: {strategy}, supported: {list(_COMPENSATE_STRATEGIES)}")
    compensate_fn = _COMPENSATE_STRATEGIES[strategy]
    logger.info(f"Running QwT compensation with strategy={strategy}")
    if strategy == "layer":
        return compensate_fn(model, wrapped_modules, dataset, seq_id_map)
    if strategy == "module":
        p = float(OmegaConf.select(hydra_cfg, "compensate.skip_p") or 0.0)
        tau_thr = float(OmegaConf.select(hydra_cfg, "compensate.tau_thr") or 0.007)
        select_metric = OmegaConf.select(hydra_cfg, "compensate.select_metric")
        keep_ratio = OmegaConf.select(hydra_cfg, "compensate.keep_ratio")
        seed = int(OmegaConf.select(hydra_cfg, "compensate.seed") or 42)

        # Ablation knobs: wire tail_ratio (TRE) + rank (SVD low-rank) into runtime.
        global _RUNTIME_TAIL_RATIO
        _RUNTIME_TAIL_RATIO = float(OmegaConf.select(hydra_cfg, "compensate.tail_ratio") or 0.01)
        rank_override = OmegaConf.select(hydra_cfg, "compensate.rank")
        if rank_override is not None:
            _QwTBase.rank = int(rank_override)
        logger.info(f"[hparams] tail_ratio={_RUNTIME_TAIL_RATIO}, rank={_QwTBase.rank}, "
                    f"tau_thr={tau_thr}, skip_p={p}")

        return compensate_fn(
            model, dataset, seq_id_map,
            p=p, tau_thr=tau_thr,
            select_metric=(str(select_metric) if select_metric not in (None, "") else None),
            keep_ratio=(float(keep_ratio) if keep_ratio is not None else None),
            seed=seed,
        )
    return compensate_fn(model, dataset, seq_id_map)


def _calibration_dataset(hydra_cfg):
    """Return the frozen calibration dataset and sequence map.

    Calibration must come from ``optim_datasets`` (the authoritative TMM
    setting is ``DTU_train_8``), never from the evaluation list.
    """
    names = OmegaConf.select(hydra_cfg, "optim_datasets")
    if not names:
        names = OmegaConf.select(hydra_cfg, "test_datasets")
    if not names:
        raise ValueError("No calibration dataset configured in optim_datasets/test_datasets")
    dataset_name = str(names[0])
    if dataset_name not in hydra_cfg.data:
        raise ValueError(f"Unknown calibration dataset: {dataset_name}")
    dataset_info = hydra_cfg.data[dataset_name]
    dataset = hydra.utils.instantiate(dataset_info.cfg)
    with open(dataset_info.seq_id_map, "r", encoding="utf-8") as f:
        seq_id_map = json.load(f)
    return dataset_name, dataset_info, dataset, seq_id_map


def run_calibrate(hydra_cfg, *, run_compensate=False, save_ckpt=True):
    """Calibrate (and optionally compensate), then save a ckpt. No evaluation."""
    logger = logging.getLogger("taptq")
    model, wrapped_modules, quant_cfg, config_name = build_model_and_wrap(hydra_cfg, logger)

    dataset_name, dataset_info, dataset, seq_id_map = _calibration_dataset(hydra_cfg)
    logger.info(f"Calibrating on {dataset_name} (sampling: {dataset_info.sampling.strategy})")

    t0 = time.time()
    calibrator = HessianQuantCalibrator(
        model, wrapped_modules, dataset, seq_id_map,
        sequential=False, batch_size=1, device=hydra_cfg.device, logger=logger,
    )
    calibrator.batching_quant_calib()
    enable_quant(model)

    if run_compensate:
        model = _apply_compensation(hydra_cfg, model, wrapped_modules, dataset, seq_id_map, logger)

    _print_quant_summary(current_run_config.name, current_run_config.calib_size,
                         config_name, quant_cfg, time.time() - t0)

    if save_ckpt:
        fmt = OmegaConf.select(hydra_cfg, "ckpt.fmt") or "pt"
        path = OmegaConf.select(hydra_cfg, "ckpt.path") or _ckpt_default_path(
            current_run_config.name, quant_cfg, fmt
        )
        save_checkpoint(model, path, fmt=fmt)
    return model


def run_smoke(hydra_cfg):
    """Build/wrap a model and run one raw forward without calibration."""
    logger = logging.getLogger("taptq")
    model, wrapped_modules, _quant_cfg, _config_name = build_model_and_wrap(hydra_cfg, logger)
    dataset_name, _dataset_info, dataset, seq_id_map = _calibration_dataset(hydra_cfg)
    sequence_name, ids = next(iter(seq_id_map.items()))
    data = dataset.get_data(sequence_name=sequence_name, ids=ids)
    images = data["images"].to(hydra_cfg.device)
    disable_quant(model)
    with torch.no_grad():
        output = model(images)
    if not isinstance(output, dict) or "world_points" not in output:
        raise RuntimeError(f"Smoke output must contain world_points, got {type(output)}")
    points = output["world_points"]
    if not torch.isfinite(points).all():
        raise RuntimeError("Smoke forward produced non-finite world points")
    logger.info(
        "Smoke passed: model=%s dataset=%s wrapped=%d shape=%s",
        current_run_config.name,
        dataset_name,
        len(wrapped_modules),
        tuple(points.shape),
    )


def run_fp_eval(hydra_cfg):
    """Run the full evaluation protocol with quantizers disabled."""
    logger = logging.getLogger("taptq")
    model, _wrapped, _quant_cfg, _config_name = build_model_and_wrap(hydra_cfg, logger)
    disable_quant(model)
    evaluate(hydra_cfg, model, logger)


def run_test(hydra_cfg):
    """Load checkpoint and run evaluation only."""
    logger = logging.getLogger("taptq")
    path = OmegaConf.select(hydra_cfg, "ckpt.path")
    if not path:
        raise ValueError("mode=test requires ckpt.path=<...>")

    model, _wrapped, quant_cfg, _cfg_name = build_model_and_wrap(hydra_cfg, logger)
    load_checkpoint(model, path)
    enable_quant(model)
    evaluate(hydra_cfg, model, logger)


def run_compensate_eval(hydra_cfg):
    """Load a calibrated ckpt, apply QwT compensation, evaluate, optionally save.

    Lets you calibrate once (mode=calib) then compare multiple compensation
    strategies (TRE / MSE / random) without re-running calibration each time.

    Required: ckpt.path=<calibrated json or pt>
    Optional: ckpt.save_path=<...> to persist the compensated model
              (note: re-loading a compensated ckpt via mode=test won't rebuild
              the QwT modules — current load_state_dict uses strict=False).
    """
    logger = logging.getLogger("taptq")
    path = OmegaConf.select(hydra_cfg, "ckpt.path")
    if not path:
        raise ValueError("mode=compensate_eval requires ckpt.path=<calibrated ckpt>")

    model, wrapped_modules, quant_cfg, _cfg_name = build_model_and_wrap(hydra_cfg, logger)
    load_checkpoint(model, path)
    enable_quant(model)
    logger.info(f"Loaded calibrated ckpt from {path}")

    # Compensation uses the same frozen dtu_8 set as calibration.
    dataset_name, dataset_info, dataset, seq_id_map = _calibration_dataset(hydra_cfg)
    logger.info(f"Compensation forward sampled from {dataset_name}")

    model = _apply_compensation(hydra_cfg, model, wrapped_modules, dataset, seq_id_map, logger)

    save_path = OmegaConf.select(hydra_cfg, "ckpt.save_path")
    if save_path:
        fmt = OmegaConf.select(hydra_cfg, "ckpt.fmt") or "pt"
        save_checkpoint(model, save_path, fmt=fmt)

    evaluate(hydra_cfg, model, logger)


def run_e2e(hydra_cfg):
    """Original behaviour: calibrate -> eval(quant-only) -> compensate -> eval -> save."""
    logger = logging.getLogger("taptq")
    model, wrapped_modules, quant_cfg, config_name = build_model_and_wrap(hydra_cfg, logger)

    dataset_name, dataset_info, dataset, seq_id_map = _calibration_dataset(hydra_cfg)
    logger.info(f"Calibrating on {dataset_name} (sampling: {dataset_info.sampling.strategy})")

    t0 = time.time()
    calibrator = HessianQuantCalibrator(
        model, wrapped_modules, dataset, seq_id_map,
        sequential=False, batch_size=1, device=hydra_cfg.device, logger=logger,
    )
    calibrator.batching_quant_calib()
    _print_quant_summary(current_run_config.name, current_run_config.calib_size,
                         config_name, quant_cfg, time.time() - t0)
    enable_quant(model)
    logger.info("Quant-only eval (no compensation):")
    evaluate(hydra_cfg, model, logger)

    model = _apply_compensation(hydra_cfg, model, wrapped_modules, dataset, seq_id_map, logger)
    logger.info("Post-compensation eval:")
    evaluate(hydra_cfg, model, logger)

    fmt = OmegaConf.select(hydra_cfg, "ckpt.fmt") or "json"
    path = OmegaConf.select(hydra_cfg, "ckpt.path") or _ckpt_default_path(
        current_run_config.name, quant_cfg, fmt
    )
    save_checkpoint(model, path, fmt=fmt)

    del model
    torch.cuda.empty_cache()
    logger.info("Finished evaluating PTQ4VGGT on all datasets.")


_MODE_HANDLERS = {
    "smoke": run_smoke,
    "fp_eval": run_fp_eval,
    "calib": lambda cfg: run_calibrate(cfg, run_compensate=False, save_ckpt=True),
    "calib_compensate": lambda cfg: run_calibrate(cfg, run_compensate=True, save_ckpt=True),
    "compensate_eval": run_compensate_eval,
    "test": run_test,
    "e2e": run_e2e,
}


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    if current_run_config is None:
        raise ValueError("RunConfig must be set before invoking main()")

    model_name = OmegaConf.select(hydra_cfg, "model_name")
    if model_name not in (None, ""):
        model_name = str(model_name).lower()
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"model_name must be one of {sorted(MODEL_REGISTRY)}, got {model_name}")
        current_run_config.name = model_name

    mode = OmegaConf.select(hydra_cfg, "mode") or "e2e"
    if mode not in _MODE_HANDLERS:
        raise ValueError(f"Unknown mode: {mode}, supported: {list(_MODE_HANDLERS)}")
    _MODE_HANDLERS[mode](hydra_cfg)


# ============================================================================
# Quant-config modifier (CLI-friendly wrapper around bit/metric/ptq settings)
# ============================================================================


class cfg_modifier:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, cfg):
        cfg.bit = self.bit_setting
        cfg.w_bit = {n: self.bit_setting[0] for n in cfg.conv_fc_name_list}
        cfg.a_bit = {n: self.bit_setting[1] for n in cfg.conv_fc_name_list}
        cfg.A_bit = {n: self.bit_setting[1] for n in cfg.matmul_name_list}
        cfg.B_bit = {n: self.bit_setting[1] for n in cfg.matmul_name_list}

        # conv2d
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False
        # linear
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False
        # matmul
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        lc = getattr(self, "linear_channelwise", None)
        if lc is not None:
            cfg.linear_channelwise = bool(lc)
        return cfg


# ============================================================================
# Entry: build run-config grid and dispatch
# ============================================================================

if __name__ == "__main__":
    names = [os.environ.get("TAPTQ_MODEL", "vggt")]
    metrics = [os.environ.get("TAPTQ_METRIC", "hessian")]
    linear_ptq_settings = [(1, 1, 1)]  # n_V, n_H, n_a
    calib_sizes = [int(os.environ.get("TAPTQ_CALIB_SIZE", "32"))]
    bits = os.environ.get("TAPTQ_BITS", "4,8").split(",")
    if len(bits) != 2:
        raise ValueError("TAPTQ_BITS must be W,A, e.g. 4,8")
    bit_settings = [(int(bits[0]), int(bits[1]))]
    config_names = [os.environ.get("TAPTQ_CONFIG", "PTQ4ViT")]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(
        names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names
    ):
        cfg_list.append({
            "name": name,
            "cfg_modifier": cfg_modifier(
                linear_ptq_setting=linear_ptq_setting,
                metric=metric,
                bit_setting=bit_setting,
            ),
            "calib_size": calib_size,
            "config_name": config_name,
            "linear_channelwise": None,
        })

    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = "1"

    for cfg in cfg_list:
        current_run_config = RunConfig(
            name=cfg["name"],
            cfg_modifier=cfg["cfg_modifier"],
            calib_size=cfg["calib_size"],
            config_name=cfg["config_name"],
            linear_channelwise=cfg.get("linear_channelwise"),
        )
        main()


# ----------------------------------------------------------------------------
# Usage cheatsheet
# ----------------------------------------------------------------------------
# 五种 mode 说明:
#   e2e              原始完整流程: 校准 -> 量化-only 评估 -> QwT 补偿 -> 补偿后评估 -> 保存 ckpt
#                    (在 hydra_cfg.test_datasets 的每个数据集上依次跑)
#   calib            只做量化校准, 保存 ckpt, 不做 QwT 补偿、不做评估
#                    (校准数据从 test_datasets[0] 采样)
#   calib_compensate 量化校准 + QwT 补偿, 保存 ckpt, 不做评估
#                    (compensate.strategy ∈ {layer, block, module}, 默认 module)
#   compensate_eval  加载已校准 ckpt -> QwT 补偿 -> 评估 -> 可选保存
#                    (推荐做对比实验: calib 跑一次, 然后多次复用做不同策略)
#   test             加载已有 ckpt, 只跑评估 (不重新校准/补偿)
#                    (.pt ckpt 加载时不会重建 QwT 模块, 所以仅适合 calib 产物;
#                     带补偿的对比测评请用 e2e 或 compensate_eval)
#
# 常用命令示例:
# CUDA_VISIBLE_DEVICES=1 python taptq.py                                          # 默认 e2e
# CUDA_VISIBLE_DEVICES=1 python taptq.py ++mode=e2e ++ptq.linear_channelwise=true # e2e + 启用 linear 通道级量化
# CUDA_VISIBLE_DEVICES=1 python taptq.py ++mode=e2e ++ptq.bit=[8,8]               # 覆盖比特数: W8A8 (默认 W4A8)
# CUDA_VISIBLE_DEVICES=1 python taptq.py ++mode=calib ++ckpt.path=out/w4a8.pt     # 只校准并保存
# CUDA_VISIBLE_DEVICES=1 python taptq.py ++mode=calib_compensate \                # 校准 + module 级 QwT 补偿
#     ++compensate.strategy=module ++compensate.tau_thr=0.007 ++ckpt.path=out/w4a8.pt
# CUDA_VISIBLE_DEVICES=1 python taptq.py ++mode=test ++ckpt.path=out/w4a8.pt      # 加载 ckpt 直接评估
#
# --- 选择策略对比实验 (TRE vs MSE vs Random, 相同 keep_ratio=0.3) ---
#   推荐流程: 校准跑一次, 然后用 compensate_eval 复用做多组补偿+评估对比。
#
#   # Step 1: 一次性校准, 保存 quantizer 参数 (json 即可, 体积小)
#   python taptq.py ++mode=calib ++ckpt.path=out/w4a8.json ++ckpt.fmt=json
#
#   # Step 2: 用同一份校准 ckpt 跑四组对比, 各自写 metric csv (合并到 tre_diff/all_metrics.csv)
#   python taptq.py ++mode=compensate_eval ++ckpt.path=out/w4a8.json \
#       ++compensate.strategy=module ++compensate.select_metric=tre \
#       ++compensate.keep_ratio=0.3 ++save_suffix=tre_30
#   python taptq.py ++mode=compensate_eval ++ckpt.path=out/w4a8.json \
#       ++compensate.strategy=module ++compensate.select_metric=mse \
#       ++compensate.keep_ratio=0.3 ++save_suffix=mse_30
#   python taptq.py ++mode=compensate_eval ++ckpt.path=out/w4a8.json \
#       ++compensate.strategy=module ++compensate.select_metric=hessian \
#       ++compensate.keep_ratio=0.3 ++save_suffix=hess_30
#   python taptq.py ++mode=compensate_eval ++ckpt.path=out/w4a8.json \
#       ++compensate.strategy=module ++compensate.select_metric=random \
#       ++compensate.keep_ratio=0.3 ++save_suffix=rand_30
#
#   产物: ${output_dir}/tre_diff/all_metrics.csv (累积所有 dataset / 所有 run, 用 dataset+tag 列区分),
#   按 dataset 分组后再按 tag 排序就能直接看相同补偿数量下三种策略的 Acc / Comp / NC 差异。
#   扫不同比例: 0.1 / 0.3 / 0.5 / 0.7 各跑一遍, 用不同 save_suffix (如 tre_10/tre_30/...) 累积到同一 csv。
#   random 多种子稳健性: 额外加 ++compensate.seed=1 / 2 / 3 重跑, save_suffix 取 rand_30_s1 之类。
#
#   备选 (单步 e2e, 每组都重跑校准, 慢但鲁棒):
#   python taptq.py ++mode=e2e ++compensate.strategy=module \
#       ++compensate.select_metric=tre ++compensate.keep_ratio=0.3 \
#       ++save_suffix=tre_30 ++ckpt.path=out/tre_30.pt
#
# Hydra 三种 override 前缀 (重要!):
#   key=value     只能覆盖 eval.yaml 里已存在的 key, 否则报 "not in struct"
#   +key=value    新增 key, 已存在会报错
#   ++key=value   新增或覆盖, 任何情况都能成功 -> 推荐对下面所有自定义 key 使用
#   (eval.yaml 里没预定义 mode/ptq/compensate/ckpt, 所以这些必须用 ++ 或 +)
#
# ----------------------------------------------------------------------------
# 全部可覆盖的 CLI 参数 (Hydra override 语法: key=value, 列表用 [a,b])
# ----------------------------------------------------------------------------
#
# [运行模式]
#   mode=<calib|calib_compensate|compensate_eval|test|e2e>
#         默认 e2e。决定走 _MODE_HANDLERS 里的哪个分支。
#         compensate_eval 是新加的: load 已校准 ckpt -> 补偿 -> 评估, 适合多策略对比。
#
# [checkpoint 相关 ckpt.*]
#   ckpt.path=<file>
#         读/写 ckpt 路径。test / compensate_eval 必填; calib / calib_compensate / e2e
#         不填则用默认路径 param/<model>/channelwise_8scan_w{w}a{a}.{pt|json}。
#   ckpt.fmt=<pt|json>
#         保存格式。pt 同时存 state_dict + 量化参数 (能恢复 QwT 的 LoRA);
#         json 只存量化参数, 体积小、人类可读, 但丢补偿权重。
#         (e2e 默认 json, calib/calib_compensate/compensate_eval 默认 pt; load 时按文件后缀自动识别)
#         读取支持 .pt / .pth / .json / .txt 四种 (.txt 为远古遗留格式, 见下); 写入只产 pt/json。
#   ckpt.path 支持的后缀:
#         .pt / .pth   torch.save 的 payload (state_dict + quant_records), 全量恢复
#         .json        新格式; tensor 用嵌套 list 存
#         .txt         远古遗留格式; 外层 JSON, tensor 字段是 Python repr 字符串
#                      ("tensor([...], device='cuda:0')"). 被 PyTorch '...' 截断的
#                      条目会跳过并 WARNING (无法复原原始数据); 未截断的正常加载。
#   ckpt.save_path=<file>
#         仅 compensate_eval 用; 把补偿后模型另存为该路径。不填则只跑评估不保存。
#         (注意: 当前 mode=test 加载补偿后 ckpt 不会重建 QwT, 复用受限)
#
# [量化超参 ptq.*]
#   ptq.bit=[w_bit, a_bit]
#         权重 / 激活比特数, 默认 (4, 8)。matmul 两路输入 (A_bit/B_bit) 都用 a_bit。
#         也可改 __main__ 里 bit_settings 列表做多组 sweep。
#   ptq.linear_channelwise=<true|false>
#         linear 层是否按输出通道使用独立 scale。开启精度通常更高、计算开销略增。
#   ptq.quant_config_name=<name>
#         切换 ./configs/<name>.py 里的量化策略 (如 PTQ4ViT)。默认沿用
#         current_run_config.config_name (在 __main__ 里写死为 "PTQ4ViT")。
#
# [QwT 补偿超参 compensate.*]   (仅在 calib_compensate 与 e2e 中生效)
#   compensate.strategy=<layer|block|module>
#         补偿粒度, 默认 module。
#         - layer:  逐 nn.Linear/Conv/MatMul 加 CompensationBlock (最细)
#         - block:  每个 transformer block 一个 CompensationBlock
#         - module: 每个 block 拆 attn / mlp 两路 AttnQwT/MlpQwT (低秩 SVD)
#
#   --- 比例选择模式 (新, 仅 strategy=module 生效) ---
#   compensate.select_metric=<tre|mse|hessian|random>
#         按指标对所有 (attn/mlp) 模块全局排序后, 选 top-K 做补偿。
#         - tre:     按 tail relative error 降序选 (大误差优先)
#         - mse:     按 mean squared error 降序选 (大误差优先)
#         - hessian: Fisher-diag 近似 — 用 fp_out 幅度² 加权 MSE, 高量级输出
#                    + 大量化噪声的层优先 (cheap, 无需 backward)
#         - random:  不排序, 随机抽 K 个 (公平基线)
#         不设置 (None) 时走下面的旧 tau_thr/skip_p 阈值模式。
#   compensate.keep_ratio=<float in [0,1]>
#         配合 select_metric, K = round(总模块数 × keep_ratio)。
#         1.0=全部补偿, 0.0=都不补偿。例: 0.3 即只补偿 30% 的模块。
#   compensate.seed=<int>
#         metric=random 的随机种子, 默认 42。tre/mse 不受影响。
#
#   --- 旧的阈值模式 (默认, 当 select_metric 未设置时) ---
#   compensate.skip_p=<float in [0,1]>
#         随机选 p 比例的子模块强制跳过 QwT (做消融用), 默认 0.0。
#   compensate.tau_thr=<float>
#         tail relative error 阈值, 默认 0.007。
#         本身量化误差就低 (TRE<thr) 且被随机门选中的模块, r2 设为 -3 -> QwT 退化为 0。
#         注意 skip_p=0.0 (默认) 时随机门全 True, 等价"纯阈值"模式。
#         skip_p=1.0 反而禁用阈值门 (rand_skip 全 False)。
#
#   --- 消融 (改变 TRE / SVD 行为, 任何模式都生效) ---
#   compensate.tail_ratio=<float in (0,1]>
#         tail_relative_error() 取 fp_out 绝对值 top-K 的 K=ratio*numel, 默认 0.01 (1%)。
#         调大 (如 0.05/0.1) 让 TRE 看更广的分布; 调小让指标更聚焦极端 outlier。
#   compensate.rank=<int>
#         _QwTBase SVD 低秩补偿的秩, 默认 32。直接影响 QwT 的表达能力 / 显存。
#
# [数据 / 评估 (来自 configs/eval.yaml, 也可 CLI 覆盖)]
#   eval_datasets=[ds1,ds2,...]   evaluate() 跑哪些数据集 (test 与 e2e 用)
#   test_datasets=[ds1,ds2,...]   校准用哪些数据集 (calib/calib_compensate 用第 0 个)
#   data.<name>.cfg=...           Hydra instantiate 的具体数据集配置
#   output_dir=<path>             点云、metric csv 的输出根目录
#   save_suffix=<str>             给 metric csv 行加 tag, 标识这次实验 (例如 tre_50 / tau_007 / default)。
#                                 所有 mode/run 都合写到 ${output_dir}/tre_diff/all_metrics.csv,
#                                 一行/run, 用 dataset+tag+timestamp 列区分。
#                                 不设时 tag 默认 "default"。
#   device=<cuda|cuda:0|cpu>      模型设备 (默认 cuda)
#
# [Hydra 内置常用]
#   hydra.run.dir=<path>          覆盖 Hydra 的 run 输出目录
#   --config-name=<yaml>          换主 config (默认 ../configs/eval.yaml)
#   ++key=value                   强制新增 key (eval.yaml 里没定义时用, 比如 ++mode=test)
