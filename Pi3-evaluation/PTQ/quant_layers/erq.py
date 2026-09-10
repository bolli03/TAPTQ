"""VGGT port of zysxmu/ERQ official Aqer and Wqer algorithms."""
import torch
import torch.nn.functional as F

from .baseline import AffineQParams
from .official_quant import QParams, affine_fake_quant, erq_two_part_qparams, repq_qparams


def _quantize(tensor, qparams, rounding="round"):
    scale = qparams.scale.to(tensor.device, tensor.dtype)
    zero = qparams.zero.to(tensor.device, tensor.dtype)
    if scale.shape != tensor.shape:
        shape = [1] * tensor.ndim
        if qparams.axis is not None:
            shape[qparams.axis] = tensor.shape[qparams.axis]
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
    values = tensor / scale + zero
    if rounding == "round":
        values = values.round()
    elif rounding == "floor":
        values = values.floor()
    elif rounding == "ceil":
        values = values.ceil()
    else:
        raise ValueError(rounding)
    return (values.clamp(0, 2**qparams.bits - 1) - zero) * scale


def _factor(matrix):
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    mean = matrix.diagonal().abs().mean().clamp_min(1e-8)
    for jitter in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * mean * eye)
        if int(info.max().item()) == 0:
            return factor
    raise RuntimeError("ERQ ridge system is not positive definite")


def _solve(matrix, rhs, batch_size):
    factor = _factor(matrix)
    return torch.cat([
        torch.cholesky_solve(rhs[:, start:start + batch_size], factor)
        for start in range(0, rhs.shape[1], batch_size)
    ], dim=1)


@torch.no_grad()
def aqer(weight, bias, fp_input, quant_input, ridge, batch_size=128, group_size=0):
    """Official replace_W ridge correction; group_size is only a memory fallback."""
    x, xq, weight = fp_input.float(), quant_input.float(), weight.float()
    fp_bias = None if bias is None else bias.float()
    target = F.linear(x, weight, fp_bias)
    adjusted_weight = weight.clone()
    adjusted_bias = None if fp_bias is None else fp_bias.clone()
    current = F.linear(xq, adjusted_weight, adjusted_bias)
    width = xq.shape[1]
    step = width if not group_size else min(group_size, width)
    for start in range(0, width, step):
        end = min(width, start + step)
        design = xq[:, start:end]
        include_bias = adjusted_bias is not None and start == 0
        if include_bias:
            design = torch.cat((design, torch.ones(design.shape[0], 1, device=design.device)), dim=1)
        residual = target - current
        system = design.t() @ design
        system.diagonal().add_(ridge)
        correction = _solve(system, design.t() @ residual, batch_size)
        adjusted_weight[:, start:end].add_(correction[:end - start].t())
        if include_bias:
            adjusted_bias.add_(correction[-1])
        current.add_(design @ correction)
    return adjusted_weight, adjusted_bias


@torch.no_grad()
def _refine(raw, gram, qparams, iterations, row_batch):
    """Official Rounding Refinement, vectorized over output rows, top-k=1."""
    nearest = _quantize(raw, qparams)
    floor = _quantize(raw, qparams, "floor")
    ceil = _quantize(raw, qparams, "ceil")
    for start in range(0, raw.shape[0], row_batch):
        end = min(raw.shape[0], start + row_batch)
        original = raw[start:end]
        error = nearest[start:end] - original
        floor_error = floor[start:end] - original
        ceil_error = ceil[start:end] - original
        failed = torch.zeros(end - start, dtype=torch.bool, device=raw.device)
        for _ in range(iterations):
            gradient = 2 * (gram @ error.transpose(0, 1)).transpose(0, 1)
            gradient[(error * gradient) <= 0] = 0
            gradient[failed] = 0
            active = gradient.abs().amax(dim=1) > 0
            if not bool(active.any()):
                break
            indices = gradient.abs().argmax(dim=1, keepdim=True)
            proposal = error.gather(1, indices) - gradient.gather(1, indices)
            floor_value = floor_error.gather(1, indices)
            ceil_value = ceil_error.gather(1, indices)
            proposal = torch.where(
                (proposal - ceil_value).abs() <= (proposal - floor_value).abs(),
                ceil_value,
                floor_value,
            )
            before = torch.einsum("bi,ij,bj->b", error, gram, error)
            previous = error.gather(1, indices).clone()
            error.scatter_(1, indices, proposal)
            after = torch.einsum("bi,ij,bj->b", error, gram, error)
            reject = (after > before) | ~active
            if bool(reject.any()):
                rows = torch.arange(end - start, device=raw.device)[reject]
                error[rows, indices[reject, 0]] = previous[reject, 0]
            failed |= reject
            if bool(failed.all()):
                break
        nearest[start:end] = original + error
    return nearest


def _column_qparams(qparams, columns):
    if qparams.scale.ndim == 2:
        return QParams(qparams.scale[:, columns], qparams.zero[:, columns], qparams.bits)
    return qparams


@torch.no_grad()
def wqer(weight, quant_input, qparams, ridge, iterations=100, row_batch=500, group_size=0):
    """Official progressive first-half quantization and ridge error transfer."""
    if qparams.axis != 0 and qparams.scale.shape != weight.shape:
        raise ValueError("Wqer requires per-output-channel or per-weight qparams")
    if group_size and weight.shape[1] > group_size:
        output = weight.float().clone()
        for start in range(0, weight.shape[1], group_size):
            end = min(weight.shape[1], start + group_size)
            columns = torch.arange(start, end, device=weight.device)
            output[:, start:end] = wqer(
                weight[:, start:end], quant_input[:, start:end], _column_qparams(qparams, columns),
                ridge, iterations, row_batch, 0,
            )
        return output
    work = weight.float().clone()
    gram = quant_input.float().t() @ quant_input.float()
    remaining = torch.arange(work.shape[1], device=work.device)
    while remaining.numel():
        count = max(1, remaining.numel() // 2)
        selected, rest = remaining[:count], remaining[count:]
        raw = work[:, selected].clone()
        selected_qparams = _column_qparams(qparams, selected)
        quantized = _refine(raw, gram[selected][:, selected], selected_qparams, iterations, row_batch)
        work[:, selected] = quantized
        if rest.numel():
            system = gram[rest][:, rest].clone()
            system.diagonal().add_(ridge)
            rhs = -((quantized - raw) @ gram[selected][:, rest]).t().contiguous()
            work[:, rest].add_(_solve(system, rhs, row_batch).t())
        remaining = rest
    return work


@torch.no_grad()
def rtn_weight(weight, bits, candidates=20):
    del candidates
    return affine_fake_quant(weight.float(), repq_qparams(weight, bits, axis=0))


@torch.no_grad()
def erq_linear(
    module, fp_input, w_bit, ridge, candidates=20, iterations=100,
    row_batch=500, group_size=0, two_part=False,
):
    x = fp_input.reshape(-1, module.in_features).to(module.weight.device, torch.float32)
    aqparams = repq_qparams(x, module.a_bit)
    module.set_input_qparams(AffineQParams(aqparams.scale, aqparams.zero, module.a_bit))
    xq = affine_fake_quant(x, aqparams)
    original_weight = module.weight.detach().float()
    original_bias = None if module.bias is None else module.bias.detach().float()
    adjusted_weight, adjusted_bias = aqer(
        original_weight, original_bias, x, xq, ridge, row_batch, group_size
    )
    if two_part:
        wqparams, top_indices = erq_two_part_qparams(adjusted_weight, w_bit)
    else:
        wqparams = repq_qparams(adjusted_weight, w_bit, axis=0)
        top_indices = torch.empty(0, dtype=torch.long, device=adjusted_weight.device)
    quant_weight = wqer(
        adjusted_weight, xq, wqparams, ridge, iterations, row_batch, group_size
    )
    target = F.linear(x, original_weight, original_bias)
    rtn = F.linear(xq, rtn_weight(original_weight, w_bit, candidates), original_bias)
    output = F.linear(xq, quant_weight, adjusted_bias)
    module.weight.copy_(quant_weight.to(module.weight.dtype))
    if module.bias is not None:
        module.bias.copy_(adjusted_bias.to(module.bias.dtype))
    return {
        "tokens": x.shape[0],
        "rtn_mse": F.mse_loss(rtn, target).item(),
        "erq_mse": F.mse_loss(output, target).item(),
        "two_part": bool(two_part),
        "two_part_columns": int(top_indices.numel()),
    }
