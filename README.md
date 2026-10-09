# TAPTQ — Tail-Aware Post-Training Quantization for 3D Reconstruction

English | [简体中文](README.zh-CN.md)

TAPTQ is a **tail-aware post-training quantization (PTQ)** framework for 3D geometry
models, including multi-view reconstruction Transformers such as VGGT. It includes
a unified multi-view reconstruction (MV-Recon) evaluation pipeline and baseline
implementations.

Key features:

- **Unified quantization layers**: `PTQ/quant_layers` supports W4A8, W6A6, and W8A8,
  with configurable per-tensor, per-channel, and **per-output-channel (channel-wise)**
  quantization.
- **QwT compensation**: `mv_recon/taptq.py` applies module-wise tail-aware
  compensation with the calibration parameters held fixed to reduce geometric
  degradation at low bit widths.
- **Unified evaluation protocol**: `mv_recon/eval.py` evaluates 7-Scenes
  (dense, `kf=40`) and ETH3D (13 sequences, `kf=5`) using fixed frame maps,
  Sim(3)+ICP point-cloud alignment, and consistent Acc. / Comp. / N.C. computation
  for fair comparisons across methods.
- **Baselines under a shared protocol**: `mv_recon/external_baselines.py` and
  `quant_layers` integrate RTN, PTQ4ViT, RepQ-ViT, ERQ, GPTQ, SmoothQuant,
  SVDQuant, and QuantVGGT under the same evaluation protocol.

---

## Repository Structure

```
TAPTQ/
├── Pi3-evaluation/             # Main code: unified evaluation and TAPTQ quantization/compensation
│   ├── mv_recon/               # Multi-view reconstruction evaluation and TAPTQ workflow
│   │   ├── taptq.py            # Calibration / quantized evaluation / QwT compensation / e2e
│   │   ├── eval.py             # Unified Acc./Comp./N.C. evaluation
│   │   ├── ptq.py              # Quantized model construction
│   │   ├── baseline_quant.py   # Quantization wrappers for baselines
│   │   └── external_*.py       # External baseline and base model integration
│   ├── PTQ/                   # Quantization layers and calibration utilities
│   │   ├── quant_layers/      # Linear/conv quantizers and baseline quantizers
│   │   ├── vggt/              # VGGT model integration
│   │   └── utils/             # Hessian-based and baseline calibration
│   ├── configs/               # Hydra configurations tracked in Git
│   ├── datasets/              # Preprocessing scripts and sequence ID maps
│   └── deployment/            # W4A8 (int8) / W8A8 (Triton) deployment and validation
├── projects/QuantVGGT/         # QuantVGGT integration with the shared evaluation protocol
├── third_party/               # External baseline repositories, excluded from Git by default
├── vendor/                    # Third-party data/code, excluded from Git by default
├── doc/                       # Experiment plans, paper drafts, result inventories, and manifests
└── scripts/                   # Batch experiment scripts
```

> External baselines in `third_party/` and `vendor/` have their own Git repositories
> and are excluded by `.gitignore`. Add them separately after cloning; they are not
> part of the TAPTQ core code.

---

## Quick Start

```bash
cd Pi3-evaluation

# 1) Calibrate and save quantization parameters (usually needed only once)
python mv_recon/taptq.py \
  ++mode=calib \
  ptq.bit=[8,8] ptq.search_mode=ternary ptq.linear_channelwise=true \
  'optim_datasets=[DTU_train_8]' \
  ckpt.fmt=pt ckpt.path=/path/to/ckpt.pt

# 2) Reuse calibration parameters, apply QwT compensation, and evaluate
#    (useful for repeated comparisons)
python mv_recon/taptq.py \
  ++mode=compensate_eval \
  'eval_datasets=[ETH3D]' \
  ptq.bit=[8,8] ptq.search_mode=ternary ptq.linear_channelwise=true \
  'optim_datasets=[DTU_train_8]' \
  ckpt.fmt=pt ckpt.path=/path/to/ckpt.pt \
  ++compensate.strategy=module ++compensate.tail_ratio=0.01 \
  ++compensate.tau_thr=0.005 ++compensate.rank=256

# 3) Load quantization parameters and evaluate without compensation
python mv_recon/taptq.py ++mode=test 'eval_datasets=[ETH3D]' ...
```

Available `mode` values:

| Mode | Behavior |
|---|---|
| `calib` | Calibrate and save quantization parameters, without evaluation or compensation |
| `calib_compensate` | Calibrate, apply QwT compensation, and save, without evaluation |
| `compensate_eval` | Load saved parameters, apply compensation, and evaluate |
| `test` | Load saved parameters and evaluate the quantized model directly |
| `e2e` | Calibrate → evaluate the quantized model → compensate → evaluate and save the compensated model |

---

## Quantization Granularity and Hyperparameters

The main ETH3D results use the **selected configuration** for each bit width.
Hyperparameters are listed in the paper appendix; quantization granularity
terminology is not detailed here.

| Bit Width | Reference Configuration `(ρ, τ, r)` | Selected Configuration `(ρ, τ, r)` |
|---|---|---|
| W4A8 | (0.1, 0.007, 256) | (0.01, 0.005, 128) |
| W6A6 | (0.1, 0.007, 16)  | (0.01, 0.005, 256) |
| W8A8 | (0.1, 0.007, 16)  | (0.01, 0.005, 256) |

> Earlier W6A6 quantization records that did not follow the shared protocol are
> deprecated. The current channel-wise W6A6 and W8A8 results use checkpoints
> recalibrated with the same pipeline as W8A8:
> `DTU_train_8 + ternary + channel-wise`.

---

## Datasets and Artifacts

- **Calibration**: `DTU_train_8` (8 scans × 10 frames, with a 4096-token budget).
- **Evaluation**: 7-Scenes dense (`kf=40`) and ETH3D (13 sequences, `kf=5`),
  using Pi3-compatible fixed frame maps.
- Model weights, large point clouds, and data files such as `.pth`, `.pt`, `.ply`,
  and `.npy` are excluded by `.gitignore`. Store evaluation outputs and intermediate
  data locally or on external storage such as CephFS, and keep them out of Git.

---

## Results

The main experimental results, including the selected ETH3D configurations and
baseline comparisons under the shared protocol, are recorded in
`doc/TMM_EXPERIMENTS_RESTRUCTURE_DRAFT.tex` and `doc/tmm_*_manifest.json`.
The manifests record the configuration, calibration protocol, and output artifacts
for each experiment to support reproducibility.
