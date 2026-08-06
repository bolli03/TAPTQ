"""
Same quant setup as PTQ4ViT, but linear layers use per-output-channel weight scales.

Equivalent to PTQ4ViT + `linear_channelwise=True` (see configs/PTQ4ViT.py).
"""
import configs.PTQ4ViT as _base

_base.linear_channelwise = True

from configs.PTQ4ViT import *  # noqa: F401,F403
