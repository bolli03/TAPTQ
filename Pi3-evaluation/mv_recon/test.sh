# python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.select_metric=tre \
#     ++compensate.keep_ratio=0.50 ++save_suffix=tre_50

# python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.select_metric=mse \
#     ++compensate.keep_ratio=0.50 ++save_suffix=mse_50

# python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.select_metric=random \
#     ++compensate.keep_ratio=0.50 ++save_suffix=rand_50

# python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.select_metric=tre \
#     ++compensate.keep_ratio=0.25 ++save_suffix=tre_25

# python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.select_metric=mse \
#     ++compensate.keep_ratio=0.25 ++save_suffix=mse_25

# python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.select_metric=random \
#     ++compensate.keep_ratio=0.25 ++save_suffix=rand_25

python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
    ++compensate.strategy=module ++compensate.select_metric=hessian \
    ++compensate.keep_ratio=0.50 ++save_suffix=hess_50

python mv_recon/taptq.py ++mode=compensate_eval ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
    ++compensate.strategy=module ++compensate.select_metric=hessian \
    ++compensate.keep_ratio=0.25 ++save_suffix=hess_25

# python mv_recon/taptq.py ++mode=compensate_eval \
#     ++ckpt.path=/root/Pi3-evaluation/param/vggt/8scan_w4a8.txt \
#     ++compensate.strategy=module ++compensate.tau_thr=0.007 \
#     ++compensate.skip_p=1.0 ++save_suffix=tau_007
