
CUDA_VISIBLE_DEVICES=0 python run_co3d.py \
    --model_path ../VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /wmq/3d_datasets/ \
    --co3d_anno_dir /wmq/datasets/co3d_v2_annotations/ \
    --dtype quarot_w4a4\
    --seed 0 \
    --lac \
    --lwc \
    --cache_path ./outputs/cache_data.pt \
    --class_mode all \
    --each_nsamples 10 \
    --exp_name a44 \
    --fast_eval \
    --resume_qs \




