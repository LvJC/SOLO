export PYTHONPATH=`pwd`:$PYTHONPATH

# class-agnostic
CONFIG='solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_crop'

# Notice! Need to check 
#   mmdet/datasets/synthetics.py#21-244
#   tools/test_ins_vis.py#44 vis_seg()
CUDA_VISIBLE_DEVICES=2 python tools/test_ins_vis.py \
configs/solov2/synthetics/solov2_x101_dcn_fpn_8gpu_3x.py \
work_dirs/$CONFIG/latest.pth \
--out results/$CONFIG/test.pkl \
--eval segm \
--save_dir results/$CONFIG/viz/ \
--show


# # class-specified
# CONFIG='solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask'

# # Notice! Need to check 
# #   mmdet/datasets/synthetics.py#21-244
# #   tools/test_ins_vis.py#44 vis_seg()
# CUDA_VISIBLE_DEVICES=2 python tools/test_ins_vis.py \
# configs/solov2/synthetics/solov2_x101_dcn_fpn_8gpu_3x.py \
# work_dirs/$CONFIG/latest.pth \
# --out results/$CONFIG/test.pkl \
# --eval segm \
# --save_dir results/$CONFIG/viz/ \
# --show

# # python tools/crop_patch.py \
