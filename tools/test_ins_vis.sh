export PYTHONPATH=`pwd`:$PYTHONPATH
CONFIG='solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_clsag_fixhead'

CUDA_VISIBLE_DEVICES=2 python tools/test_ins_vis.py \
configs/solov2/synthetics/solov2_x101_dcn_fpn_8gpu_3x_clsag.py \
work_dirs/$CONFIG/epoch_1.pth \
--out results/$CONFIG/test.pkl \
--eval segm \
--save_dir results/$CONFIG/viz/ \
--show
