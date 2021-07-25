import mmcv
from mmcv.runner import checkpoint

from mmdet.apis import (inference_detector, init_detector, show_result_ins,
                        show_result_pyplot,)

# TOFIX: show_result_pyplot is of no use

# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
config_file = '../configs/solov2/synthetics/solov2_x101_dcn_fpn_8gpu_3x_clsag.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
checkpoint_file = '../work_dirs/solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_clsag_fixhead_fixcate/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
import pdb; pdb.set_trace()

# test a single image
img = '/ldap_home/jincheng.lyu/data/product_segmentation/thin+hard/test/20c52038386534cf5ac006ab2e9b2f77.png'
result = inference_detector(model, img)
import pdb; pdb.set_trace()

img_show, cur_mask = show_result_ins(img, result, model.CLASSES, score_thr=0.3)#, out_file="demo_out/20c52038386534cf5ac006ab2e9b2f77.jpg")
import pdb; pdb.set_trace()
