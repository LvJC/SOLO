import mmcv
import numpy as np
from mmcv.runner import checkpoint

from mmdet.apis import (inference_detector, init_detector, show_result_ins,
                        show_result_pyplot,)
from tools.crop_patch import proc_one_image

# TOFIX: show_result_pyplot is of no use

# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
config_file = '../configs/solov2/synthetics/solov2_x101_dcn_fpn_8gpu_3x_clsag.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
checkpoint_file1 = '../work_dirs/solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_clsag_fixhead_fixcate/latest.pth'
checkpoint_file2 = '../work_dirs/solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_clsag_fixhead_fixcate_crop/latest.pth'


# build the model from a config file and a checkpoint file
model1 = init_detector(config_file, checkpoint_file1, device='cuda:0')
model2 = init_detector(config_file, checkpoint_file2, device='cuda:0')


# test a single image
# img = '/ldap_home/jincheng.lyu/data/product_segmentation/thin+hard/test/20c52038386534cf5ac006ab2e9b2f77.png'
name = 'test0728_3'
img = f'/ldap_home/jincheng.lyu/project/SOLO/demo/demo_in/{name}.jpg'

result1 = inference_detector(model1, img)
img_show1, cur_mask1 = show_result_ins(img, result1, model1.CLASSES, score_thr=0.3)

png1 = np.dstack((mmcv.imread(img), cur_mask1*255))
mmcv.imwrite(png1, f'demo_out/{name}_stage1.png')

img_crop_path = proc_one_image(f'demo_out/{name}_stage1.png', 'demo_in/')
result2 = inference_detector(model2, img_crop_path)

img_show2, cur_mask2 = show_result_ins(img_crop_path, result2, model2.CLASSES, score_thr=0.3, out_file=f"demo_out/{name}_stage2.jpg")
png2 = np.dstack((mmcv.imread(img_crop_path), cur_mask2*255))
mmcv.imwrite(png2, f'demo_out/{name}_stage2.png')
