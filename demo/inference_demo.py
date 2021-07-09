from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv


config_file = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/SOLOv2_X101_DCN_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = 'demo.jpg'
# img = 'demo_in/00210a6681cdfab738e094a2409e9509.png'
# result = inference_detector(model, img)
# show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out/00210a6681cdfab738e094a2409e9509_out.jpg")

# test a forlder of images
import glob
import os.path as osp
# for img in glob.glob('demo_in/*.png'):
for img in glob.glob("/ldap_home/regds_cvcore/product_segmentation/datasets/opensource/COCO/val2017/*.jpg")[:10]:
    imgname = osp.basename(img).split('.')[0]
    result = inference_detector(model, img)
    show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=f"COCO/val2017/{imgname}.jpg")

