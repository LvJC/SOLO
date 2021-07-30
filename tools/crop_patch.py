#! python3
# -*- encoding: utf-8 -*-
'''
@Time    :   2021/07/24
@Author  :   jincheng.lyu
'''
import cv2
import numpy as np
import os.path as osp
import requests
from tqdm import tqdm
from typing import Any, Union


def get_test_img(img_str: str) -> Union[np.ndarray, bytes]:
    """
    get image array from image hash, image url or image path.
    :param img_hash: image hash, image url, or image path.
    :return: image array.
    """
    if osp.isfile(img_str):
        with open(img_str, "rb") as f:
            img_bytes = f.read()
        img = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_bytes


def get_box_from_seg(img: np.ndarray, th=0.5, expand=10) -> Union[int, Any]:
    """Return bbox from a segmentation result.

    Args:
        img (np.ndarray): 4-channel image.
        th (float, optional): threshold for mask. Defaults to 0.5.
        expand (int, optional): expansion for bbox. Defaults to 10.

    Returns:
        Union[int, Any]: [description]
    """
    mask = img[:, :, 3]
    h, w = mask.shape
    y, x = np.where(mask >= th * 255)
    x2 = min(x.max() + expand, w)
    x1 = max(x.min() - expand, 0)
    y2 = min(y.max() + expand, h)
    y1 = max(y.min() - expand, 0)
    return x1, y1, x2, y2

def run_on_image(img: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        img (np.ndarray): np.ndarray, RGBA image.

    Returns:
        np.ndarray: RGBA image
    """
    x1, y1, x2, y2 = get_box_from_seg(img)
    patch = img[y1:y2, x1:x2, :]
    return patch

def proc_one_image(src_path: str, dst_dir: str):
    img, img_bytes = get_test_img(src_path)
    imgname = osp.basename(src_path)
    patch = run_on_image(img)
    dst_path = osp.join(dst_dir, imgname)
    cv2.imwrite(dst_path, cv2.cvtColor(patch, cv2.COLOR_RGBA2BGRA))
    return dst_path


if __name__ == '__main__':
    import concurrent.futures
    import glob
    import mmcv
    import os

    pred_dir = '/ldap_home/jincheng.lyu/data/product_segmentation/synthetics/val'
    crop_pred_dir = '/ldap_home/jincheng.lyu/data/product_segmentation/synthetics_crop/val'
    img_list = glob.glob(pred_dir + '/**/*.png')
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {}
        for i, imgpath in enumerate(tqdm(img_list)):
            subcat = imgpath.split('/')[-2]
            os.makedirs(osp.join(crop_pred_dir, subcat), exist_ok=True)
            filename = osp.basename(imgpath)
            res_path = osp.join(crop_pred_dir, subcat, filename)
            futures[executor.submit(proc_one_image, imgpath, osp.join(crop_pred_dir, subcat))] = res_path

        prog_bar = mmcv.ProgressBar(len(img_list))
        for future in concurrent.futures.as_completed(futures):
            res_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (res_path, exc))
            prog_bar.update()


    # for subcat in os.listdir(pred_dir):
    #     img_list = glob.glob(pred_dir + f'/{subcat}/*.png')
    #     print(f"*** Processing subcategory {subcat}:")
    #     os.makedirs(osp.join(crop_pred_dir, subcat), exist_ok=True)
    #     for i in tqdm(range(len(img_list))):
    #         try:
    #             img, img_bytes = get_test_img(img_list[i])
    #             patch = run_on_image(img)
    #             res_name = osp.join(crop_pred_dir, subcat, osp.basename(img_list[i]))
    #             cv2.imwrite(res_name, cv2.cvtColor(patch, cv2.COLOR_RGBA2BGRA))
    #         except Exception as e:
    #             print("image: {} processing error: {}".format(img_list[i], e))
