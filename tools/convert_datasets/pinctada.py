#! python3
# -*- encoding: utf-8 -*-
'''
@Time    :   2021/07/13
@Author  :   jincheng.lyu@shopee.com

1. Get all categories
2. Generate label for each image
3. All images by using multithreading
'''
import cv2
import datetime
import glob
import imagesize
import json
import numpy as np
import os
import os.path as osp
from pycococreatortools import pycococreatortools
from pycocotools.mask import area, decode, encode, frPyObjects, toBbox
from tqdm import tqdm
from typing import List

INFO = {
    "description": "Leaf Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Jincheng_Lyu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

def getSize(imgpath):
    width, height = imagesize.get(imgpath)
    # print(width, height)
    return width, height

# def encodeBinMask(imgpath):
#     img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
#     mask = img[:, :, -1]
#     uint8_mask = np.uint8(mask)
#     bin_mask = np.where(mask > 128, 1, 0)

#     Rs = encode(np.asfortranarray(uint8_mask))
#     uint8_mask = decode(Rs)

#     binRs = encode(np.asfortranarray(bin_mask, dtype=np.uint8))
#     bin_mask = decode(binRs)

#     x, y, w, h = toBbox(binRs)
#     annotation_info = pycococreatortools.create_annotation_info(
#                             segmentation_id, image_id, category_info, bin_mask,
#                             image.size, tolerance=2) 
    
#     return 

def genCats(dataroot) -> List:
    dataroot = "/ldap_home/jincheng.lyu/data/product_segmentation/synthetics/train"
    cat_list = os.listdir(dataroot)
    cat_list.sort()
    categories = []
    # cat id starts from 1
    for idx, cat in enumerate(cat_list):
        categories.append({
            "supercategory": cat,
            "id": idx+1,  # noqa
            "name": cat
        })
    return categories

def genImgs(dataroot):
    imgpaths = glob.glob(dataroot + '/*/*.png')
    imgpaths.sort()
    images = []
    # img id starts from 0
    for id, imgpath in enumerate(imgpaths):
        w, h = getSize(imgpath)
        filename = osp.basename(imgpath)
        images.append({
            "height": h,
            "width": w,
            "id": id,
            "file_name": filename
        })
    return images

def genAnns(dataroot):
    imgpaths = glob.glob(dataroot + '/*/*.png')
    imgpaths.sort()
    CATEGORIES = genCats(dataroot)
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # img id starts from 0
    segmentation_id = 1
    for image_id, imgpath in enumerate(tqdm(imgpaths)):
        # image
        filename = osp.basename(imgpath)
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        image_info = pycococreatortools.create_image_info(
            image_id,
            filename,
            getSize(imgpath)
        )
        coco_output["images"].append(image_info)

        # annotation
        mask = img[:, :, -1]
        bin_mask = np.where(mask > 128, 1, 0)
        class_id = [x['id'] for x in CATEGORIES if x['name'] in imgpath][0]
        category_info = {'id': class_id, 'is_crowd': 'crowd' in filename}
        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id,
            image_id,
            category_info,
            bin_mask,
            getSize(imgpath),
            tolerance=2
        ) 
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
        segmentation_id += 1
    
    return coco_output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pinctada dataset convert to COCO format')
    parser.add_argument(
        '--dataroot', '-d', type=str, help='dataset root directory'
    )
    parser.add_argument(
        '--split', type=str, help='choose from train/val/test'
    )
    args = parser.parse_args()
    
    # dataroot = "/ldap_home/jincheng.lyu/data/product_segmentation/synthetics/train"
    dataroot = osp.join(args.dataroot, args.split)
    coco_output = genAnns(dataroot)
    
    with open(f"instances_{args.split}.json", 'w') as f:
        json.dump(coco_output, f, indent=4)
