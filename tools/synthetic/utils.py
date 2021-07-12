import cv2
import hashlib
import numpy as np
import os
import random
from tqdm import tqdm


def tbi_checking(img: np.ndarray) -> bool:
    """
    check whether input image is tbi.
    :param img: input tbi image.
    :return bool.
    """
    flag = len(img.shape) == 3 and img.shape[-1] == 4
    return flag and (img[:, :, 3] == 0).any()


def resize_tbi(
    img: np.ndarray, width: int, height: int, keep_ratio: bool
) -> np.ndarray:
    """
    resize tbi image.
    :param img: tbi image.
    :param width: new width.
    :param height: new height.
    :param keep_ratio: bool, whether keep ratio.
    """
    if not keep_ratio:
        return cv2.resize(img, (width, height))
    else:
        img_h, img_w = img.shape[0:2]
        r = min(height / img_h, width / img_w)
        r = min(r, 1.0)
        unpad_h, unpad_w = int(round(img_h * r)), int(round(img_w * r))
        dh, dw = height - unpad_h, width - unpad_w
        dh /= 2
        dw /= 2
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        new_cavas = np.zeros((height, width, 4), dtype=np.uint8)
        new_cavas[top : top + unpad_h, left : left + unpad_w, :] = img
        return new_cavas


def syn_img(fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """

    :param fg: RGBA image.
    :param bg: RGB image.
    :return: RGBA image
    """
    assert tbi_checking(fg)
    fg = resize_tbi(fg, bg.shape[1], bg.shape[0], keep_ratio=True)
    mask = fg[:, :, 3]
    new_cavas = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
    judge_mat = mask == 0
    bg[:, :, 0:3] = np.multiply(
        bg[:, :, 0:3], np.broadcast_to(judge_mat[..., None], bg[:, :, 0:3].shape)
    )
    fg[:, :, 0:3] = np.multiply(
        fg[:, :, 0:3], np.broadcast_to(~judge_mat[..., None], bg[:, :, 0:3].shape)
    )
    new_cavas[:, :, 0:3] = bg[:, :, 0:3] + fg[:, :, 0:3]
    new_cavas[:, :, 3] = mask
    return new_cavas


class Synthesizer(object):
    def __init__(self, bg_pool, fg_pool, savedir):
        self.bg_pool = [
            os.path.join(bg_pool, x)
            for x in os.listdir(bg_pool)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.fg_pool = [
            os.path.join(fg_pool, x)
            for x in os.listdir(fg_pool)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.savedir = savedir
        print("bg pool has: {} images".format(len(self.bg_pool)))

    def __len__(self):
        return len(self.fg_pool)

    def gen(self, num=None):
        if num is None:
            fg_pool = self.fg_pool
        else:
            fg_pool = random.sample(self.fg_pool, num)
        for img in tqdm(fg_pool):
            bg_img = random.choice(self.bg_pool)
            print("Randomly Selected bg_img is: ")
            print(bg_img)
            fg_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            bg_img = cv2.imread(bg_img, cv2.IMREAD_UNCHANGED)
            try:
                print("Trying to generate new img")
                new_img = syn_img(fg=fg_img, bg=bg_img)
            except AssertionError:
                print("ERROR!")
                continue
            md5_hash = hashlib.md5(new_img.tobytes()).hexdigest()
            cv2.imwrite(os.path.join(self.savedir, md5_hash + ".png"), new_img)
