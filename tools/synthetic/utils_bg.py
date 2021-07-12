import copy
import cv2
import hashlib
import numpy as np
import os
import random
import string
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


def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    """
    randomly transform image in HSV space.
    :param img: RGBA image.
    :param hue_vari: H channel variance range.
    :param sat_vari: S channel variance range.
    :param val_vari: V channel variance range.
    :return RGBA image.
    """
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)

    if img.shape[2] != 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img[:, :, 3] = 0

    img_RGB = img[:, :, :3]
    img_hsv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    img[:, :, :3] = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img


def random_string(slen=10):
    return "".join(
        random.sample(string.ascii_letters + string.digits + "!@#$%^&*()_+=-", slen)
    )


def random_add_text(new_canvas: np.ndarray):
    """
    :param new_canvas: RGBA image.
    :return RGBA image.
    """
    # font
    font_list = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_ITALIC,
    ]
    font = font_list[random.randint(0, len(font_list) - 1)]
    font_string = random_string(slen=random.randint(4, 7))
    font_size = random.uniform(1, 2)
    font_thickness = random.randint(1, 2)

    # position
    p1 = random.randint(120, 200)
    p21 = random.randint(50, 100)
    p22 = random.randint(420, 450)
    if random.uniform(0, 1) < 0.5:
        p2 = p21
    else:
        p2 = p22

    # color
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)

    new_canvas = cv2.putText(
        new_canvas,
        font_string,
        (p1, p2),
        font,
        font_size,
        (c1, c2, c3),
        font_thickness,
        bottomLeftOrigin=False,
    )
    return new_canvas


def syn_img_logo(
    fg_0: np.ndarray, fg_1: np.ndarray, fg_2: np.ndarray, fg_3: np.ndarray
):
    """
    :param fg_0: RGBA image.
    :return list of RGBA image.
    """
    assert tbi_checking(fg_0)
    bg = np.zeros((500, 500, 3), dtype=np.uint8)
    bg[:, :, :] = 255

    fg_list = [fg_0, fg_1, fg_2, fg_3]  # for big logos
    for i in range(4):
        fg = fg_list[i]
        r = 1
        if max(fg.shape[0], fg.shape[1]) > 0.4 * bg.shape[0]:
            r = 100 / max(fg.shape[0], fg.shape[1])
        if min(fg.shape[0], fg.shape[1]) > 0.2 * bg.shape[0]:
            r = min(100 / min(fg.shape[0], fg.shape[1]), r)
        if r != 1:
            fg_list[i] = cv2.resize(fg, (int(fg.shape[1] * r), int(fg.shape[0] * r)))
        if len(fg.shape) < 3:
            fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB)
        if fg.shape[2] != 4:
            fg = cv2.cvtColor(fg, cv2.COLOR_RGB2RGBA)
            fg[:, :, 3] = 0
            fg_list[i] = fg
    fg_0, fg_1, fg_2, fg_3 = fg_list

    # randomly transform image in HSV space.
    fg_1 = random_hsv_transform(fg_1, 50, 0.2, 0.2)
    fg_2 = random_hsv_transform(fg_2, 50, 0.2, 0.2)
    fg_3 = random_hsv_transform(fg_3, 50, 0.2, 0.2)

    fg_right = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
    fg_right[: fg_1.shape[0], bg.shape[1] - fg_1.shape[1] :, :] = fg_1
    fg_right[bg.shape[0] - fg_3.shape[0] :, bg.shape[1] - fg_3.shape[1] :, :] = fg_3

    fg_all = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
    fg_all[: fg_0.shape[0], : fg_0.shape[1], :] = fg_0
    fg_cv_0 = copy.deepcopy(fg_all)
    fg_all[bg.shape[0] - fg_2.shape[0] :, : fg_2.shape[1], :] = fg_2
    mask_left = fg_all[:, :, 3]
    judge_mat = mask_left == 0
    fg_right[:, :, :] = np.multiply(
        fg_right[:, :, :],
        np.broadcast_to(judge_mat[..., None], fg_right[:, :, :].shape),
    )
    fg_all[:, :, :] = np.multiply(
        fg_all[:, :, :], np.broadcast_to(~judge_mat[..., None], fg_right[:, :, :].shape)
    )
    fg_all[:, :, :] = fg_right[:, :, :] + fg_all[:, :, :]

    fg_cv_1 = np.rot90(fg_cv_0)
    fg_cv_2 = np.rot90(fg_cv_0, 2)
    fg_cv_3 = np.rot90(fg_cv_0, 3)

    fg_list = [fg_cv_0, fg_cv_1, fg_cv_2, fg_cv_3, fg_all]  # for small logos

    canvas_list = []
    for i in range(len(fg_list)):
        bg = np.zeros((500, 500, 3), dtype=np.uint8)
        bg[:, :, :] = 255
        fg = fg_list[i]
        mask = fg[:, :, 3]
        new_canvas = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
        judge_mat = mask == 0
        bg[:, :, 0:3] = np.multiply(
            bg[:, :, 0:3], np.broadcast_to(judge_mat[..., None], bg[:, :, 0:3].shape)
        )
        fg[:, :, 0:3] = np.multiply(
            fg[:, :, 0:3], np.broadcast_to(~judge_mat[..., None], bg[:, :, 0:3].shape)
        )
        new_canvas[:, :, 0:3] = bg[:, :, 0:3] + fg[:, :, 0:3]
        new_canvas[:, :, 3] = mask

        # Add Text Randomly
        if np.random.uniform(0, 1) < 0.5:
            new_canvas = random_add_text(new_canvas)

        canvas_list.append(new_canvas)

    return canvas_list


def syn_frame(fg_0: np.ndarray):  # for generate frames only
    """
    :param fg_0: RGBA image.
    :return: list of RGBA image
    """
    assert tbi_checking(fg_0)
    bg = np.zeros((500, 500, 3), dtype=np.uint8)
    bg[:, :, :] = 255
    # fg_0 = resize_tbi(fg_0, bg.shape[1], bg.shape[0], keep_ratio=True)  # for small decorator
    fg_0 = cv2.resize(fg_0, (bg.shape[1], bg.shape[0]))  # for big frames

    fg_1 = np.rot90(fg_0)
    fg_2 = np.rot90(fg_0, 2)
    fg_3 = np.rot90(fg_0, 3)
    fg_ud = fg_0[:, :, :] + fg_2[:, :, :]
    mask_ud = fg_ud[:, :, 3]
    fg_lr = fg_1[:, :, :] + fg_3[:, :, :]
    fg_all = copy.deepcopy(fg_ud)
    judge_mat = mask_ud == 0
    fg_lr[:, :, :] = np.multiply(
        fg_lr[:, :, :], np.broadcast_to(judge_mat[..., None], fg_lr[:, :, :].shape)
    )
    fg_all[:, :, :] = np.multiply(
        fg_all[:, :, :], np.broadcast_to(~judge_mat[..., None], fg_lr[:, :, :].shape)
    )
    fg_all[:, :, :] = fg_lr[:, :, :] + fg_all[:, :, :]
    fg_list = [fg_0, fg_1, fg_2, fg_3, fg_ud, fg_all]  # for big frames

    canvas_list = []
    for i in range(len(fg_list)):
        bg = np.zeros((500, 500, 3), dtype=np.uint8)
        bg[:, :, :] = 255
        fg = fg_list[i]
        mask = fg[:, :, 3]
        new_canvas = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
        judge_mat = mask == 0
        bg[:, :, 0:3] = np.multiply(
            bg[:, :, 0:3], np.broadcast_to(judge_mat[..., None], bg[:, :, 0:3].shape)
        )
        fg[:, :, 0:3] = np.multiply(
            fg[:, :, 0:3], np.broadcast_to(~judge_mat[..., None], bg[:, :, 0:3].shape)
        )
        new_canvas[:, :, 0:3] = bg[:, :, 0:3] + fg[:, :, 0:3]
        new_canvas[:, :, 3] = mask
        canvas_list.append(new_canvas)

    return canvas_list


class Synthesizer(object):
    def __init__(self, fg_pool, save_bg_dir, savedir):
        self.fg_pool = [
            os.path.join(fg_pool, x)
            for x in os.listdir(fg_pool)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.savedir = savedir
        self.save_bg_dir = save_bg_dir
        print("fg pool has: {} images".format(len(self.fg_pool)))

    def __len__(self):
        return len(self.fg_pool)

    def gen_frame(self, num=None):

        if num is None:
            fg_pool = self.fg_pool
        else:
            fg_pool = random.sample(self.fg_pool, num)

        for img in tqdm(fg_pool):
            fg_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            try:
                print("Trying to generate new img")
                new_imgs = syn_frame(fg_0=fg_img)
            except AssertionError:
                print("ERROR!")
                continue
            for new_img in new_imgs:
                md5_hash = hashlib.md5(new_img.tobytes()).hexdigest()
                cv2.imwrite(
                    os.path.join(self.savedir, md5_hash + ".png"), new_img
                )  # with alpha channel
                new_bg = new_img[:, :, 0:3]
                cv2.imwrite(
                    os.path.join(self.save_bg_dir, md5_hash + ".png"), new_bg
                )  # without alpha channel

    def gen_logo_text(self, num=None):

        if num is None:
            fg_pool = self.fg_pool
        else:
            fg_pool = random.sample(self.fg_pool, num)

        for img in tqdm(fg_pool):
            fg_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            aux_list = random.sample(fg_pool, 3)
            aux1 = cv2.imread(aux_list[0], cv2.IMREAD_UNCHANGED)
            aux2 = cv2.imread(aux_list[1], cv2.IMREAD_UNCHANGED)
            aux3 = cv2.imread(aux_list[2], cv2.IMREAD_UNCHANGED)
            try:
                print("Trying to generate new img")
                new_imgs = syn_img_logo(fg_0=fg_img, fg_1=aux1, fg_2=aux2, fg_3=aux3)
            except AssertionError:
                print("ERROR!")
                continue
            for new_img in new_imgs:
                md5_hash = hashlib.md5(new_img.tobytes()).hexdigest()
                cv2.imwrite(
                    os.path.join(self.savedir, md5_hash + ".png"), new_img
                )  # with alpha channel
                new_bg = new_img[:, :, 0:3]
                cv2.imwrite(
                    os.path.join(self.save_bg_dir, md5_hash + ".png"), new_bg
                )  # without alpha channel
