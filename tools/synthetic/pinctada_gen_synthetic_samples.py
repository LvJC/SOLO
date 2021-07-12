import random
from shutil import copy2
from tqdm import tqdm


def transfer_files(fg_pool_list, save_loc):
    # fg_pool_save = save_loc+'fg_pool/'
    fg_pool_save = save_loc  # modified by Kailin
    # bg_pool_save = save_loc+'bg_pool/'

    # for myfile in bg_pool_list:
    #     # filename = myfile.split('/')[-1]
    #     copy2(myfile,bg_pool_save)

    for myfile in tqdm(fg_pool_list):
        copy2(myfile, fg_pool_save)


def main(args):
    # fg abd bg pool path
    # fg_pool_path= '/ldap_home/kailin.chen/product_segmentation/datasets/synthetic/logo_pool/'
    # #bg_pool_path= '/ldap_home/regds_cvcore/product_segmentation/datasets/synthetic/bg_pool/simple_bg/'
    # bg_pool_path= '/ldap_home/kailin.chen/product_segmentation/datasets/synthetic/bg_pool2/bg_synthetic/'

    fg_pool_path = "/ldap_home/jincheng.lyu/project/SOLO/data/fg_pool/"
    bg_pool_path = "/ldap_home/jincheng.lyu/project/SOLO/data/bg_pool/"

    import glob

    fg_pool_folders = glob.glob(fg_pool_path + "*")
    bg_pool_images = glob.glob(bg_pool_path + "*.png")
    print("Number of fg pool folders:", len(fg_pool_folders))

    train_images = []
    test_images = []
    val_images = []

    # Set random seed
    random.seed(101)

    # generate list of fg_pool_images
    fg_pool_images = []
    for folder in fg_pool_folders:
        images = glob.glob(folder + "/*")
        random.shuffle(images)
        k = int(0.2 * len(images))
        test_set = images[:k]
        val_set = images[k : 2 * k]
        train_set = images[2 * k :]

        train_images.extend(train_set)
        test_images.extend(test_set)
        val_images.extend(val_set)
        # fg_pool_images.extend(glob.glob(folder+'/*'))

    # print(fg_pool_images)
    # print(len(fg_pool_images))
    # print(len(bg_pool_images))

    test_split_ratio = 0.1
    val_split_ratio = 0.1
    # k_fg_pool_test = test_split_ratio*len(fg_pool_images)
    k_bg_pool_test = test_split_ratio * len(bg_pool_images)
    # k_fg_pool_val = val_split_ratio*len(fg_pool_images)
    k_bg_pool_val = val_split_ratio * len(bg_pool_images)

    # random.shuffle(fg_pool_images)
    random.shuffle(bg_pool_images)

    print(len(train_images))
    print(len(test_images))
    print(len(val_images))
    # fg_pool_test = fg_pool_images[:int(k_fg_pool_test)]
    # fg_pool_val = fg_pool_images[int(k_fg_pool_test):int(k_fg_pool_test+k_fg_pool_val)]
    # fg_pool_train = fg_pool_images[int(k_fg_pool_test+k_fg_pool_val):]

    bg_pool_test = bg_pool_images[: int(k_bg_pool_test)]
    bg_pool_val = bg_pool_images[
        int(k_bg_pool_test) : int(k_bg_pool_test + k_bg_pool_val)
    ]
    bg_pool_train = bg_pool_images[int(k_bg_pool_test + k_bg_pool_val) :]

    # print(bg_pool_test)


if __name__ == "__main__":
    import argparse
    import os.path as osp

    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_pool", "-val_pool", type=str, help="foreground pool dir.")
    parser.add_argument(
        "--bg_pool", "-test_pool", type=str, help="background pool dir."
    )
    parser.add_argument(
        "--output", "-train_pool", type=str, help="output synthetic pool dir."
    )
    args = parser.parse_args()

    transfer_files(train_images, osp.join(args.output, "train"))
    transfer_files(test_images, osp.join(args.output, "test"))
    transfer_files(val_images, osp.join(args.output, "val"))

    # copyfile(src, dst)
