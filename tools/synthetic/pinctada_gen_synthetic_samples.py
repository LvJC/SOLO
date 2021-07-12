import glob
import os
import os.path as osp
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
    fg_pool_folders = glob.glob(args.fg_pool + "*")
    bg_pool_images = glob.glob(args.bg_pool + "*.png")
    print("Number of fg pool classes:", len(fg_pool_folders))

    for fg_class_folder in fg_pool_folders:
        fg_class = osp.basename(fg_class_folder)
        split_images = {split: [] for split in ["train", "val", "test"]}

        # Set random seed
        random.seed(101)

        # generate list of fg_pool_images
        images = glob.glob(fg_class_folder + "/*")
        random.shuffle(images)

        k = int(0.2 * len(images))
        test_set = images[:k]
        val_set = images[k : 2 * k]
        train_set = images[2 * k :]

        split_images["train"].extend(train_set)
        split_images["test"].extend(test_set)
        split_images["val"].extend(val_set)

        test_split_ratio = 0.1
        val_split_ratio = 0.1
        k_bg_pool_test = test_split_ratio * len(bg_pool_images)
        k_bg_pool_val = val_split_ratio * len(bg_pool_images)

        # random.shuffle(fg_pool_images)
        random.shuffle(bg_pool_images)

        print("Number of synthetic train images:\t", len(split_images["train"]))
        print("Number of synthetic val images:\t", len(split_images["test"]))
        print("Number of synthetic test images:\t", len(split_images["val"]))

        bg_pool_test = bg_pool_images[: int(k_bg_pool_test)]
        bg_pool_val = bg_pool_images[
            int(k_bg_pool_test) : int(k_bg_pool_test + k_bg_pool_val)
        ]
        bg_pool_train = bg_pool_images[int(k_bg_pool_test + k_bg_pool_val) :]

        # print(bg_pool_test)
        for split in ["train", "val", "test"]:
            os.makedirs(osp.join(args.output, split, fg_class), exist_ok=True)
            transfer_files(split_images[split], osp.join(args.output, split, fg_class))

        # copyfile(src, dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_pool", "-fg_pool", type=str, help="foreground pool dir.")
    parser.add_argument("--bg_pool", "-bg_pool", type=str, help="background pool dir.")
    parser.add_argument(
        "--output", "-output", type=str, help="output synthetic pool dir."
    )
    args = parser.parse_args()
    main(args)
