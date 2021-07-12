import argparse
import os
import shutil
import utils_bg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fg_type",
        "-fg_type",
        default="frame",
        choices=["frame", "logo"],
        type=str,
        help="frame or logo",
    )
    parser.add_argument(
        "--fg_pool",
        "-fg_pool",
        default="/ldap_home/kailin.chen/product_segmentation/datasets/synthetic/logo_pool_sp/val/",
        type=str,
        help="frame pool or logo pool dir.",
    )
    parser.add_argument(
        "--save_bg_dir",
        "-save_bg_dir",
        default="/ldap_home/kailin.chen/product_segmentation/datasets/synthetic/bg_pool/bg_logo_text/val/",
        type=str,
        help="new image without alpha channel save dir.",
    )
    parser.add_argument(
        "--save_fg_dir",
        "-save_fg_dir",
        default="/ldap_home/kailin.chen/product_segmentation/datasets/synthetic/logo_pool_sp/val_aug_text/",
        type=str,
        help="new image with alpha channel save dir.",
    )

    args = parser.parse_args()

    if os.path.isdir(args.save_bg_dir):  # clear the save folder
        shutil.rmtree(args.save_bg_dir)
    os.mkdir(args.save_bg_dir)
    if os.path.isdir(args.save_fg_dir):
        shutil.rmtree(args.save_fg_dir)
    os.mkdir(args.save_fg_dir)

    syner = utils_bg.Synthesizer(
        fg_pool=args.fg_pool, save_bg_dir=args.save_bg_dir, savedir=args.save_fg_dir
    )
    if args.fg_type == "frame":
        syner.gen_frame()  # for frame
    elif args.fg_type == "logo":
        syner.gen_logo_text()  # for logo and text
