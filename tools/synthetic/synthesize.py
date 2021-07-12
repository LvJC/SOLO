import argparse
import glob
import os
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_pool", "-bg_pool", type=str, help="background pool dir.")
    parser.add_argument("--fg_pool", "-fg_pool", type=str, help="foreground pool dir.")
    parser.add_argument("--savedir", "-savedir", type=str, help="new image save dir.")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        bg_pool = os.path.join(args.bg_pool, split)
        fg_pool_dir = os.path.join(args.fg_pool, split)
        savedir = os.path.join(args.savedir, split)
        os.makedirs(savedir, exist_ok=True)
        print(bg_pool)
        for fg_pool in glob.glob(fg_pool_dir + '/*'):
            fg_class = os.path.basename(fg_pool)
            class_savedir = os.path.join(savedir, fg_class)
            syner = utils.Synthesizer(bg_pool=bg_pool, fg_pool=fg_pool, savedir=class_savedir)
            syner.gen(bg_num=4)
