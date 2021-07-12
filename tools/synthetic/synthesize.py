import argparse
import os
from synthetic import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_pool", "-bg_pool", type=str, help="background pool dir.")
    parser.add_argument("--fg_pool", "-fg_pool", type=str, help="foreground pool dir.")
    parser.add_argument("--savedir", "-savedir", type=str, help="new image save dir.")
    args = parser.parse_args()
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
    print(args.bg_pool)
    syner = utils.Synthesizer(
        bg_pool=args.bg_pool, fg_pool=args.fg_pool, savedir=args.savedir
    )
    syner.gen()
