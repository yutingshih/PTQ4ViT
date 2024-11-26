import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=6)
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--dataset_root", type=str, default="/datasets/imagenet")
    args = parser.parse_args()
    return args
