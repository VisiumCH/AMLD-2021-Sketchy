import os
from glob import glob
import shutil
import cv2 as cv
import matplotlib.pyplot as plt
from src.data.watch_dataset.hed import HED


# GLOBAL VARIABLE DECLARATION
INPUT_DIR = 'io/data/raw/'
OUTPUT_DIR = 'io/data/processed'
HED_MODEL_PATH = 'io/models/watch_pretrained_models/hed'


def make_sketches(args):


if __name__ == '__main__':
    from src.options import Options

    # Parse options
    args = Options().parse()
    print("Parameters:\t" + str(args))

    make_sketches(args)
