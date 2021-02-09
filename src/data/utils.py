from glob import glob
import os

import cv2
import numpy as np
from PIL import Image


def create_dict_texts(texts):

    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def get_file_list(dir_skim, class_list, skim="sketch"):
    if skim == "sketch":
        _ext = "*.png"
    elif skim == "images":
        _ext = "*.jpg"
    else:
        NameError(skim + " not implemented!")

    fnames = []
    fnames_cls = []
    for cls in class_list:
        path_file = glob(os.path.join(dir_skim, cls, _ext))
        fnames += [os.path.basename(x) for x in path_file]
        fnames_cls += [cls] * len(path_file)
    return fnames, fnames_cls


def default_image_loader(path):

    img = Image.fromarray(
        cv2.resize(np.array(Image.open(path).convert("RGB")), (224, 224))
    )
    return img


def get_random_file_from_path(file_path):
    _ext = "*.jpg"
    f_list = glob(os.path.join(file_path, _ext))
    return np.random.choice(f_list, 1)[0]
