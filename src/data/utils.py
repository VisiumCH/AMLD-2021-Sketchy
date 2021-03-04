from glob import glob
import os
import random

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
        #print(os.path.join(dir_skim, cls, _ext))
        path_file = glob(os.path.join(dir_skim, cls, _ext))
        # print(path_file)
        fnames += [os.path.basename(x) for x in path_file]
        fnames_cls += [cls] * len(path_file)
    return fnames, fnames_cls


def default_image_loader(path):

    img = Image.fromarray(cv2.resize(np.array(Image.open(path).convert("RGB")), (224, 224)))
    return img


def default_image_loader_tuberlin(path):

    img = Image.fromarray(cv2.resize(np.array(Image.open(path).convert('RGB')), (224, 224)))
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    return img


def get_random_file_from_path(file_path):
    _ext = "*.jpg"
    f_list = glob(os.path.join(file_path, _ext))
    return np.random.choice(f_list, 1)[0]


def dataset_split(args, dataset_folder="Sketchy", image_folder="extended_photo", name='sketchy'):
    # Random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Getting the classes
    class_directories = glob(os.path.join(args.data_path, dataset_folder, image_folder, "*/"))
    list_class = [class_path.split("/")[-2] for class_path in class_directories]
    dicts_class = create_dict_texts(list_class)

    # Read test classes
    with open(os.path.join(args.data_path, dataset_folder, "zeroshot_classes_" + name + ".txt")) as fp:
        test_class = fp.read().splitlines()
    list_class = [x for x in list_class if x not in test_class]

    # Random Shuffle
    shuffled_list_class = list_class
    random.shuffle(shuffled_list_class)

    # Dividing the classes
    train_class = shuffled_list_class[: int(0.9 * len(shuffled_list_class))]
    valid_class = shuffled_list_class[int(0.9 * len(shuffled_list_class)):]

    # Save split
    with open(os.path.join(args.save, name + '_train.txt'), 'w') as fp:
        for item in train_class:
            fp.write("%s\n" % item)
    with open(os.path.join(args.save, name + '_valid.txt'), 'w') as fp:
        for item in valid_class:
            fp.write("%s\n" % item)
    with open(os.path.join(args.save, name + '_test.txt'), 'w') as fp:
        for item in test_class:
            fp.write("%s\n" % item)

    return train_class, valid_class, test_class, dicts_class
