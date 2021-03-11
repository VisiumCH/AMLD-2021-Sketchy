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
    _ext = "*.png" if skim == "sketches" else "*.jpg"

    fnames = []
    fnames_cls = []
    for cls in class_list:
        path_file = glob(os.path.join(dir_skim, cls, _ext))
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


def get_class_dict(args, dataset_folder):
    # Getting the classes
    class_directories = glob(os.path.join(args.data_path, dataset_folder, "images/*/"))
    list_class = [class_path.split("/")[-2] for class_path in class_directories]
    dicts_class = create_dict_texts(list_class)
    return dicts_class


def dataset_class_split(args, dataset_folder, training_split, validation_split, image_type):
    # Random seed
    random.seed(args.seed)

    _ext = ".png" if image_type == "sketches" else ".jpg"

    fnames_train, fnames_valid, fnames_test = [], [], []
    cls_train, cls_valid, cls_test = [], [], []

    class_directories = glob(os.path.join(args.data_path, dataset_folder, image_type, "*/"))

    for class_dir in class_directories:
        cls_name = class_dir.split('/')[-2]
        images_path = glob(class_dir + '/*')
        images_path = [path for path in images_path if path.endswith(_ext)]
        random.shuffle(images_path)

        train_split = int(training_split * len(images_path))
        valid_split = int((training_split + validation_split) * len(images_path))

        train_images = images_path[: train_split]
        valid_images = images_path[train_split:valid_split]
        test_images = images_path[valid_split:]

        fnames_train += [os.path.basename(x) for x in train_images]
        cls_train += [cls_name] * len(train_images)

        fnames_valid += [os.path.basename(x) for x in valid_images]
        cls_valid += [cls_name] * len(valid_images)

        fnames_test += [os.path.basename(x) for x in test_images]
        cls_test += [cls_name] * len(test_images)

    return fnames_train, cls_train, fnames_valid, cls_valid, fnames_test, cls_test


def dataset_split(args, dataset_folder, training_split, valid_split):
    dicts_class = get_class_dict(args, dataset_folder)

    (
        fnames_image_train, cls_image_train,
        fnames_image_valid, cls_image_valid,
        fnames_image_test, cls_image_test
    ) = dataset_class_split(args, dataset_folder, training_split, valid_split, "images")

    (
        fnames_sketch_train, cls_sketch_train,
        fnames_sketch_valid, cls_sketch_valid,
        fnames_sketch_test, cls_sketch_test
    ) = dataset_class_split(args, dataset_folder, training_split, valid_split, "sketches")

    train_data = [fnames_image_train, cls_image_train, fnames_sketch_train, cls_sketch_train]
    valid_data = [fnames_image_valid, cls_image_valid, fnames_sketch_valid, cls_sketch_valid]
    test_data = [fnames_image_test, cls_image_test, fnames_sketch_test, cls_sketch_test]

    return dicts_class, train_data, valid_data, test_data


def get_loader(dataset):
    if dataset == 'TU-Berlin':
        loader = default_image_loader_tuberlin
    else:
        loader = default_image_loader
    return loader


def get_dict(dataset, dict_class):
    if isinstance(dict_class, dict):
        dict_class = dict_class
    else:
        if dataset == 'Sketchy':
            dict_class = dict_class[0]
        elif dataset == 'TU-Berlin':
            dict_class = dict_class[1]
        elif dataset == 'Quickdraw':
            dict_class = dict_class[2]
        else:
            raise(f"Error with dataset name: {dataset}.")
    return dict_class
