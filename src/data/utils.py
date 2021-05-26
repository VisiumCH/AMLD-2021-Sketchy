from glob import glob
import os
import random

import cv2
import numpy as np
from PIL import Image


def create_dict_texts(texts):
    """ Dictionnary with key: number and value: class names"""
    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def get_file_list(dir_skim, class_list, skim):
    """ Lists of image/sketch files of a directory """
    _ext = "*.png" if skim == "sketches" else "*.jpg"

    fnames = []
    fnames_cls = []
    for cls in class_list:
        path_file = glob(os.path.join(dir_skim, cls, _ext))
        fnames += [x for x in path_file]
        fnames_cls += [cls] * len(path_file)
    return fnames, fnames_cls


def default_image_loader(path):
    """ Loads RGB data """
    img = Image.fromarray(
        cv2.resize(np.array(Image.open(path).convert("RGB")), (224, 224))
    )
    return img


def default_image_loader_tuberlin(path):
    """ Loads BGR data for TU-Berlin dataset """
    img = Image.fromarray(
        cv2.resize(np.array(Image.open(path).convert("RGB")), (224, 224))
    )
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    return img


def get_random_file_from_path(file_path):
    _ext = "*.jpg"
    f_list = glob(os.path.join(file_path, _ext))
    return np.random.choice(f_list, 1)[0]


def get_class_dict(args, dataset_folder):
    """ Get a dictionnary of the class based on the folders' classes """
    class_directories = glob(os.path.join(args.data_path, dataset_folder, "images/*/"))
    list_class = [class_path.split("/")[-2] for class_path in class_directories]
    dicts_class = create_dict_texts(list_class)
    return dicts_class


def dataset_class_split(
    args, dataset_folder, training_split, validation_split, image_type
):
    """
    Splits the data of a class between test, valid and train split
    Args:
        - args: metadata provided in src/options.py
        - dataset_folder: name of the folder containing the data
        - training_split: proportion of data in the training set
        - validation_split: proportion of data in the validation set
        - image_type: sketch or image
    Return:
        - fnames_train: path to the images in the training set
        - cls_train: classes associated to the images in the training set
        - fnames_valid: path to the images in the validation set
        - cls_valid: classes associated to the images in the validaiton set
        - fnames_test: path to the images in the test set
        - cls_test: classes associated to the images in the test set
    """
    # Random seed
    random.seed(args.seed)

    _ext = ".png" if image_type == "sketches" else ".jpg"

    fnames_train, fnames_valid, fnames_test = [], [], []
    cls_train, cls_valid, cls_test = [], [], []

    class_directories = glob(
        os.path.join(args.data_path, dataset_folder, image_type, "*/")
    )

    for class_dir in class_directories:
        cls_name = class_dir.split("/")[-2]
        images_path = glob(class_dir + "/*")
        images_path = [path for path in images_path if path.endswith(_ext)]
        random.shuffle(images_path)

        train_split = int(training_split * len(images_path))
        valid_split = int((training_split + validation_split) * len(images_path))

        train_images = images_path[:train_split]
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
    """
    Splits the data of all classes between test, valid and train split
    Args:
        - args: metadata provided in src/options.py
        - dataset_folder: name of the folder containing the data
        - training_split: proportion of data in the training set
        - validation_split: proportion of data in the validation set
    Return:
        - train_data: [images path, image classes, sketch paths, sketch classes] of the training set
        - valid_data: [images path, image classes, sketch paths, sketch classes] of the validation set
        - test_data: [images path, image classes, sketch paths, sketch classes] of the testing set
    """
    (
        fnames_image_train,
        cls_image_train,
        fnames_image_valid,
        cls_image_valid,
        fnames_image_test,
        cls_image_test,
    ) = dataset_class_split(args, dataset_folder, training_split, valid_split, "images")

    (
        fnames_sketch_train,
        cls_sketch_train,
        fnames_sketch_valid,
        cls_sketch_valid,
        fnames_sketch_test,
        cls_sketch_test,
    ) = dataset_class_split(
        args, dataset_folder, training_split, valid_split, "sketches"
    )

    train_data = [
        fnames_image_train,
        cls_image_train,
        fnames_sketch_train,
        cls_sketch_train,
    ]
    valid_data = [
        fnames_image_valid,
        cls_image_valid,
        fnames_sketch_valid,
        cls_sketch_valid,
    ]
    test_data = [fnames_image_test, cls_image_test, fnames_sketch_test, cls_sketch_test]

    return train_data, valid_data, test_data


def get_loader(dataset):
    if dataset == "TU-Berlin":
        loader = default_image_loader_tuberlin
    else:
        loader = default_image_loader
    return loader


def get_dict(dataset, dict_class):
    """
    Returns the appropriate dictionnary based on the training training folder
    """
    if isinstance(dict_class, dict):
        dict_class = dict_class
    else:
        if dataset == "Sketchy":
            dict_class = dict_class[0]
        elif dataset == "TU-Berlin":
            dict_class = dict_class[1]
        elif dataset == "Quickdraw":
            dict_class = dict_class[2]
        else:
            raise (f"Error with dataset name: {dataset}.")
    return dict_class


def get_limits(dataset, valid_data, image_type):
    """
    Returns the limit of index at which to switch between datasets
    Args:
        - dataset: dataset in training
        - valid_data: validation data loader
        - image_type: sketch or image
    Return:
        In the case of training a single datasets, it is always (None, None)
        In the case of training multiple datasets,
            - sketchy_limit is the limit between the indexes of Sketchy dataset and TU_Berlin dataset
            - tuberlin_limit is the limit between the indexes of TU_Berlin dataset and Quickdraw dataset

    """
    if dataset == "sk+tu" or dataset == "sk+tu+qd":
        if image_type == "images":
            sketchy_limit = valid_data.sketchy_limit_images
        else:
            sketchy_limit = valid_data.sketchy_limit_sketch
    else:
        sketchy_limit = None

    if dataset == "sk+tu+qd":
        if image_type == "images":
            tuberlin_limit = valid_data.tuberlin_limit_images
        else:
            tuberlin_limit = valid_data.tuberlin_limit_sketch
    else:
        tuberlin_limit = None

    return sketchy_limit, tuberlin_limit


def get_dataset_dict(dict_class, idx, sketchy_limit, tuberlin_limit):
    """
    Based on the index and the indexes limits (sketchy_limit, tuberlin_limit),
    returns the dictionnary associated to the right dataset.
    """
    if sketchy_limit is None:  # single dataset
        pass
    else:  # multiple datasets
        if idx < sketchy_limit:  # sketchy dataset
            dict_class = dict_class[0]
        else:
            if tuberlin_limit is None or idx < tuberlin_limit:  # tuberlin dataset
                dict_class = dict_class[1]
            else:  # quickdraw dataset
                dict_class = dict_class[2]

    return dict_class
