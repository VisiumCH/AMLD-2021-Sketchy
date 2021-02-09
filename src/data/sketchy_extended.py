#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

import os
import pickle
import random
import sys
from glob import glob

import numpy as np
from scipy.spatial.distance import cdist

from utils import *

dataset_folder = "Sketchy"


def Sketchy_Extended(args, transform="None"):

    # Getting the classes
    class_directories = glob(
        os.path.join(args.data_path, dataset_folder, "extended_photo/*/")
    )
    list_class = [class_path.split("/")[-2] for class_path in class_directories]
    dicts_class = create_dict_texts(list_class)

    # Random seed
    np.random.seed(args.seed)

    # Read test classes
    with open(
        os.path.join(args.data_path, dataset_folder, "zeroshot_classes_sketchy.txt")
    ) as fp:  # zeroshot_classes.txt
        test_class = fp.read().splitlines()

    list_class = [x for x in list_class if x not in test_class]
    # Random Shuffle
    random.seed(args.seed)
    shuffled_list_class = list_class
    random.shuffle(shuffled_list_class)

    # Dividing the classes
    train_class = shuffled_list_class[: int(0.9 * len(shuffled_list_class))]
    valid_class = shuffled_list_class[int(0.9 * len(shuffled_list_class)) :]

    # if args.exp_idf is not None:
    #     if args.save is None:
    #         args.save = os.path.join("./checkpoint", args.exp_idf)
    #     with open(os.path.join(args.save, "train.txt"), "w") as fp:
    #         for item in train_class:
    #             fp.write("%s\n" % item)
    #     with open(os.path.join(args.save, "valid.txt"), "w") as fp:
    #         for item in valid_class:
    #             fp.write("%s\n" % item)

    #     if args.plot is False:
    #         with open(os.path.join(args.save, "valid.txt"), "r") as fp:
    #             valid_class = fp.read().splitlines()

    # Data Loaders
    train_loader = Sketchy_Extended_train(args, train_class, dicts_class, transform)
    valid_sk_loader = Sketchy_Extended_valid_test(
        args, valid_class, dicts_class, transform, type_skim="sketch"
    )
    valid_im_loader = Sketchy_Extended_valid_test(
        args, valid_class, dicts_class, transform, type_skim="images"
    )
    test_sk_loader = Sketchy_Extended_valid_test(
        args, test_class, dicts_class, transform, type_skim="sketch"
    )
    test_im_loader = Sketchy_Extended_valid_test(
        args, test_class, dicts_class, transform, type_skim="images"
    )

    return (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        dicts_class,
    )


class Sketchy_Extended_valid_test(data.Dataset):
    def __init__(
        self,
        args,
        set_class,
        dicts_class,
        transform=None,
        type_skim="images",
    ):
        self.transform = transform
        self.plot = args.plot
        self.set_class = set_class
        self.dicts_class = dicts_class

        if type_skim == "images":
            self.dir_file = os.path.join(
                args.data_path, dataset_folder, "extended_photo"
            )
        elif type_skim == "sketch":
            self.dir_file = os.path.join(
                args.data_path, dataset_folder, "sketch", "tx_000000000000"
            )
        else:
            NameError(type_skim + " not implemented!")

        self.fnames, self.cls = get_file_list(self.dir_file, self.set_class, type_skim)
        self.loader = default_image_loader

    def __getitem__(self, index):
        label = self.cls[index]
        fname = os.path.join(self.dir_file, label, self.fnames[index])
        photo = self.transform(self.loader(fname))
        lbl = self.dicts_class.get(label)

        return photo, fname, lbl

    def __len__(self):
        return len(self.fnames)

    def get_class_dict(self):
        return self.set_class


class Sketchy_Extended_train(data.Dataset):
    def __init__(self, args, train_class, dicts_class, transform=None):

        self.transform = transform
        self.train_class = train_class
        self.dicts_class = dicts_class

        self.dir_image = os.path.join(args.data_path, dataset_folder, "extended_photo")
        self.dir_sketch = os.path.join(
            args.data_path, dataset_folder, "sketch", "tx_000000000000"
        )
        self.fnames_sketch, self.cls_sketch = get_file_list(
            self.dir_sketch, self.train_class, "sketch"
        )
        self.loader = default_image_loader

    def __getitem__(self, index):
        # Read sketch
        fname = os.path.join(
            self.dir_sketch,
            self.cls_sketch[index],
            self.fnames_sketch[index],
        )
        sketch = self.loader(fname)
        sketch = self.transform(sketch)

        # Target
        label = self.cls_sketch[index]
        lbl_pos = self.dicts_class.get(label)

        # Positive image
        # The constraint according to the ECCV 2018
        fname = get_random_file_from_path(os.path.join(self.dir_image, label))
        image_pos = self.transform(self.loader(fname))

        # Negative class
        possible_classes = [x for x in self.train_class if x != label]
        label_neg = np.random.choice(possible_classes, 1)[0]
        fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
        image_neg = self.transform(self.loader(fname))
        lbl_neg = self.dicts_class.get(label_neg)

        return sketch, image_pos, image_neg, lbl_pos, lbl_neg

    def __len__(self):
        return len(self.fnames_sketch)

    def get_class_dict(self):
        return self.train_class