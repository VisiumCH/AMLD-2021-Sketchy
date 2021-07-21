import os

import random
import numpy as np
import torch.utils.data as data

from src.data.utils import get_class_dict, dataset_split

from src.constants import DATA_PATH, SKETCHY, QUICKDRAW, TUBERLIN, FOLDERS, SKTU, SKTUQD
from src.data.default_dataset import DefaultDataset


def make_composite_dataset(args, transform, dataset_name):
    """
    Creates all the data loaders for training with Sketchy, TU-Berlin and Quickdraw datasets
    Args:
        - args: arguments reveived from the command line (argparse)
        - transform: pytorch transform to apply on the data
        - dataset_name: name of composite dataset (sk+tu or sk+tu+qd)
    Return:
        - train_loader: data loader for the training set
        - valid_sk_loader: data loader of sketches for the validation set
        - valid_im_loader: data loader of images for the validation set
        - test_sk_loader: data loader of sketches for the test set
        - test_im_loader: data loader of images for the test set
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    if dataset_name == SKTU:
        dataset_class = SkTu
        datasets_list = [SKETCHY, TUBERLIN]
    elif dataset_name == SKTUQD:
        dataset_class = SkTuQd
        datasets_list = [SKETCHY, TUBERLIN, QUICKDRAW]
    else:
        raise Exception(
            f"Composite dataset only possible with {SKTU} or {SKTUQD}.\nHere {dataset_name}"
        )

    # Get Sketchy, TU-Berlin and Quickdraw datasets
    dicts_class, train_data, valid_data, test_data = [], [], [], []
    for dataset in datasets_list:
        dataset_folder = os.path.join(DATA_PATH, FOLDERS[dataset])
        dict_class = get_class_dict(dataset_folder)
        train_dataset, valid_dataset, test_dataset = dataset_split(
            dataset_folder, args.training_split, args.valid_split
        )
        dicts_class.append(dict_class)
        train_data.append(train_dataset)
        valid_data.append(valid_dataset)
        test_data.append(test_dataset)

    # Data Loaders
    train_loader = dataset_class("train", dicts_class, train_data, transform)
    valid_sk_loader = dataset_class(
        "valid", dicts_class, valid_data, transform, "sketches"
    )
    valid_im_loader = dataset_class(
        "valid", dicts_class, valid_data, transform, "images"
    )
    test_sk_loader = dataset_class(
        "test", dicts_class, test_data, transform, "sketches"
    )
    test_im_loader = dataset_class("test", dicts_class, test_data, transform, "images")

    return (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        dicts_class,
    )


class SkTu(data.Dataset):
    """
    Custom dataset for Sketchy and TU-Berlin common training
    """

    def __init__(self, mode, dicts_class, data, transform, image_type=None):
        """
        Initialises the dataset with the corresponding images/sketch path and classes
        Args:
            - mode: dataset split ('train', 'valid' or 'test')
            - dicts_class:  dictionnnary mapping number to classes
            - data: list data for each dataset [sketchy_data, tuberlin data]
                each containing a list [images path, image classes, sketch paths, sketch classes]
            - transform: pytorch transform to apply on the data
            - image_type: type of the data: can be either 'sketches' or 'images'
        """
        self.mode = mode
        self.image_type = image_type
        self.dicts_class = dicts_class

        self.data_args = {
            "mode": mode,
            "transform": transform,
            "image_type": image_type,
        }

        # Sketchy data
        self.sketchy = DefaultDataset(
            data=data[0],
            dicts_class=dicts_class[0],
            dataset_folder=os.path.join(DATA_PATH, FOLDERS[SKETCHY]),
            **self.data_args,
        )

        # Tuberlin data
        self.tuberlin = DefaultDataset(
            data=data[1],
            dicts_class=dicts_class[1],
            dataset_folder=os.path.join(DATA_PATH, FOLDERS[TUBERLIN]),
            **self.data_args,
        )

        # No quickdraw with SkTu only
        self.quidraw = None

        # Separator between sketchy and tuberlin datasets
        self.sketchy_limit_sketch = len(self.sketchy.fnames_sketch)
        self.sketchy_limit_images = len(self.sketchy.fnames_image)

        # Separator between tuberlin and quickdraw datasets
        self.tuberlin_limit_sketch = len(self.sketchy.fnames_sketch) + len(
            self.tuberlin.fnames_sketch
        )
        self.tuberlin_limit_images = len(self.sketchy.fnames_image) + len(
            self.tuberlin.fnames_image
        )

        # Length of the dataset
        if self.mode == "train" or self.image_type == "sketches":
            self.length = len(self.sketchy.fnames_sketch) + len(
                self.tuberlin.fnames_sketch
            )
        else:
            self.length = len(self.sketchy.fnames_image) + len(
                self.tuberlin.fnames_image
            )

    def __getitem__(self, index):
        """
        Get training data based on index
        Args:
            - index: index of the sketch or image
        Return:
            if training:
            - sketch: sketch image
            - image_pos: image of same category of sketch
            - image_neg: image of different category of sketch
            - lbl_pos: category of sketch and image_pos
            - lbl_neg: category of image_neg
            otherwise (validating or testing):
            - photo: photo or sketch image
            - fname: path to the photo or sketch
            - lbl: category of the sketch or image
        """
        if (
            (self.mode == "train" and index < self.sketchy_limit_sketch)
            or (self.image_type == "images" and index < self.sketchy_limit_images)
            or (self.image_type == "sketches" and index < self.sketchy_limit_sketch)
        ):
            return self.sketchy.__getitem__(index)

        elif (
            (self.mode == "train" and index < self.tuberlin_limit_sketch)
            or (self.image_type == "images" and index < self.tuberlin_limit_images)
            or (self.image_type == "sketches" and index < self.tuberlin_limit_sketch)
        ):
            if self.image_type == "sketches" or self.mode == "train":
                index -= self.sketchy_limit_sketch
            elif self.image_type == "images":
                index -= self.sketchy_limit_images
            return self.tuberlin.__getitem__(index)

        else:
            if self.image_type == "sketches" or self.mode == "train":
                index -= self.tuberlin_limit_sketch
            elif self.image_type == "images":
                index -= self.tuberlin_limit_images
            return self.quickdraw.__getitem__(index)

    def __len__(self):
        """ Number of sketches/images in the dataset """
        return self.length

    def get_class_dict(self):
        """ Dictionnary of categories of the dataset """
        return self.dicts_class


class SkTuQd(SkTu):
    """
    Custom dataset for Sketchy, TU-Berlin and Quickdraw common training
    """

    def __init__(self, mode, dicts_class, data, transform, image_type=None):
        """
        Initialises the dataset with the corresponding images/sketch path and classes
        Args:
            - mode: dataset split ('train', 'valid' or 'test')
            - dicts_class:  dictionnnary mapping number to classes
            - data: list data for each dataset [sketchy_data, tuberlin data, quickdraw_data]
                each containing a list [images path, image classes, sketch paths, sketch classes]
            - transform: pytorch transform to apply on the data
            - image_type: type of the data: can be either 'sketches' or 'images'
        """
        # Sketchy and Quickdraw data
        super().__init__(DATA_PATH, mode, dicts_class, data, transform, image_type)

        # Quickdraw data
        self.quickdraw = DefaultDataset(
            data=data[2],
            dicts_class=dicts_class[2],
            dataset_folder=os.path.join(DATA_PATH, FOLDERS[QUICKDRAW]),
            **self.data_args,
        )

        # Update length of the dataset
        if self.mode == "train" or self.image_type == "sketches":
            self.length += len(self.quickdraw.fnames_sketch)
        else:
            self.length += len(self.quickdraw.fnames_image)
