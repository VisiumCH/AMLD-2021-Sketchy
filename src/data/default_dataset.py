import os
import random

import numpy as np
import torch.utils.data as data

from src.constants import DATA_PATH, QUICKDRAW, TUBERLIN, FOLDERS
from src.data.utils import (
    default_image_loader,
    default_image_loader_tuberlin,
    get_random_file_from_path,
    dataset_split,
    get_class_dict,
)


def make_default_dataset(args, dataset_folder, transform):
    """
    Creates all the data loaders for any single dataset (Sketchy, TU_Berlin or Quickdraw)
    Args:
        - args: arguments reveived from the command line (argparse)
        - dataset_folder: name of the folder containing the data
        - transform: pytorch transform to apply on the data
    Return:
        - train_loader: data loader for the training set
        - valid_sk_loader: data loader of sketches for the validation set
        - valid_im_loader: data loader of images for the validation set
        - test_sk_loader: data loader of sketches for the test set
        - test_im_loader: data loader of images for the test set
        - dicts_class: dictionnnary mapping number to classes.
                        The key is a unique number and the value is the class name.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Memory error otherwise
    if dataset_folder == FOLDERS[QUICKDRAW]:
        training_split = args.qd_training_split
        valid_split = args.qd_valid_split
    else:
        training_split = args.training_split
        valid_split = args.valid_split

    # Get dataset classes
    dataset_folder = os.path.join(DATA_PATH, dataset_folder)
    dicts_class = get_class_dict(dataset_folder)
    train_data, valid_data, test_data = dataset_split(
        dataset_folder, training_split, valid_split
    )

    # Data Loaders
    data_args = {
        "dataset_folder": dataset_folder,
        "dicts_class": dicts_class,
        "transform": transform,
    }
    train_loader = DefaultDataset(mode="train", data=train_data, **data_args)
    valid_sk_loader = DefaultDataset(
        mode="valid", data=valid_data, image_type="sketches", **data_args
    )
    valid_im_loader = DefaultDataset(
        mode="valid", data=valid_data, image_type="images", **data_args
    )
    test_sk_loader = DefaultDataset(
        mode="test", data=test_data, image_type="sketches", **data_args
    )
    test_im_loader = DefaultDataset(
        mode="test", data=test_data, image_type="images", **data_args
    )

    return (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        dicts_class,
    )


class DefaultDataset(data.Dataset):
    """
    Default dataset to get data
    """

    def __init__(
        self,
        dataset_folder,
        mode,
        dicts_class,
        data,
        transform,
        image_type=None,
    ):
        """
        Initialises the dataset with the corresponding images/sketch path and classes
        Args:
            - dataset_folder: path to the folder containing the data
            - mode: dataset split ('train', 'valid' or 'test')
            - dicts_class:  dictionnnary mapping number to classes
            - data: list data of the classes [images path, image classes, sketch paths, sketch classes]
            - transform: pytorch transform to apply on the data
            - image_type: type of the data: can be either 'sketches' or 'images'
        """
        self.transform = transform
        self.mode = mode
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.image_type = image_type

        if dataset_folder.split("/")[-1] == FOLDERS[TUBERLIN]:
            self.loader_image = default_image_loader_tuberlin
        else:
            self.loader_image = default_image_loader

        self.dir_sketch = os.path.join(dataset_folder, "sketches")
        self.dir_image = os.path.join(dataset_folder, "images")

        self.fnames_image, self.cls_image = data[0], data[1]
        self.fnames_sketch, self.cls_sketch = data[2], data[3]

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
        if self.mode == "train":
            # Read sketch
            sketch_fname = os.path.join(
                self.dir_sketch, self.cls_sketch[index], self.fnames_sketch[index]
            )
            sketch = self.transform(self.loader(sketch_fname))

            # Target
            label = self.cls_sketch[index]
            lbl_pos = self.dicts_class.get(label)

            # Positive image
            im_pos_fname = get_random_file_from_path(
                os.path.join(self.dir_image, label)
            )
            image_pos = self.transform(self.loader_image(im_pos_fname))

            # Negative class
            # Hard negative
            possible_classes = [x for x in self.dicts_class if x != label]
            label_neg = np.random.choice(possible_classes, 1)[0]
            lbl_neg = self.dicts_class.get(label_neg)

            im_neg_fname = get_random_file_from_path(
                os.path.join(self.dir_image, label_neg)
            )
            image_neg = self.transform(self.loader_image(im_neg_fname))

            return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        else:
            if self.image_type == "images":
                label = self.cls_image[index]
                fname = os.path.join(self.dir_image, label, self.fnames_image[index])
                photo = self.transform(self.loader_image(fname))

            elif self.image_type == "sketches":
                label = self.cls_sketch[index]
                fname = os.path.join(self.dir_sketch, label, self.fnames_sketch[index])
                photo = self.transform(self.loader(fname))

            lbl = self.dicts_class.get(label)
            return photo, fname, lbl

    def __len__(self):
        """ Number of sketches/images in the dataset """
        if self.mode == "train" or self.image_type == "sketches":
            return len(self.fnames_sketch)
        else:
            return len(self.fnames_image)

    def get_class_dict(self):
        """ Dictionnary of categories of the dataset """
        return self.dicts_class
