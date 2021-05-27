import random

import numpy as np
import torch.utils.data as data

from src.constants import SKETCHY, TUBERLIN, FOLDERS
from src.data.default_dataset import DefaultDataset


class SkTu(data.Dataset):
    """
    Custom dataset for Sketchy and TU-Berlin common training
    """

    def __init__(
        self, args, dataset_type, dicts_class, data, transform, image_type=None
    ):
        """
        Initialises the dataset with the corresponding images/sketch path and classes
        Args:
            - args: arguments reveived from the command line (argparse)
            - dataset_type: dataset split ('train', 'valid' or 'test')
            - dicts_class:  dictionnnary mapping number to classes
            - data: list data for each dataset [sketchy_data, tuberlin data]
                each containing a list [images path, image classes, sketch paths, sketch classes]
            - transform: pytorch transform to apply on the data
            - image_type: type of the data: can be either 'sketches' or 'images'
        """
        self.dataset_type = dataset_type
        self.image_type = image_type
        self.dicts_class = dicts_class

        # Sketchy data
        self.sketchy = DefaultDataset(
            args,
            FOLDERS[SKETCHY],
            dataset_type,
            dicts_class[0],
            data[0],
            transform,
            image_type,
        )
        # Tuberlin data
        self.tuberlin = DefaultDataset(
            args,
            FOLDERS[TUBERLIN],
            dataset_type,
            dicts_class[1],
            data[1],
            transform,
            image_type,
        )

        # Separator between sketchy and tuberlin datasets
        self.sketchy_limit_sketch = len(self.sketchy.fnames_sketch)
        self.sketchy_limit_images = len(self.sketchy.fnames_image)

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
            (self.dataset_type == "train" and index < self.sketchy_limit_sketch)
            or (self.image_type == "images" and index < self.sketchy_limit_images)
            or (self.image_type == "sketches" and index < self.sketchy_limit_sketch)
        ):
            return self.sketchy.__getitem__(index)

        else:
            if self.image_type == "sketches" or self.dataset_type == "train":
                index -= self.sketchy_limit_sketch
            elif self.image_type == "images":
                index -= self.sketchy_limit_images
            return self.tuberlin.__getitem__(index)

    def __len__(self):
        """ Number of sketches/images in the dataset """
        if self.dataset_type == "train" or self.image_type == "sketches":
            return len(self.sketchy.fnames_sketch) + len(self.tuberlin.fnames_sketch)
        else:
            return len(self.sketchy.fnames_image) + len(self.tuberlin.fnames_image)

    def get_class_dict(self):
        """ Dictionnary of categories of the dataset """
        return self.dicts_class
