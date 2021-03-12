import random

import numpy as np
import torch.utils.data as data

from src.data.constants import DatasetFolder, ImageType, Split
from src.data.default_dataset import DefaultDataset
from src.data.utils import dataset_split, get_class_dict


def SkTu_Extended(args, transform):
    '''
    Creates all the data loaders for training with Sketchy and TU-Berlin datasets
    Args:
        - args: arguments reveived from the command line (argparse)
        - transform: pytorch transform to apply on the data
    Return:
        - train_loader: data loader for the training set
        - valid_sk_loader: data loader of sketches for the validation set
        - valid_im_loader: data loader of images for the validation set
        - test_sk_loader: data loader of sketches for the test set
        - test_im_loader: data loader of images for the test set
        - dicts_class: dictionnnary mapping number to classes.
                        The key is a unique number and the value is the class name.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # # Sketchy
    dicts_class_sketchy = get_class_dict(args, DatasetFolder.sketchy)
    train_data_sketchy, valid_data_sketchy, test_data_sketchy = dataset_split(
        args, DatasetFolder.sketchy, args.training_split, args.valid_split)

    # TU-Berlin
    dicts_class_tuberlin = get_class_dict(args, DatasetFolder.tuberlin)
    train_data_tuberlin, valid_data_tuberlin, test_data_tuberlin = dataset_split(
        args, DatasetFolder.tuberlin, args.training_split, args.valid_split)

    dicts_class = [dicts_class_sketchy, dicts_class_tuberlin]
    train_data = [train_data_sketchy, train_data_tuberlin]
    valid_data = [valid_data_sketchy, valid_data_tuberlin]
    test_data = [test_data_sketchy, test_data_tuberlin]

    # Data Loaders
    train_loader = SkTu(args, Split.train, dicts_class, train_data, transform)
    valid_sk_loader = SkTu(args, Split.valid, dicts_class, valid_data, transform, ImageType.sketch)
    valid_im_loader = SkTu(args, Split.valid, dicts_class, valid_data, transform, ImageType.image)
    test_sk_loader = SkTu(args, Split.test, dicts_class, test_data, transform, ImageType.sketch)
    test_im_loader = SkTu(args, Split.test, dicts_class, test_data, transform, ImageType.image)
    return (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        [dicts_class_sketchy, dicts_class_tuberlin]
    )


if __name__ == "__main__":
    from src.options import Options

    # Parse options
    args = Options().parse()
    SkTu_Extended(args)


class SkTu(data.Dataset):
    '''
    Custom dataset for Sketchy and TU-Berlin common training
    '''

    def __init__(self, args, dataset_type, dicts_class, data, transform, image_type=None):
        '''
        Initialises the dataset with the corresponding images/sketch path and classes
        Args:
            - args: arguments reveived from the command line (argparse)
            - dataset_type: dataset split ('train', 'valid' or 'test')
            - dicts_class:  dictionnnary mapping number to classes
            - data: list data for each dataset [sketchy_data, tuberlin data]
                each containing a list [images path, image classes, sketch paths, sketch classes]
            - transform: pytorch transform to apply on the data
            - image_type: type of the data: can be either 'sketches' or 'images'
        '''
        self.dataset_type = dataset_type
        self.image_type = image_type
        self.dicts_class = dicts_class

        # Sketchy data
        self.sketchy = DefaultDataset(args, DatasetFolder.sketchy, dataset_type,
                                      dicts_class[0], data[0], transform, image_type)
        # Tuberlin data
        self.tuberlin = DefaultDataset(args, DatasetFolder.tuberlin, dataset_type,
                                       dicts_class[1], data[1], transform, image_type)

        # Separator between sketchy and tuberlin datasets
        self.sketchy_limit_sketch = len(self.sketchy.fnames_sketch)
        self.sketchy_limit_images = len(self.sketchy.fnames_image)

    def __getitem__(self, index):
        '''
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
        '''
        if ((self.dataset_type == Split.train and index < self.sketchy_limit_sketch)
                or (self.image_type == ImageType.image and index < self.sketchy_limit_images)
                or (self.image_type == ImageType.sketch and index < self.sketchy_limit_sketch)):
            return self.sketchy.__getitem__(index)

        else:
            if (self.image_type == ImageType.sketch or self.dataset_type == Split.train):
                index -= self.sketchy_limit_sketch
            elif self.image_type == ImageType.image:
                index -= self.sketchy_limit_images
            return self.tuberlin.__getitem__(index)

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == Split.train or self.image_type == ImageType.sketch:
            return len(self.sketchy.fnames_sketch) + len(self.tuberlin.fnames_sketch)
        else:
            return len(self.sketchy.fnames_image) + len(self.tuberlin.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.dicts_class
