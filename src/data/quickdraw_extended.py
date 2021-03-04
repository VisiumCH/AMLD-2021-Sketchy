import os
import random

import numpy as np

from src.data.parent_dataset import ParentDataset
from src.data.utils import get_file_list, default_image_loader, dataset_split


def Quickdraw_Extended(args, transform='None'):
    '''
    Creates all the data loaders for Quickdraw dataset
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get dataset classes
    train_class, valid_class, test_class, dicts_class = dataset_split(
        args, dataset_folder="Quickdraw", image_folder="images", name='quickdraw')

    # Data Loaders
    train_loader = Quickdraw(args, 'train', train_class, dicts_class, transform)
    valid_sk_loader = Quickdraw(args, 'valid', valid_class, dicts_class, transform, 'sketch')
    valid_im_loader = Quickdraw(args, 'valid', valid_class, dicts_class, transform, 'images')
    test_sk_loader = Quickdraw(args, 'test', test_class, dicts_class, transform, 'sketch')
    test_im_loader = Quickdraw(args, 'test', test_class, dicts_class, transform, 'images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class Quickdraw(ParentDataset):
    '''
    Custom dataset for Quickdraw
    '''

    def __init__(self, args, dataset_type, set_class, dicts_class, transform=None, image_type=None):
        super().__init__(dataset_type, set_class, dicts_class, transform, image_type)
        self.loader_image = default_image_loader

        self.dir_sketch = os.path.join(args.data_path, 'Quickdraw/sketches')
        self.dir_image = os.path.join(args.data_path, 'Quickdraw/images')

        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.set_class, 'sketch')
        self.fnames_image, self.cls_images = get_file_list(self.dir_image, self.set_class, 'images')
