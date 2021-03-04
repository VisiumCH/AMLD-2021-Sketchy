import os
import random

from src.data.parent_dataset import ParentDataset
from src.data.utils import (
    get_file_list,
    default_image_loader_tuberlin,
    dataset_split
)


def TUBerlin_Extended(args, transform='None'):
    '''
    Creates all the data loaders for TU-Berlin dataset
    '''
    random.seed(args.seed)

    # Get dataset classes
    train_class, valid_class, test_class, dicts_class = dataset_split(
        args, dataset_folder="TU-Berlin", image_folder="images", name='tuberlin')

    # Data Loaders
    train_loader = TUBerlin(args, 'train', train_class, dicts_class, transform)
    valid_sk_loader = TUBerlin(args, 'valid', valid_class, dicts_class, transform, 'sketch')
    valid_im_loader = TUBerlin(args, 'valid', valid_class, dicts_class, transform, 'images')
    test_sk_loader = TUBerlin(args, 'test', test_class, dicts_class, transform, 'sketch')
    test_im_loader = TUBerlin(args, 'test', test_class, dicts_class, transform, 'images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class TUBerlin(ParentDataset):
    '''
    Custom dataset for TU-Berlin's
    '''

    def __init__(self, args, dataset_type, set_class, dicts_class, transform=None, image_type=None):
        super().__init__(dataset_type, set_class, dicts_class, transform, image_type)
        self.loader_image = default_image_loader_tuberlin

        self.dir_sketch = os.path.join(args.data_path, 'TU-Berlin/sketches')
        self.dir_image = os.path.join(args.data_path, 'TU-Berlin/images')

        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.set_class, 'sketch')
        self.fnames_image, self.cls_images = get_file_list(self.dir_image, self.set_class, 'images')
