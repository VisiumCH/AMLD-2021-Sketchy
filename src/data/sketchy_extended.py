import os

from src.data.parent_dataset import ParentDataset
from src.data.utils import get_file_list, default_image_loader, dataset_split


def Sketchy_Extended(args, transform="None"):
    '''
    Creates all the data loaders for Sketchy dataset
    '''
    # Get dataset classes
    train_class, valid_class, test_class, dicts_class = dataset_split(
        args, dataset_folder="Sketchy", image_folder="extended_photo", name='sketchy')

    # Data Loaders
    train_loader = Sketchy(args, 'train', train_class, dicts_class, transform)
    valid_sk_loader = Sketchy(args, 'valid', valid_class, dicts_class, transform, "sketch")
    valid_im_loader = Sketchy(args, 'valid', valid_class, dicts_class, transform, "images")
    test_sk_loader = Sketchy(args, 'test', test_class, dicts_class, transform, "sketch")
    test_im_loader = Sketchy(args, 'test', test_class, dicts_class, transform, "images")

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class Sketchy(ParentDataset):
    '''
    Custom dataset for Stetchy's training
    '''

    def __init__(self, args, dataset_type, set_class, dicts_class, transform=None, image_type=None):
        super().__init__(dataset_type, set_class, dicts_class, transform, image_type=None)
        self.loader_image = default_image_loader

        self.dir_image = os.path.join(args.data_path, "Sketchy/extended_photo")
        self.dir_sketch = os.path.join(args.data_path, "Sketchy/sketch/tx_000000000000")

        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.set_class, "sketch")
        self.fnames_image, self.cls_image = get_file_list(self.dir_image, self.set_class, "images")
