import random
import numpy as np
import torch.utils.data as data
from src.data.quickdraw_extended import Quickdraw
from src.data.sketchy_extended import Sketchy
from src.data.tuberlin_extended import TUBerlin
from src.data.utils import dataset_split


def All_Extended(args, transform="None"):
    '''
    Creates all the data loaders for Sketchy dataset
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sketchy and TU-Berlin
    train_class_sketchy, valid_class_sketchy, test_class_sketchy, dicts_class_sketchy = dataset_split(
        args, dataset_folder="Sketchy", image_folder="extended_photo", name='sketchy')

    train_class_tuberlin, valid_class_tuberlin, test_class_tuberlin, dicts_class_tuberlin = dataset_split(
        args, dataset_folder="TU-Berlin", image_folder="images", name='tuberlin')

    train_class_quickdraw, valid_class_quickdraw, test_class_quickdraw, dicts_class_quickdraw = dataset_split(
        args, dataset_folder="Quickdraw", image_folder="images", name='quickdraw')

    # Data Loaders
    train_loader = All(args, 'train', train_class_sketchy, train_class_tuberlin, train_class_quickdraw,
                       dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw, transform)
    valid_sk_loader = All(args, 'valid', valid_class_sketchy, valid_class_tuberlin, valid_class_quickdraw,
                          dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw, transform, "sketch")
    valid_im_loader = All(args, 'valid', valid_class_sketchy, valid_class_tuberlin, valid_class_quickdraw,
                          dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw, transform, "images")
    test_sk_loader = All(args, 'test', test_class_sketchy, test_class_tuberlin, test_class_quickdraw,
                         dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw, transform, "sketch")
    test_im_loader = All(args, 'test', test_class_sketchy, test_class_tuberlin, test_class_quickdraw,
                         dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw, transform, "images")
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
    All_Extended(args)


class All(data.Dataset):
    '''
    Custom dataset for Stetchy's training
    '''

    def __init__(self, args, dataset_type, set_class_sketchy, set_class_tuberlin, set_class_quickdraw,
                 dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                 transform=None, image_type=None):
        self.dataset_type = dataset_type
        self.image_type = image_type

        # Sketchy data
        self.sketchy = Sketchy(args, dataset_type, set_class_sketchy,
                               dicts_class_sketchy, transform, image_type)

        # Tuberlin data
        self.tuberlin = TUBerlin(args, dataset_type, set_class_tuberlin,
                                 dicts_class_tuberlin, transform, image_type)

        # Quickdraw data
        self.quickdraw = Quickdraw(args, dataset_type, set_class_quickdraw,
                                   dicts_class_quickdraw, transform, image_type)

        # Separator between sketchy and tuberlin datasets
        self.sketchy_limit_sketch = len(self.sketchy.fnames_sketch)
        self.sketchy_limit_images = len(self.sketchy.fnames_image)

        # Separator between tuberlin and quickdraw datasets
        self.tuberlin_limit_sketch = len(self.sketchy.fnames_sketch) + len(self.tuberlin.fnames_sketch)
        self.tuberlin_limit_images = len(self.sketchy.fnames_image) + len(self.tuberlin.fnames_sketch)

    def __getitem__(self, index):
        '''
        Get training data based on sketch index
        Args:
            - index: index of the sketch
        Return:
            - sketch: sketch image
            - image_pos: image of same category of sketch
            - image_neg: image of different category of sketch
            - lbl_pos: category of sketch and image_pos
            - lbl_neg: category of image_neg
        '''
        if ((self.dataset_type == 'train' and index < self.sketchy_limit_sketch)
                or (self.image_type == 'images' and index < self.sketchy_limit_images)
                or (self.image_type == 'sketch' and index < self.sketchy_limit_sketch)):
            return self.sketchy.__getitem__(index)

        elif ((self.dataset_type == 'train' and index < self.tuberlin_limit_sketch)
                or (self.image_type == 'images' and index < self.tuberlin_limit_images)
                or (self.image_type == 'sketch' and index < self.tuberlin_limit_sketch)):

            if (self.image_type == 'sketch' or self.dataset_type == 'train'):
                index -= self.sketchy_limit_sketch
            elif self.image_type == 'images':
                index -= self.sketchy_limit_images
            return self.tuberlin.__getitem__(index)

        else:
            if (self.image_type == 'sketch' or self.dataset_type == 'train'):
                index -= self.tuberlin_limit_sketch
            elif self.image_type == 'images':
                index -= self.tuberlin_limit_images
            return self.quickdraw.__getitem__(index)

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == 'train' or self.image_type == 'sketch':
            return len(self.sketchy.fnames_sketch) + \
                len(self.tuberlin.fnames_sketch) + \
                len(self.quickdraw.fnames_sketch)
        else:
            return len(self.sketchy.fnames_image) + \
                len(self.tuberlin.fnames_image) + \
                len(self.quickdraw.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.set_class_sketchy, self.set_class_tuberlin, self.set_class_quickdraw
