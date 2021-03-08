import random
import numpy as np
import torch.utils.data as data
from src.data.utils import dataset_split
from src.data.default_dataset import DefaultDataset


def SkTuQd_Extended(args, transform="None"):
    '''
    Creates sketchy, tu-berlin and quickdraw data loader in one go
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sketchy, TU-Berlin and Quickdraw
    dicts_class_sketchy, train_data_sketchy, valid_data_sketchy, test_data_sketchy = dataset_split(
        args, dataset_folder="Sketchy")
    dicts_class_tuberlin, train_data_tuberlin, valid_data_tuberlin, test_data_tuberlin = dataset_split(
        args, dataset_folder="TU-Berlin")
    dicts_class_quickdraw, train_data_quickdraw, valid_data_quickdraw, test_data_quickdraw = dataset_split(
        args, dataset_folder="Quickdraw")

    # Data Loaders
    train_loader = SkTuQd(args, 'train', dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                          train_data_sketchy, train_data_tuberlin, train_data_quickdraw, transform)
    valid_sk_loader = SkTuQd(args, 'valid', dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                             valid_data_sketchy, valid_data_tuberlin, valid_data_quickdraw, transform, "sketches")
    valid_im_loader = SkTuQd(args, 'valid', dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                             valid_data_sketchy, valid_data_tuberlin, valid_data_quickdraw, transform, "images")
    test_sk_loader = SkTuQd(args, 'test', dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                            test_data_sketchy, test_data_tuberlin, test_data_quickdraw, transform, "sketches")
    test_im_loader = SkTuQd(args, 'test', dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                            test_data_sketchy, test_data_tuberlin, test_data_quickdraw, transform, "images")
    return (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        [dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw]
    )


if __name__ == "__main__":
    from src.options import Options

    # Parse options
    args = Options().parse()
    SkTuQd_Extended(args)


class SkTuQd(data.Dataset):
    '''
    Custom dataset for Stetchy's training
    '''

    def __init__(self, args, dataset_type, dicts_class_sketchy, dicts_class_tuberlin, dicts_class_quickdraw,
                 data_sketchy, data_tuberlin, data_quickdraw, transform=None, image_type=None):
        self.dataset_type = dataset_type
        self.image_type = image_type
        self.dicts_class_sketchy = dicts_class_sketchy
        self.dicts_class_tuberlin = dicts_class_tuberlin
        self.dicts_class_quickdraw = dicts_class_quickdraw

        # Sketchy data
        self.sketchy = DefaultDataset(args, "Sketchy", dataset_type, dicts_class_sketchy,
                                      data_sketchy, transform, image_type)

        # Tuberlin data
        self.tuberlin = DefaultDataset(args, "TU-Berlin", dataset_type, dicts_class_tuberlin,
                                       data_tuberlin, transform, image_type)

        # Quickdraw data
        self.quickdraw = DefaultDataset(args, "Quickdraw", dataset_type, dicts_class_quickdraw,
                                        data_quickdraw, transform, image_type)

        # Separator between sketchy and tuberlin datasets
        self.sketchy_limit_sketch = len(self.sketchy.fnames_sketch)
        self.sketchy_limit_images = len(self.sketchy.fnames_image)

        # Separator between tuberlin and quickdraw datasets
        self.tuberlin_limit_sketch = len(self.sketchy.fnames_sketch) + len(self.tuberlin.fnames_sketch)
        self.tuberlin_limit_images = len(self.sketchy.fnames_image) + len(self.tuberlin.fnames_image)

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
                or (self.image_type == 'sketches' and index < self.sketchy_limit_sketch)):
            return self.sketchy.__getitem__(index)

        elif ((self.dataset_type == 'train' and index < self.tuberlin_limit_sketch)
                or (self.image_type == 'images' and index < self.tuberlin_limit_images)
                or (self.image_type == 'sketches' and index < self.tuberlin_limit_sketch)):

            if (self.image_type == 'sketches' or self.dataset_type == 'train'):
                index -= self.sketchy_limit_sketch
            elif self.image_type == 'images':
                index -= self.sketchy_limit_images
            return self.tuberlin.__getitem__(index)

        else:
            if (self.image_type == 'sketches' or self.dataset_type == 'train'):
                index -= self.tuberlin_limit_sketch
            elif self.image_type == 'images':
                index -= self.tuberlin_limit_images
            return self.quickdraw.__getitem__(index)

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == 'train' or self.image_type == 'sketches':
            return len(self.sketchy.fnames_sketch) + \
                len(self.tuberlin.fnames_sketch) + \
                len(self.quickdraw.fnames_sketch)
        else:
            return len(self.sketchy.fnames_image) + \
                len(self.tuberlin.fnames_image) + \
                len(self.quickdraw.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return [self.dicts_class_sketchy, self.dicts_class_tuberlin, self.dicts_class_quickdraw]
