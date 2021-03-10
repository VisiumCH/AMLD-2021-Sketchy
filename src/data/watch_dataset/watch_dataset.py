import os
import random
from glob import glob
import numpy as np
import torch.utils.data as data

from src.data.utils import default_image_loader
from src.data.watch_dataset.utils import watch_dataset_split, get_watch_dict, get_watch_label


def Watch_Extended(args, transform='None'):
    '''
    Creates all the data loaders for any dataset
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get all images path
    images_path = glob(args.data_path + '/Watch/*/*/*/*image.png')

    # Split train, valid, test sets
    train_data, valid_data, test_data = watch_dataset_split(images_path)

    # Get dict
    dicts_class = get_watch_dict(images_path)

    # Data Loaders
    train_loader = WatchDataset(args, 'train', dicts_class, train_data, transform)
    valid_sk_loader = WatchDataset(args, 'valid', dicts_class, valid_data, transform, 'sketches')
    valid_im_loader = WatchDataset(args, 'valid', dicts_class, valid_data, transform, 'images')
    test_sk_loader = WatchDataset(args, 'test', dicts_class, test_data, transform, 'sketches')
    test_im_loader = WatchDataset(args, 'test', dicts_class, test_data, transform, 'images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class WatchDataset(data.Dataset):
    '''
    Custom dataset for TU-Berlin's
    '''

    def __init__(self, args, dataset_type, dicts_class, data, transform=None, image_type=None):

        self.transform = transform
        self.dataset_type = dataset_type
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.image_type = image_type

        self.dir_sketch = os.path.join(args.data_path, 'Watch', 'sketches')
        self.dir_image = os.path.join(args.data_path, 'Watch', 'images')

        self.fnames_image, self.fnames_sketch = data[0], data[1]
        self.number_images = len(self.fnames_image)

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
        if self.dataset_type == 'train':
            # Read sketch
            sketch_fname = self.fnames_sketch[index]
            sketch = self.transform(self.loader(sketch_fname))

            # Positive image (the one from which the sketch is generated)
            im_pos_fname = self.fnames_image[index]
            image_pos = self.transform(self.loader(im_pos_fname))
            lbl_pos = get_watch_label(self.fnames_image[index])

            # Negative image (any other image)
            index_neg = random.randint(0, self.number_images)
            while index_neg == index:  # make sure not same index
                index_neg = random.randint(0, self.number_images)
            im_neg_fname = self.fnames_image[index_neg]
            image_neg = self.transform(self.loader(im_neg_fname))
            lbl_neg = get_watch_label(self.fnames_image[index_neg])

            return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        else:
            if self.image_type == 'images':
                fname = self.fnames_image[index]
            elif self.image_type == 'sketches':
                fname = self.fnames_sketch[index]

            photo = self.transform(self.loader(fname))
            lbl = get_watch_label(self.fnames_image[index])
            return photo, fname, lbl

    def __len__(self):
        # Number of sketches/images in the dataset
        return len(self.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.dicts_class
