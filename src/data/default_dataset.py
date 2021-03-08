import os
import random

import numpy as np
import torch.utils.data as data

from src.data.utils import (default_image_loader, default_image_loader_tuberlin,
                            get_random_file_from_path, get_file_list, dataset_split)


def DefaultDataset_Extended(args, dataset_folder, transform='None'):
    '''
    Creates all the data loaders for any dataset
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get dataset classes
    train_class, valid_class, test_class, dicts_class = dataset_split(args, dataset_folder)

    # Data Loaders
    train_loader = DefaultDataset(args, dataset_folder, 'train', train_class, dicts_class, transform)
    valid_sk_loader = DefaultDataset(args, dataset_folder, 'valid', valid_class, dicts_class, transform, 'sketch')
    valid_im_loader = DefaultDataset(args, dataset_folder, 'valid', valid_class, dicts_class, transform, 'images')
    test_sk_loader = DefaultDataset(args, dataset_folder, 'test', test_class, dicts_class, transform, 'sketch')
    test_im_loader = DefaultDataset(args, dataset_folder, 'test', test_class, dicts_class, transform, 'images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class DefaultDataset(data.Dataset):
    '''
    Custom dataset for TU-Berlin's
    '''

    def __init__(self, args, dataset_folder, dataset_type, set_class, dicts_class, transform=None, image_type=None):

        self.transform = transform
        self.dataset_type = dataset_type
        self.set_class = set_class
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.image_type = image_type

        if dataset_folder == 'TU-Berlin':
            self.loader_image = default_image_loader
        else:
            self.loader_image = default_image_loader_tuberlin

        self.dir_sketch = os.path.join(args.data_path, dataset_folder, 'sketches')
        self.dir_image = os.path.join(args.data_path, dataset_folder, 'images')

        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.set_class, 'sketch')
        self.fnames_image, self.cls_images = get_file_list(self.dir_image, self.set_class, 'images')

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
            sketch_fname = os.path.join(self.dir_sketch, self.cls_sketch[index], self.fnames_sketch[index])
            sketch = self.transform(self.loader(sketch_fname))

            # Target
            label = self.cls_sketch[index]
            lbl_pos = self.dicts_class.get(label)

            # Positive image
            im_pos_fname = get_random_file_from_path(os.path.join(self.dir_image, label))
            image_pos = self.transform(self.loader_image(im_pos_fname))

            # Negative class
            # Hard negative
            possible_classes = [x for x in self.set_class if x != label]
            label_neg = np.random.choice(possible_classes, 1)[0]
            lbl_neg = self.dicts_class.get(label_neg)

            im_neg_fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
            image_neg = self.transform(self.loader_image(im_neg_fname))

            return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        else:
            if self.image_type == 'images':
                label = self.cls_images[index]
                fname = os.path.join(self.dir_image, label, self.fnames_image[index])
                photo = self.transform(self.loader_image(fname))

            elif self.image_type == 'sketch':
                label = self.cls_sketch[index]
                fname = os.path.join(self.dir_sketch, label, self.fnames_sketch[index])
                photo = self.transform(self.loader(fname))

            lbl = self.dicts_class.get(label)
            return photo, fname, lbl

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == 'train' or self.image_type == 'sketch':
            return len(self.fnames_sketch)
        else:
            return len(self.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.set_class
