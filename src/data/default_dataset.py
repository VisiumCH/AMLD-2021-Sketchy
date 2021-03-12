import os
import random

import numpy as np
import torch.utils.data as data

from src.data.constants import DatasetFolder, ImageType, Split
from src.data.utils import (default_image_loader, default_image_loader_tuberlin,
                            get_random_file_from_path, dataset_split, get_class_dict)


def DefaultDataset_Extended(args, dataset_folder, transform):
    '''
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
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    if dataset_folder == DatasetFolder.quickdraw:
        training_split = args.qd_training_split
        valid_split = args.qd_valid_split
    else:
        training_split = args.training_split
        valid_split = args.valid_split

    # Get dataset classes
    dicts_class = get_class_dict(args, dataset_folder)
    train_data, valid_data, test_data = dataset_split(args, dataset_folder, training_split, valid_split)

    # Data Loaders
    train_loader = DefaultDataset(args, dataset_folder, Split.train, dicts_class,
                                  train_data, transform)
    valid_sk_loader = DefaultDataset(args, dataset_folder, Split.valid, dicts_class,
                                     valid_data, transform, ImageType.sketch)
    valid_im_loader = DefaultDataset(args, dataset_folder, Split.valid, dicts_class,
                                     valid_data, transform, ImageType.image)
    test_sk_loader = DefaultDataset(args, dataset_folder, Split.test, dicts_class,
                                    test_data, transform, ImageType.sketch)
    test_im_loader = DefaultDataset(args, dataset_folder, Split.test, dicts_class,
                                    test_data, transform, ImageType.image)

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class DefaultDataset(data.Dataset):
    '''
    Default dataset to get data
    '''

    def __init__(self, args, dataset_folder, dataset_type, dicts_class, data, transform, image_type=None):
        '''
        Initialises the dataset with the corresponding images/sketch path and classes
        Args:
            - args: arguments reveived from the command line (argparse)
            - dataset_folder: name of the folder containing the data
            - dataset_type: dataset split ('train', 'valid' or 'test')
            - dicts_class:  dictionnnary mapping number to classes
            - data: list data of the classes [images path, image classes, sketch paths, sketch classes]
            - transform: pytorch transform to apply on the data
            - image_type: type of the data: can be either 'sketches' or 'images'
        '''
        self.transform = transform
        self.dataset_type = dataset_type
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.image_type = image_type

        if dataset_folder == DatasetFolder.tuberlin:
            self.loader_image = default_image_loader
        else:
            self.loader_image = default_image_loader_tuberlin

        self.dir_sketch = os.path.join(args.data_path, dataset_folder, ImageType.sketch)
        self.dir_image = os.path.join(args.data_path, dataset_folder, ImageType.image)

        self.fnames_image, self.cls_image = data[0], data[1]
        self.fnames_sketch, self.cls_sketch = data[2], data[3]

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
        if self.dataset_type == Split.train:
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
            possible_classes = [x for x in self.dicts_class if x != label]
            label_neg = np.random.choice(possible_classes, 1)[0]
            lbl_neg = self.dicts_class.get(label_neg)

            im_neg_fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
            image_neg = self.transform(self.loader_image(im_neg_fname))

            return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        else:
            if self.image_type == ImageType.image:
                label = self.cls_image[index]
                fname = os.path.join(self.dir_image, label, self.fnames_image[index])
                photo = self.transform(self.loader_image(fname))

            elif self.image_type == ImageType.sketch:
                label = self.cls_sketch[index]
                fname = os.path.join(self.dir_sketch, label, self.fnames_sketch[index])
                photo = self.transform(self.loader(fname))

            lbl = self.dicts_class.get(label)
            return photo, fname, lbl

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == Split.train or self.image_type == ImageType.sketch:
            return len(self.fnames_sketch)
        else:
            return len(self.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.dicts_class
