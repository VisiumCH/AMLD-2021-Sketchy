from glob import glob
import os
import random

import numpy as np
import torch.utils.data as data

from src.data.utils import (
    create_dict_texts,
    get_file_list,
    default_image_loader,
    default_image_loader_tuberlin,
    get_random_file_from_path
)


def TUBerlin_Extended(args, transform='None'):
    '''
    Creates all the data loaders for TU-Berlin dataset
    '''
    # TU-Berlin datapath
    args.data_path = os.path.join(args.data_path, 'TU-Berlin')

    # Getting the classes
    class_labels_directory = os.path.join(args.data_path, 'sketches')
    list_class = os.listdir(class_labels_directory)
    # Only folders
    list_class = [name for name in list_class if os.path.isdir(os.path.join(class_labels_directory, name))]
    nc = len(list_class)
    dicts_class = create_dict_texts(list_class)

    images_directory = os.path.join(args.data_path, 'images')
    im_per_class = [len(glob(os.path.join(images_directory, c, '*jpg'))) for c in list_class]
    possible_test = np.where(np.array(im_per_class) >= 400)[0]

    # Random Shuffle
    random.seed(args.seed)
    random.shuffle(possible_test)
    random.shuffle(list_class)

    test_class = [list_class[possible_test[i]] for i in range(int(0.12 * nc))]
    list_class = [x for x in list_class if x not in test_class]

    # Dividing the classes
    train_class = list_class[:int(0.8 * nc)]
    valid_class = list_class[int(0.8 * nc):]

    with open(os.path.join(args.save, 'train.txt'), 'w') as fp:
        for item in train_class:
            fp.write("%s\n" % item)
    with open(os.path.join(args.save, 'valid.txt'), 'w') as fp:
        for item in valid_class:
            fp.write("%s\n" % item)
    with open(os.path.join(args.save, 'test.txt'), 'w') as fp:
        for item in test_class:
            fp.write("%s\n" % item)

    # Data Loaders
    train_loader = TUBerlin_Extended_train(
        args, train_class, dicts_class, transform)
    valid_sk_loader = TUBerlin_Extended_valid_test(
        args, valid_class, dicts_class, transform, type_skim='sketch')
    valid_im_loader = TUBerlin_Extended_valid_test(
        args, valid_class, dicts_class, transform, type_skim='images')
    test_sk_loader = TUBerlin_Extended_valid_test(
        args, test_class, dicts_class, transform, type_skim='sketch')
    test_im_loader = TUBerlin_Extended_valid_test(
        args, test_class, dicts_class, transform, type_skim='images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class TUBerlin_Extended_valid_test(data.Dataset):
        '''
    Custom dataset for TU-Berlin's validation and testing
    '''
    def __init__(self, args, set_class, dicts_class, transform=None, type_skim='images'):
        self.transform = transform
        self.set_class = set_class
        self.dicts_class = dicts_class
        self.type_skim = type_skim

        if type_skim == 'images':
            self.dir_file = os.path.join(args.data_path, 'images')
        elif type_skim == 'sketch':
            self.dir_file = os.path.join(args.data_path, 'sketches')
        else:
            NameError(type_skim + ' not implemented!')

        self.fnames, self.cls = get_file_list(self.dir_file, self.set_class, type_skim)
        self.loader = default_image_loader
        self.loader_image = default_image_loader_tuberlin

    def __getitem__(self, index):
        label = self.cls[index]
        fname = os.path.join(self.dir_file, label, self.fnames[index])
        if self.type_skim == 'images':
            photo = self.transform(self.loader_image(fname))
        else:
            photo = self.transform(self.loader(fname))

        lbl = self.dicts_class.get(label)

        return photo, fname, lbl

    def __len__(self):
        # Number of sketches/images in the dataset
        return len(self.fnames)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.set_class


class TUBerlin_Extended_train(data.Dataset):
    '''
    Custom dataset for TU-Berlin's training
    '''
    def __init__(self, args, train_class, dicts_class, transform=None):

        self.transform = transform
        self.train_class = train_class
        self.dicts_class = dicts_class

        self.dir_image = os.path.join(args.data_path, 'images')
        self.dir_sketch = os.path.join(args.data_path, 'sketches')
        self.loader = default_image_loader
        self.loader_image = default_image_loader_tuberlin
        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.train_class, 'sketch')

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
        # Read sketch
        fname = os.path.join(self.dir_sketch, self.cls_sketch[index], self.fnames_sketch[index])
        sketch = self.loader(fname)
        sketch = self.transform(sketch)

        # Target
        label = self.cls_sketch[index]
        lbl_pos = self.dicts_class.get(label)

        # Positive image
        fname = get_random_file_from_path(os.path.join(self.dir_image, label))
        image_pos = self.transform(self.loader_image(fname))

        # Negative class
        # Hard negative
        possible_classes = [x for x in self.train_class if x != label]
        label_neg = np.random.choice(possible_classes, 1)[0]
        lbl_neg = self.dicts_class.get(label_neg)

        fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
        image_neg = self.transform(self.loader_image(fname))

        return sketch, image_pos, image_neg, lbl_pos, lbl_neg

    def __len__(self):
        # Number of sketches/images in the dataset
        return len(self.fnames_sketch)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.train_class
