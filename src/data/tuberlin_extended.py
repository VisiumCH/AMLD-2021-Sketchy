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

    # Getting the classes
    class_labels_directory = os.path.join(args.data_path, 'TU-Berlin/sketches')
    list_class = os.listdir(class_labels_directory)
    # Only folders
    list_class = [name for name in list_class if os.path.isdir(os.path.join(class_labels_directory, name))]
    nc = len(list_class)
    dicts_class = create_dict_texts(list_class)

    images_directory = os.path.join(args.data_path, 'TU-Berlin/images')
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
    train_loader = TUBerlin(args, 'train', train_class, dicts_class, transform)
    valid_sk_loader = TUBerlin(args, 'valid', valid_class, dicts_class, transform, 'sketch')
    valid_im_loader = TUBerlin(args, 'valid', valid_class, dicts_class, transform, 'images')
    test_sk_loader = TUBerlin(args, 'test', test_class, dicts_class, transform, 'sketch')
    test_im_loader = TUBerlin(args, 'test', test_class, dicts_class, transform, 'images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class TUBerlin(data.Dataset):
    '''
    Custom dataset for TU-Berlin's
    '''

    def __init__(self, args, dataset_type, set_class, dicts_class, transform=None, image_type=None):

        self.transform = transform
        self.dataset_type = dataset_type
        self.set_class = set_class
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.loader_image = default_image_loader_tuberlin
        self.image_type = image_type

        self.dir_sketch = os.path.join(args.data_path, 'TU-Berlin/sketches')
        self.dir_image = os.path.join(args.data_path, 'TU-Berlin/images')

        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.set_class, 'sketch')
        self.fnames_image, self.cls_image = get_file_list(self.dir_image, self.set_class, 'images')

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

        # return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        if self.dataset_type == 'train':
            return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        else:
            if self.image_type == 'images':
                return image_pos, im_pos_fname, lbl_pos
            elif self.image_type == 'sketch':
                return sketch, sketch_fname, lbl_pos

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == 'train' or self.image_type == 'sketch':
            return len(self.fnames_sketch)
        else:
            return len(self.fnames_image)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.set_class
