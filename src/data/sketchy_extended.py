import os
import random
from glob import glob

import numpy as np
import torch.utils.data as data

from src.data.utils import (
    create_dict_texts,
    get_file_list,
    default_image_loader,
    get_random_file_from_path
)


def Sketchy_Extended(args, transform="None"):
    '''
    Creates all the data loaders for Sketchy dataset
    '''
    # Getting the classes
    class_directories = glob(os.path.join(args.data_path, "Sketchy/extended_photo/*/"))
    list_class = [class_path.split("/")[-2] for class_path in class_directories]
    dicts_class = create_dict_texts(list_class)

    # Random seed
    np.random.seed(args.seed)

    # Read test classes
    with open(os.path.join(args.data_path, "Sketchy/zeroshot_classes_sketchy.txt")) as fp:
        test_class = fp.read().splitlines()
    list_class = [x for x in list_class if x not in test_class]

    # Random Shuffle
    random.seed(args.seed)
    shuffled_list_class = list_class
    random.shuffle(shuffled_list_class)

    # Dividing the classes
    train_class = shuffled_list_class[: int(0.9 * len(shuffled_list_class))]
    valid_class = shuffled_list_class[int(0.9 * len(shuffled_list_class)):]

    # Data Loaders
    train_loader = Sketchy(args, 'train', train_class, dicts_class, transform)
    valid_sk_loader = Sketchy(args, 'valid', valid_class, dicts_class, transform, "sketch")
    valid_im_loader = Sketchy(args, 'valid', valid_class, dicts_class, transform, "images")
    test_sk_loader = Sketchy(args, 'test', test_class, dicts_class, transform, "sketch")
    test_im_loader = Sketchy(args, 'test', test_class, dicts_class, transform, "images")

    return (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        dicts_class,
    )


class Sketchy(data.Dataset):
    '''
    Custom dataset for Stetchy's training
    '''

    def __init__(self, args, dataset_type, set_class, dicts_class, transform=None, image_type=None):
        self.transform = transform
        self.dataset_type = dataset_type
        self.set_class = set_class
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.image_type = image_type

        self.dir_image = os.path.join(args.data_path, "Sketchy/extended_photo")
        self.dir_sketch = os.path.join(args.data_path, "Sketchy/sketch", "tx_000000000000")

        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch, self.set_class, "sketch")
        self.fnames_image, self.cls_image = get_file_list(self.dir_image, self.set_class, "images")

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
        sketch_fname = os.path.join(
            self.dir_sketch,
            self.cls_sketch[index],
            self.fnames_sketch[index],
        )
        sketch = self.loader(sketch_fname)
        sketch = self.transform(sketch)

        # Target
        label = self.cls_sketch[index]
        lbl_pos = self.dicts_class.get(label)

        # Positive image
        im_pos_fname = get_random_file_from_path(os.path.join(self.dir_image, label))
        image_pos = self.transform(self.loader(im_pos_fname))

        # Negative class
        possible_classes = [x for x in self.set_class if x != label]
        label_neg = np.random.choice(possible_classes, 1)[0]
        lbl_neg = self.dicts_class.get(label_neg)

        im_neg_fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
        image_neg = self.transform(self.loader(im_neg_fname))

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
