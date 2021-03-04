import os
import random
import numpy as np
import torch.utils.data as data
from src.data.tuberlin_extended import tuberlin_dataset_split
from src.data.sketchy_extended import sketchy_dataset_split
from src.data.utils import (
    get_file_list,
    default_image_loader,
    default_image_loader_tuberlin,
    get_random_file_from_path
)


def Both_Extended(args, transform="None"):
    '''
    Creates all the data loaders for Sketchy dataset
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sketchy and TU-Berlin
    train_class_sketchy, valid_class_sketchy, test_class_sketchy, dicts_class_sketchy = sketchy_dataset_split(args)
    train_class_tuberlin, valid_class_tuberlin, test_class_tuberlin, dicts_class_tuberlin = tuberlin_dataset_split(
        args)

    # Data Loaders
    train_loader = Both(args, 'train', train_class_sketchy, train_class_tuberlin,
                        dicts_class_sketchy, dicts_class_tuberlin, transform)
    valid_sk_loader = Both(args, 'valid', valid_class_sketchy, valid_class_tuberlin,
                           dicts_class_sketchy, dicts_class_tuberlin, transform, "sketch")
    valid_im_loader = Both(args, 'valid', valid_class_sketchy, valid_class_tuberlin,
                           dicts_class_sketchy, dicts_class_tuberlin, transform, "images")
    test_sk_loader = Both(args, 'test', test_class_sketchy, test_class_tuberlin,
                          dicts_class_sketchy, dicts_class_tuberlin, transform, "sketch")
    test_im_loader = Both(args, 'test', test_class_sketchy, test_class_tuberlin,
                          dicts_class_sketchy, dicts_class_tuberlin, transform, "images")
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
    Both_Extended(args)


class Both(data.Dataset):
    '''
    Custom dataset for Stetchy's training
    '''

    def __init__(self, args, dataset_type, set_class_sketchy, set_class_tuberlin,
                 dicts_class_sketchy, dicts_class_tuberlin,
                 transform=None, image_type=None):
        self.transform = transform
        self.dataset_type = dataset_type
        self.set_class_sketchy = set_class_sketchy
        self.set_class_tuberlin = set_class_tuberlin
        self.dicts_class_sketchy = dicts_class_sketchy
        self.dicts_class_tuberlin = dicts_class_tuberlin
        self.loader = default_image_loader
        self.loader_image_tuberlin = default_image_loader_tuberlin
        self.image_type = image_type

        # Sketchy data
        self.dir_image_sketchy = os.path.join(args.data_path, "Sketchy/extended_photo")
        self.dir_sketch_sketchy = os.path.join(args.data_path, "Sketchy/sketch", "tx_000000000000")
        self.fnames_sketch_sketchy, self.cls_sketch_sketchy = get_file_list(
            self.dir_sketch_sketchy, self.set_class_sketchy, "sketch")
        self.fnames_image_sketchy, self.cls_image_sketchy = get_file_list(
            self.dir_image_sketchy, self.set_class_sketchy, "images")

        # Tuberlin data
        self.dir_sketch_tuberlin = os.path.join(args.data_path, 'TU-Berlin/sketches')
        self.dir_image_tuberlin = os.path.join(args.data_path, 'TU-Berlin/images')
        self.fnames_sketch_tuberlin, self.cls_sketch_tuberlin = get_file_list(
            self.dir_sketch_tuberlin, self.set_class_tuberlin, 'sketch')
        self.fnames_image_tuberlin, self.cls_image_tuberlin = get_file_list(
            self.dir_image_tuberlin, self.set_class_tuberlin, 'images')

        # Separator between sketchy and tuberlin datasets
        self.sketchy_limit_sketch = len(self.fnames_sketch_sketchy)
        self.sketchy_limit_images = len(self.fnames_image_sketchy)

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
            dicts_class = self.dicts_class_sketchy
            dir_sketch = self.dir_sketch_sketchy
            cls_sketch = self.cls_sketch_sketchy
            fnames_sketch = self.fnames_sketch_sketchy
            dir_image = self.dir_image_sketchy
            cls_images = self.cls_image_sketchy
            fnames_image = self.fnames_image_sketchy
            set_class = self.set_class_sketchy
            loader_image = self.loader
        else:
            dicts_class = self.dicts_class_tuberlin
            dir_sketch = self.dir_sketch_tuberlin
            cls_sketch = self.cls_sketch_tuberlin
            fnames_sketch = self.fnames_sketch_tuberlin
            dir_image = self.dir_image_tuberlin
            cls_images = self.cls_image_tuberlin
            fnames_image = self.fnames_image_tuberlin
            set_class = self.set_class_tuberlin
            loader_image = self.loader_image_tuberlin

            if (self.image_type == 'sketch' or self.dataset_type == 'train'):
                index -= self.sketchy_limit_sketch
            elif self.image_type == 'images':
                index -= self.sketchy_limit_images

        if self.dataset_type == 'train':
            # Read sketch
            sketch_fname = os.path.join(dir_sketch, cls_sketch[index], fnames_sketch[index])
            sketch = self.transform(self.loader(sketch_fname))

            # Target
            label = cls_sketch[index]
            lbl_pos = dicts_class.get(label)

            # Positive image
            im_pos_fname = get_random_file_from_path(os.path.join(dir_image, label))
            image_pos = self.transform(loader_image(im_pos_fname))

            # Negative class
            # Hard negative
            possible_classes = [x for x in set_class if x != label]
            label_neg = np.random.choice(possible_classes, 1)[0]
            lbl_neg = dicts_class.get(label_neg)

            im_neg_fname = get_random_file_from_path(os.path.join(dir_image, label_neg))
            image_neg = self.transform(loader_image(im_neg_fname))

            return sketch, image_pos, image_neg, lbl_pos, lbl_neg
        else:
            if self.image_type == 'images':
                label = cls_images[index]
                fname = os.path.join(dir_image, label, fnames_image[index])
                photo = self.transform(loader_image(fname))

            elif self.image_type == 'sketch':
                label = cls_sketch[index]
                fname = os.path.join(dir_sketch, label, fnames_sketch[index])
                photo = self.transform(self.loader(fname))

            lbl = dicts_class.get(label)
            return photo, fname, lbl

    def __len__(self):
        # Number of sketches/images in the dataset
        if self.dataset_type == 'train' or self.image_type == 'sketch':
            return len(self.fnames_sketch_sketchy) + len(self.fnames_sketch_tuberlin)
        else:
            return len(self.fnames_image_sketchy) + len(self.fnames_image_tuberlin)

    def get_class_dict(self):
        # Dictionnary of categories of the dataset
        return self.set_class_sketchy, self.set_class_tuberlin