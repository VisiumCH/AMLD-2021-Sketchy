import os

import numpy as np
import torch.utils.data as data

from src.data.utils import default_image_loader, get_random_file_from_path


class ParentDataset(data.Dataset):
    '''
    Custom dataset for TU-Berlin's
    '''

    def __init__(self, dataset_type, set_class, dicts_class, transform=None, image_type=None):

        self.transform = transform
        self.dataset_type = dataset_type
        self.set_class = set_class
        self.dicts_class = dicts_class
        self.loader = default_image_loader
        self.image_type = image_type

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
