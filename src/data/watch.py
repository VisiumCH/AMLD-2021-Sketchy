import os
from glob import glob
import torch.utils.data as data

from src.data.parent_dataset import ParentDataset
from src.data.utils import get_file_list, default_image_loader, dataset_split, create_dict_texts


class Watch(data.Dataset):
    '''
    Custom dataset for watches
    '''

    def __init__(self, args, transform):

        self.transform = transform
        self.loader_image = default_image_loader

        class_directories = glob(os.path.join(args.data_path, "Watch", "*/"))
        list_class = [class_path.split("/")[-2] for class_path in class_directories]
        self.dicts_class = create_dict_texts(list_class)

        self.dir_image = os.path.join(args.data_path, "Watch")

        self.fnames_image, self.cls_images = get_file_list(
            self.dir_image, self.dicts_class, "sketch")  # sketch as it is png

    def __getitem__(self, index):
        label = self.cls_images[index]
        fname = os.path.join(self.dir_image, label, self.fnames_image[index])
        photo = self.transform(self.loader_image(fname))
        lbl = self.dicts_class.get(label)
        return photo, fname, lbl

    def __len__(self):
        return len(self.fnames_image)

    def get_class_dict(self):
        return self.dicts_class
