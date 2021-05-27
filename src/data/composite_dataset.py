import random
import numpy as np

from src.data.sktuqd_dataset import SkTuQd
from src.data.sktu_dataset import SkTu
from src.data.utils import get_class_dict, dataset_split

from src.constants import SKETCHY, QUICKDRAW, TUBERLIN, FOLDERS, SKTU, SKTUQD


def make_composite_dataset(args, transform, dataset_name):
    """
    Creates all the data loaders for training with Sketchy, TU-Berlin and Quickdraw datasets
    Args:
        - args: arguments reveived from the command line (argparse)
        - transform: pytorch transform to apply on the data
        - dataset_name: name of composite dataset
    Return:
        - train_loader: data loader for the training set
        - valid_sk_loader: data loader of sketches for the validation set
        - valid_im_loader: data loader of images for the validation set
        - test_sk_loader: data loader of sketches for the test set
        - test_im_loader: data loader of images for the test set
        - dicts_class: dictionnnary mapping number to classes.
                        The key is a unique number and the value is the class name.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    if dataset_name == SKTU:
        dataset_class = SkTu
        datasets_list = [SKETCHY, TUBERLIN]
    elif dataset_name == SKTUQD:
        dataset_class = SkTuQd
        datasets_list = [SKETCHY, TUBERLIN, QUICKDRAW]
    else:
        raise Exception(f"Composite dataset only possible with {SKTU} or {SKTUQD}.\nHere {dataset_name}")

    # Get Sketchy, TU-Berlin and Quickdraw datasets
    dicts_class, train_data, valid_data, test_data = get_multiple_datasets(args, datasets_list)

    # Data Loaders
    train_loader, valid_loader, test_loader = get_dataset_loaders(args, transform, dataset_class, dicts_class, train_data, valid_data, test_data)

    if dataset_name == SKTU:
        dicts_class = [dicts_class[0], dicts_class[1]]
    elif dataset_name == SKTUQD:
        dicts_class = [dicts_class[0], dicts_class[1], dicts_class[2]]

    return train_loader, valid_loader, test_loader, dicts_class


def get_multiple_datasets(args, datasets_list):
    """ In case of training with multiple datasets, prepare all datasets and put them in list """
    dicts_class, train_data, valid_data, test_data = [], [], [], []
    for dataset in datasets_list:
        dict_class = get_class_dict(args, FOLDERS[dataset])
        train_dataset, valid_dataset, test_dataset = dataset_split(
            args, FOLDERS[dataset], args.training_split, args.valid_split
        )
        dicts_class.append(dict_class)
        train_data.append(train_dataset)
        valid_data.append(valid_dataset)
        test_data.append(test_dataset)
    
    return dicts_class, train_data, valid_data, test_data


def get_dataset_loaders(args, transform, data_class, dicts_class, train_data, valid_data, test_data):
    """ Get the loader of the training, validation and test set for sketch and images """
    train_loader = data_class(args, "train", dicts_class, train_data, transform)
    valid_sk_loader = data_class(args, "valid", dicts_class, valid_data, transform, "sketches")
    valid_im_loader = data_class(args, "valid", dicts_class, valid_data, transform, "images")
    test_sk_loader = data_class(args, "test", dicts_class, test_data, transform, "sketches")
    test_im_loader = data_class(args, "test", dicts_class, test_data, transform, "images")

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader]
