import sys

import numpy as np

from src.data.default_dataset import DefaultDataset_Extended
from src.data.sktuqd_extended import SkTuQd_Extended
from src.data.sktu_extended import SkTu_Extended
from src.data.constants import DatasetName, DatasetFolder, ImageType
from src.data.utils import get_dataset_dict, get_limits


def load_data(args, transform=None):
    """
    Load the data of the appropriate dataset
    Args:
        - args: metadata provided in src/options.py
        - transform: pytorch transform to process the data
    Return:
        - train_loader: training dataset class
        - valid_sk_loader: sketch validation dataset class
        - valid_im_loader: image validation dataset class
        - test_sk_loader: sketch test dataset class
        - test_im_loader: image test dataset class
        - dicts_class: Ordered dictionnary {class_name, value}
    """
    if args.dataset == DatasetName.sketchy:
        return DefaultDataset_Extended(args, DatasetFolder.sketchy, transform)
    elif args.dataset == DatasetName.tuberlin:
        return DefaultDataset_Extended(args, DatasetFolder.tuberlin, transform)
    elif args.dataset == DatasetName.quickdraw:
        return DefaultDataset_Extended(args, DatasetFolder.quickdraw, transform)
    elif args.dataset == DatasetName.sktu:
        return SkTu_Extended(args, transform)
    elif args.dataset == DatasetName.sktuqd:
        return SkTuQd_Extended(args, transform)
    else:
        print(args.dataset + ' dataset not implemented. Exiting.')
        sys.exit()


def main(args):
    '''
    Loads all datasets and enables to verify the implementation.
    '''
    transform = transforms.Compose([transforms.ToTensor()])

    list_dataset = [DatasetName.sketchy, DatasetName.tuberlin, DatasetName.quickdraw,
                    DatasetName.sktu, DatasetName.sktuqd]
    for dataset in list_dataset:
        args.dataset = dataset

        (train_loader,
         [valid_sk_loader, valid_im_loader],
         [test_sk_loader, test_im_loader],
         dict_class,
         ) = load_data(args, transform)

        print(f'Dataset {dataset}')
        dataset_information(args, train_loader, valid_sk_loader, valid_im_loader, test_sk_loader, test_im_loader)
        dataset_visualisation(args, train_loader, dict_class)


def dataset_information(args, train_loader, valid_sk_loader, valid_im_loader, test_sk_loader, test_im_loader):
    '''
    Loads all datasets and print the length of each of their fields.
    Enable to verify the implementation.
    '''
    print("\t* Length sketch train: {}".format(len(train_loader)))
    if args.dataset == DatasetName.sktu:
        print("\t* Length image train: {}".format(len(train_loader.sketchy.fnames_image) +
                                                  len(train_loader.tuberlin.fnames_image)))
    elif args.dataset == DatasetName.sktuqd:
        print("\t* Length image train: {}".format(len(train_loader.sketchy.fnames_image) +
                                                  len(train_loader.tuberlin.fnames_image) +
                                                  len(train_loader.quickdraw.fnames_image)))
    else:
        print("\t* Length image train: {}".format(len(train_loader.fnames_image)))
    print("\t* Length sketch valid: {}".format(len(valid_sk_loader)))
    print("\t* Length image valid: {}".format(len(valid_im_loader)))
    print("\t* Length sketch test: {}".format(len(test_sk_loader)))
    print("\t* Length image test: {}".format(len(test_im_loader)))

    sketchy_limit, tuberlin_limit = get_limits(args.dataset, train_loader, ImageType.sketch)
    print("\t* Sketchy limit: {}".format(sketchy_limit))
    print("\t* Tuberlin limit: {}".format(tuberlin_limit))


def dataset_visualisation(args, train_loader, dict_class):
    '''
    Plots example of training triplet in the 'src/visualization/ folder'
    The first row is the sketch, the second row is the positive image and the third row the negative image.
    '''
    sketchy_limit, tuberlin_limit = get_limits(args.dataset, train_loader, ImageType.sketch)

    num_samples = 7
    rand_samples = np.random.randint(0, high=len(train_loader), size=num_samples)
    f, axarr = plt.subplots(3, num_samples)
    for i in range(len(rand_samples)):
        dataset_dict_class = get_dataset_dict(dict_class, rand_samples[i], sketchy_limit, tuberlin_limit)
        sk, im, im_neg, lbl, lbl_neg = train_loader[rand_samples[i]]
        axarr[0, i].imshow(sk.permute(1, 2, 0).numpy())
        axarr[0, i].set_title(dict_by_value(dataset_dict_class, lbl))
        axarr[0, i].axis("off")

        axarr[1, i].imshow(im.permute(1, 2, 0).numpy())
        axarr[1, i].axis("off")

        axarr[2, i].imshow(im_neg.permute(1, 2, 0).numpy())
        axarr[2, i].set_title(dict_by_value(dataset_dict_class, lbl_neg))
        axarr[2, i].axis("off")
    plt.show()
    plt.savefig("src/visualization/training_samples_" + args.dataset + ".png")


if __name__ == "__main__":
    """
    For experimentation and verification purposes during implementation
    """
    from src.options import Options
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # find key (class name) based on value
    def dict_by_value(dic, value):
        return list(dic.keys())[list(dic.values()).index(value)]

    # Parse options
    args = Options().parse()
    print("Parameters:\t" + str(args))

    main(args)
