import sys

import numpy as np

from src.constants import SKETCHY, QUICKDRAW, TUBERLIN, FOLDERS, SKTU, SKTUQD, DATASETS, PROCESSED_PATH
from src.data.default_dataset import make_default_dataset
from src.data.composite_dataset import make_composite_dataset
from src.data.utils import get_dataset_dict, get_limits


def load_data(args, transform):
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
    if args.dataset == SKETCHY:
        return make_default_dataset(args, FOLDERS[SKETCHY], transform)
    elif args.dataset == TUBERLIN:
        return make_default_dataset(args, FOLDERS[TUBERLIN], transform)
    elif args.dataset == QUICKDRAW:
        return make_default_dataset(args, FOLDERS[QUICKDRAW], transform)
    elif args.dataset == SKTU:
        return make_composite_dataset(args, transform, SKTU)
    elif args.dataset == SKTUQD:
        return make_composite_dataset(args, transform, SKTUQD)
    else:
        print(args.dataset + " dataset not implemented. Exiting.")
        sys.exit()


def main(args):
    """
    Loads all datasets and enables to verify the implementation.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    for dataset in DATASETS:
        print(f"Visualising dataset {dataset}")
        args.dataset = dataset
        (
            train_loader,
            [valid_sk_loader, valid_im_loader],
            [test_sk_loader, test_im_loader],
            dict_class,
        ) = load_data(args, transform)
        print_dataset_information(
            args,
            train_loader,
            valid_sk_loader,
            valid_im_loader,
            test_sk_loader,
            test_im_loader,
        )
        visualise_dataset(args, train_loader, dict_class)


def print_dataset_information(
    args, train_loader, valid_sk_loader, valid_im_loader, test_sk_loader, test_im_loader
):
    """
    Loads dataset and print the number of images and sketches.
    Enable to verify the implementation.
    """
    print("\t* Length sketch train: {}".format(len(train_loader)))
    if args.dataset == SKTU:
        print(
            "\t* Length image train: {}".format(
                len(train_loader.sketchy.fnames_image)
                + len(train_loader.tuberlin.fnames_image)
            )
        )
    elif args.dataset == SKTUQD:
        print(
            "\t* Length image train: {}".format(
                len(train_loader.sketchy.fnames_image)
                + len(train_loader.tuberlin.fnames_image)
                + len(train_loader.quickdraw.fnames_image)
            )
        )
    else:
        print("\t* Length image train: {}".format(len(train_loader.fnames_image)))
    print("\t* Length sketch valid: {}".format(len(valid_sk_loader)))
    print("\t* Length image valid: {}".format(len(valid_im_loader)))
    print("\t* Length sketch test: {}".format(len(test_sk_loader)))
    print("\t* Length image test: {}".format(len(test_im_loader)))

    sketchy_limit, tuberlin_limit = get_limits(args.dataset, train_loader, "sketches")
    print("\t* Sketchy limit: {}".format(sketchy_limit))
    print("\t* Tuberlin limit: {}".format(tuberlin_limit))


def visualise_dataset(args, train_loader, dict_class):
    """
    Plots example of training triplet in the 'src/visualization/ folder'
    The first row is the sketch, the second row is the positive image and the third row the negative image.
    """
    sketchy_limit, tuberlin_limit = get_limits(args.dataset, train_loader, "sketches")

    num_samples = 7
    rand_samples = np.random.randint(0, high=len(train_loader), size=num_samples)
    f, axarr = plt.subplots(3, num_samples)
    for i in range(len(rand_samples)):
        dataset_dict_class = get_dataset_dict(
            dict_class, rand_samples[i], sketchy_limit, tuberlin_limit
        )
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
    plt.savefig(PROCESSED_PATH + args.dataset + ".png")


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
