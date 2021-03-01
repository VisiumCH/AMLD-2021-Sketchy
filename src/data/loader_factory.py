import sys

import numpy as np
import torch

from src.data.both_extended import Both_Extended
from src.data.sketchy_extended import Sketchy_Extended
from src.data.tuberlin_extended import TUBerlin_Extended


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
    if args.dataset == "sketchy_extend":
        return Sketchy_Extended(args, transform)
    elif args.dataset == "tuberlin_extend":
        return TUBerlin_Extended(args, transform)
    elif args.dataset == "both":
        return Both_Extended(args, transform)
    else:
        sys.exit()


if __name__ == "__main__":
    """
    For experimentation purposes during implementation
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

    transform = transforms.Compose([transforms.ToTensor()])
    (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        dict_class,
    ) = load_data(args, transform)

    # test = load_data(args, transform)
    # print(len(test))

    print("\n--- Train Data ---")
    print("\t* Length: {}".format(len(train_loader)))
    #print("\t* Classes: {}".format(train_loader.get_class_dict()))
    print("\t* Num Classes. {}".format(len(train_loader.get_class_dict())))

    num_samples = 7
    rand_samples = np.random.randint(0, high=len(train_loader), size=num_samples)
    f, axarr = plt.subplots(3, num_samples)
    for i in range(len(rand_samples)):
        print(rand_samples[i])
        if args.dataset == "both":
            if rand_samples[i] < train_loader.sketchy_limit:
                dataset_dict_class = dict_class[0]
            else:
                dataset_dict_class = dict_class[1]
        else:
            dataset_dict_class = dict_class
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

    print("\n--- Valid Data ---")
    print("\t* Length Sketch: {}".format(len(valid_sk_loader)))
    print("\t* Length Image: {}".format(len(valid_im_loader)))
    print("\t* Classes: {}".format(valid_sk_loader.get_class_dict()))
    print("\t* Num Classes. {}".format(len(valid_sk_loader.get_class_dict())))

    print("\n--- Test Data ---")
    print("\t* Length Sketch: {}".format(len(test_sk_loader)))
    print("\t* Length Image: {}".format(len(test_im_loader)))
    print("\t* Classes: {}".format(test_sk_loader.get_class_dict()))
    print("\t* Num Classes. {}".format(len(test_sk_loader.get_class_dict())))
