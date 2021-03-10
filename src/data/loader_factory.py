import sys

import numpy as np

from src.data.sktuqd_extended import SkTuQd_Extended
from src.data.sktu_extended import SkTu_Extended
from src.data.watch_dataset.watch_dataset import Watch_Extended
from src.models.utils import get_dataset_dict, get_limits
from src.data.default_dataset import DefaultDataset_Extended


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
    if args.dataset == "sketchy":
        return DefaultDataset_Extended(args, "Sketchy", transform)
    elif args.dataset == "tuberlin":
        return DefaultDataset_Extended(args, "TU-Berlin", transform)
    elif args.dataset == "quickdraw":
        return DefaultDataset_Extended(args, "Quickdraw", transform)
    elif args.dataset == "sk+tu":
        return SkTu_Extended(args, transform)
    elif args.dataset == "sk+tu+qd":
        return SkTuQd_Extended(args, transform)
    elif args.dataset == "watch":
        return Watch_Extended(args, transform)
    else:
        print(args.dataset + ' dataset not implemented. Exiting.')
        sys.exit()


def dataset_details(args, transform):

    list_dataset = ["sketchy", "tuberlin", "sk+tu", "quickdraw", "sk+tu+qd", "watch"]
    for dataset in list_dataset:
        args.dataset = dataset

        (train_loader,
         [valid_sk_loader, valid_im_loader],
         [test_sk_loader, test_im_loader],
         dict_class,
         ) = load_data(args, transform)

        print(f'Dataset {dataset}')
        print("\t* Length sketch train: {}".format(len(train_loader)))
        if dataset == "sk+tu":
            print("\t* Length image train: {}".format(len(train_loader.sketchy.fnames_image) +
                                                      len(train_loader.tuberlin.fnames_image)))
        elif dataset == "sk+tu+qd":
            print("\t* Length image train: {}".format(len(train_loader.sketchy.fnames_image) +
                                                      len(train_loader.tuberlin.fnames_image) +
                                                      len(train_loader.quickdraw.fnames_image)))
        else:
            print("\t* Length image train: {}".format(len(train_loader.fnames_image)))
        print("\t* Length sketch valid: {}".format(len(valid_sk_loader)))
        print("\t* Length image valid: {}".format(len(valid_im_loader)))
        print("\t* Length sketch test: {}".format(len(test_sk_loader)))
        print("\t* Length image test: {}".format(len(test_im_loader)))

        # Plot check
        sketchy_limit, tuberlin_limit = get_limits(args.dataset, train_loader, 'sketch')
        num_samples = 5
        rand_samples = np.random.randint(0, high=len(train_loader), size=num_samples)
        f, axarr = plt.subplots(3, num_samples)
        for i in range(len(rand_samples)):
            dataset_dict_class = get_dataset_dict(dict_class, rand_samples[i], sketchy_limit, tuberlin_limit)
            sk, im, im_neg, lbl, lbl_neg = train_loader[rand_samples[i]]

            title_pos = dict_by_value(dataset_dict_class, lbl)
            title_neg = dict_by_value(dataset_dict_class, lbl_neg)
            fontsize = 12
            plt.suptitle(args.dataset.upper(), fontsize=fontsize)

            if args.dataset == "watch":
                meta_pos = title_pos.split('__')
                title_pos = f'{meta_pos[0]} \n{meta_pos[1]} \n{meta_pos[2]}'
                meta_neg = title_neg.split('__')
                title_neg = f'{meta_neg[0]} \n{meta_neg[1]} \n{meta_neg[2]}'
                fontsize = 6
                plt.suptitle('Watches: Brand, model, variant', fontsize=10)

            axarr[0, i].imshow(sk.permute(1, 2, 0).numpy())
            axarr[0, i].set_title(title_pos, fontsize=fontsize)
            axarr[0, i].axis("off")

            axarr[1, i].imshow(im.permute(1, 2, 0).numpy())
            axarr[1, i].set_title(title_pos, fontsize=fontsize)
            axarr[1, i].axis("off")

            axarr[2, i].imshow(im_neg.permute(1, 2, 0).numpy())
            axarr[2, i].set_title(title_neg, fontsize=fontsize)
            axarr[2, i].axis("off")
        plt.show()
        plt.savefig("src/visualization/training_samples_" + args.dataset + ".png")


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
    transform = transforms.Compose([transforms.ToTensor()])

    dataset_details(args, transform)
