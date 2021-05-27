import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.constants import DICT_CLASS, EMB_ARRAY, EMBEDDINGS, METADATA, SKETCHY, QUICKDRAW, TUBERLIN, SKTU, SKTUQD
from src.data.loader_factory import load_data
from src.models.test import get_test_data
from src.models.utils import get_model, get_parameters


def save_embeddings(args, fnames, embeddings, classes, dataset_type):
    """
    Saves the precomputed image data in the same folder as the model in a subfolder called 'precomputed_embeddings'
    Args:
        - args: arguments reveived from the command line (argparse)
        - fnames: path of the images
        - embeddings: embeddings of the images
        - classes: classes of the images
        - dataset_type: validation or test set
    """
    df = pd.DataFrame(data=[fnames, classes]).T
    df.columns = ["fnames", "classes"]
    meta_path = os.path.join(
        args.embedding_path, args.dataset + "_" + dataset_type + METADATA
    )
    df.to_csv(meta_path, sep=" ", header=True)

    array_path = os.path.join(
        args.embedding_path, args.dataset + "_" + dataset_type + EMB_ARRAY
    )
    with open(array_path, "wb") as f:
        np.save(f, embeddings)


def get_test_images(args, data, im_net):
    """
    Loads all images path, embeddings and classes from a data loader
    Args:
        - args: arguments from the command line
        - data: dataloader of the image of the dataset
        - im_net: image model
    Return:
        - fnames: list of path to the images
        - embeddings: array of embeddings [NxE] with N the number of images and E the embedding dimension
        - classes: list of classes associated to each image
    """
    loader = DataLoader(
        data,
        batch_size=1,
        num_workers=args.prefetch,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    fnames, embeddings, classes = get_test_data(loader, im_net, args)
    fnames = [fname[0] for fname in fnames]
    return fnames, embeddings, classes


def save_class_dict(args, dict_class):
    """ Saves the dictionnary mapping classes to numbers """
    dict_path = os.path.join(args.embedding_path, args.dataset + DICT_CLASS)
    dict_class = {v: k for k, v in dict_class.items()}
    with open(dict_path, "w") as fp:
        json.dump(dict_class, fp)


def preprocess_embeddings(args, im_net):
    """
    Loads all the validation and testing data from a dataset, precompute the embeddings and
    saves the data for future inference.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, valid_im_data], [_, test_im_data], dicts_class = load_data(args, transform)
    save_class_dict(args, dicts_class)

    print("Valid")
    valid_fnames, valid_embeddings, valid_classes = get_test_images(
        args, valid_im_data, im_net
    )
    save_embeddings(args, valid_fnames, valid_embeddings, valid_classes, "valid")

    print("Test")
    test_fnames, test_embeddings, test_classes = get_test_images(
        args, test_im_data, im_net
    )
    save_embeddings(args, test_fnames, test_embeddings, test_classes, "test")


if __name__ == "__main__":
    # Get parameters
    args = get_parameters()
    args.pin_memory = args.cuda

    # Path to store embeddings
    args.embedding_path = os.path.join(args.save, EMBEDDINGS)
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)

    # Load image model
    im_net, _ = get_model(args, args.save + "/checkpoint.pth")
    im_net.eval()
    torch.set_grad_enabled(False)

    # Compute embeddings on chosen dataset(s)
    dataset = args.dataset
    if dataset in [SKETCHY, TUBERLIN, QUICKDRAW]:
        preprocess_embeddings(args, im_net)

    elif dataset in [SKTU, SKTUQD]:
        args.dataset = SKETCHY
        preprocess_embeddings(args, im_net)

        args.dataset = TUBERLIN
        preprocess_embeddings(args, im_net)

        if dataset == SKTUQD:
            args.dataset = QUICKDRAW
            preprocess_embeddings(args, im_net)
    else:
        raise Exception(args.dataset + " not implemented.")
