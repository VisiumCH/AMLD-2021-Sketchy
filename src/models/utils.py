import argparse
import os
import errno

import torch
import torch.nn as nn

from src.constants import MODELS_PATH, PARAMETERS
from src.models.encoder import EncoderCNN


def get_parameters():
    """ Parse the arguments from the command line and default setup """
    parser = argparse.ArgumentParser(
        description="Sketch Based Retrieval Test and inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "name", type=str, help="Name of the training folder of the model to test."
    )
    argument = parser.parse_args()
    args = get_saved_params(MODELS_PATH + argument.name)
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    return args


def get_saved_params(save_folder):
    """
    Get the training parameters from the saved text file in the training folder
    Args:
        - save: folder of the desired training
    Return:
        - namespace of the arguments
    """
    d = {}
    d["save"] = save_folder
    d["load"] = save_folder + "/checkpoint.pth"

    with open(d["save"] + "/" + PARAMETERS) as f:
        for line in f:
            line = line.rstrip("\n")
            (key, val) = line.split(" ")

            try:
                if "." in val:
                    d[key] = float(val)  # float
                else:
                    d[key] = int(val)  # int
            except ValueError:
                d[key] = val  # string

    args = argparse.Namespace(**d)

    return args


def save_checkpoint(state, directory, file_name):
    """
    Saves the checkpoint of the best trained model with its state
    Args:
        - state: dict state of the checkpoint (epoch, map, criterion and networks variable)
        - directory: directory were to same the checkpoint
        - file_name: name of the checkpoint
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + ".pth")
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    """
    Loads the checkpoint (networks variables + state)  stored in model_file
    Args:
        - model_file: path to the checkpoint
    Return:
        - a checkpoint
    """
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        print(
            "=> loaded model '{}' (epoch {}, map {})".format(
                model_file, checkpoint["epoch"], checkpoint["best_map"]
            )
        )
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def load_model(model_path, im_net, sk_net, criterion=None):
    """
    Load model parameters in the network from a checkpoint
    Args:
        - model_path: path to the checkpoint
        - im_net: images network
        - sk_net: sketches network
        - criterion: training loss
    Return:
        - im_net: images network with the weight loaded from the checkpoint
        - sk_net: sketches network with the weight loaded from the checkpoint
        - criterion: criterion from the checkpoint
        - epoch: epoch at which the checkpoint was saved
        - best_map: mean average precision of model saved in checkpoint
    """
    checkpoint = load_checkpoint(model_path)

    im_net.load_state_dict(checkpoint["im_state"])
    sk_net.load_state_dict(checkpoint["sk_state"])
    epoch = checkpoint["epoch"]
    best_map = checkpoint["best_map"]
    print(
        "Loaded model at epoch {epoch} and mAP {mean_ap}%".format(
            epoch=epoch, mean_ap=best_map
        )
    )

    if criterion:
        # if training
        criterion.load_state_dict(checkpoint["criterion"])
        return im_net, sk_net, criterion, epoch, best_map
    else:  # testing
        return im_net, sk_net


def get_model(args, best_checkpoint):
    """
    Load a trained model.
    Get architecture and loads weights, criterion and information from the 'best checkpoint'.
    Args:
        - args: arguments to define structure of network
        - best_checkpoint:
    Return:
        - im_net: images network with the weight loaded from the checkpoint
        - sk_net: sketches network with the weight loaded from the checkpoint
    """
    im_net = EncoderCNN(out_size=args.emb_size, attention=True)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=True)

    if args.cuda:
        checkpoint = torch.load(best_checkpoint)
    else:
        checkpoint = torch.load(best_checkpoint, map_location="cpu")

    im_net.load_state_dict(checkpoint["im_state"])
    sk_net.load_state_dict(checkpoint["sk_state"])

    if args.cuda and args.ngpu > 1:
        print("\t* Data Parallel **NOT TESTED**")
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print("\t* CUDA")
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    return im_net, sk_net


def normalise_attention(attn, im):
    """
    Gets the feature map of the attention and does a min-max normalisation (for further plots)
    """
    attn = nn.Upsample(
        size=(im[0].size(1), im[0].size(2)), mode="bilinear", align_corners=False
    )(attn)
    min_attn = (
        attn.view((attn.size(0), -1))
        .min(-1)[0]
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    max_attn = (
        attn.view((attn.size(0), -1))
        .max(-1)[0]
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    return (attn - min_attn) / (max_attn - min_attn)
