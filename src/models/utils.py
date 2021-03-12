import os
import errno

import torch
import torch.nn as nn

from src.models.encoder import EncoderCNN


def save_checkpoint(state, directory, file_name):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        print("=> loaded model '{}' (epoch {}, map {})".format(
            model_file, checkpoint['epoch'], checkpoint['best_map']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def load_model(model_path, im_net, sk_net, criterion=None):
    '''
    Load model parameters from a checkpoint
    '''
    checkpoint = load_checkpoint(model_path)

    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])
    epoch = checkpoint['epoch']
    best_map = checkpoint['best_map']
    print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=epoch, mean_ap=best_map))

    if criterion:
        # if training
        criterion.load_state_dict(checkpoint['criterion'])
        return im_net, sk_net, criterion, epoch, best_map
    else:  # testing
        return im_net, sk_net


def get_model(args, best_checkpoint):
    im_net = EncoderCNN(out_size=args.emb_size, attention=True)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=True)

    if args.cuda:
        checkpoint = torch.load(best_checkpoint)
    else:
        checkpoint = torch.load(best_checkpoint, map_location='cpu')

    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])

    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    return im_net, sk_net


def get_limits(dataset, valid_data, image_type):
    if dataset == 'sk+tu' or dataset == 'sk+tu+qd':
        if image_type == 'image':
            sketchy_limit = valid_data.sketchy_limit_images
        else:
            sketchy_limit = valid_data.sketchy_limit_sketch
    else:
        sketchy_limit = None

    if dataset == 'sk+tu+qd':
        if image_type == 'image':
            tuberlin_limit = valid_data.tuberlin_limit_images
        else:
            tuberlin_limit = valid_data.tuberlin_limit_sketch
    else:
        tuberlin_limit = None

    return sketchy_limit, tuberlin_limit


def get_dataset_dict(dict_class, idx, sketchy_limit, tuberlin_limit):

    if sketchy_limit is None:  # single dataset
        pass
    else:  # multiple datasets
        if idx < sketchy_limit:  # sketchy dataset
            dict_class = dict_class[0]
        else:
            if tuberlin_limit is None or idx < tuberlin_limit:  # tuberlin dataset
                dict_class = dict_class[1]
            else:  # quickdraw dataset
                dict_class = dict_class[2]

    return dict_class
