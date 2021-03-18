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


def normalise_attention(attn, im):
    attn = nn.Upsample(size=(im[0].size(1), im[0].size(2)), mode='bilinear', align_corners=False)(attn)
    min_attn = attn.view((attn.size(0), -1)).min(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    max_attn = attn.view((attn.size(0), -1)).max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return (attn - min_attn) / (max_attn - min_attn)
