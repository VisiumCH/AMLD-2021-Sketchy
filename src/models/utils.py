import os
import errno
import pickle

import torch
import numpy as np


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


def load_model(model_path):
    '''
    Load model parameters from a checkoint 
    '''
    checkpoint = load_checkpoint(moedl_path)

    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])
    criterion.load_state_dict(checkpoint['criterion'])
    epoch = checkpoint['epoch']
    best_map = checkpoint['best_map']

    print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=epoch, mean_ap=best_map))

    return im_net, sk_net, criterion, epoch, best_map


def save_qualitative_results(sim, str_sim, acc_fnames_sk, acc_fnames_im, args):
    '''Save images with close embeddings'''
    # Qualitative Results
    flatten_acc_fnames_sk = [item for sublist in acc_fnames_sk for item in sublist]
    flatten_acc_fnames_im = [item for sublist in acc_fnames_im for item in sublist]

    retrieved_im_fnames = []
    retrieved_im_true_false = []
    for i in range(0, sim.shape[0]):
        sorted_indx = np.argsort(sim[i, :])[::-1]
        retrieved_im_fnames.append(list(np.array(flatten_acc_fnames_im)[sorted_indx][:args.num_retrieval]))
        retrieved_im_true_false.append(list(np.array(str_sim[i])[sorted_indx][:args.num_retrieval]))

    with open('src/visualisation/sketches.pkl', 'wb') as f:
        pickle.dump([flatten_acc_fnames_sk], f)

    with open('src/visualisation/retrieved_im_fnames.pkl', 'wb') as f:
        pickle.dump([retrieved_im_fnames], f)

    with open('src/visualisation/retrieved_im_true_false.pkl', 'wb') as f:
        pickle.dump([retrieved_im_true_false], f)
