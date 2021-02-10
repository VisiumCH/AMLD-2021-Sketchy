import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed
import pickle

# Own modules
from src.options import Options
from src.models.utils import load_checkpoint
from src.models.metrics import recall, precak
from src.models.networks.encoder import EncoderCNN
from src.data.loader_factory import load_data


def get_test_data(data_loader, model):
    '''
    Get features, paths and target class of all images (or sketches) of data loader
    Args:
        - data_loader: loader of the validation or test set
        - model: encoder from images (or sketches) to embeddings
    Return:
        - acc_fnames: list of the path to the images (or sketches)
        - acc_embeddings: list of the associated embeddings 
        - acc_class: list of the associated target classes
    '''
    acc_fnames = []
    for i, (image, fname, target) in enumerate(data_loader):
        # Data to Variable
        if args.cuda:
            image, target = image.cuda(), target.cuda()

        # Process
        out_features, _ = model(image)

        # Filename of the images for qualitative
        acc_fnames.append(fname)

        if i == 0:
            acc_embeddings = out_features.cpu().data.numpy()
            acc_class = target.cpu().data.numpy()
        else:
            acc_embeddings = np.concatenate((acc_embeddings, out_features.cpu().data.numpy()), axis=0)
            acc_class = np.concatenate((acc_class, target.cpu().data.numpy()), axis=0)

    return acc_fnames, acc_embeddings, acc_class


def test(im_loader, sk_loader, model, args, dict_class=None):
    # Start counting time
    end = time.time()

    # Switch to test mode
    im_net, sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)

    acc_fnames_im, acc_im_em, acc_cls_im = get_test_data(im_loader, im_net)
    acc_fnames_sk, acc_sk_em, acc_cls_sk = get_test_data(sk_loader, sk_net)

    num_cores = min(multiprocessing.cpu_count(), 32)

    mpreck, reck = get_precision_and_recall(acc_sk_em, acc_im_em, num_cores)

    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    map_ = np.mean(aps)

    if dict_class is not None:
        dict_class = {v: k for k, v in dict_class.items()}
        diff_class = set(acc_cls_sk)
        for d_class in diff_class:
            ind = acc_cls_sk == d_class
            print('mAP {} class {}'.format(str(np.array(aps)[ind].mean()), dict_class[d_class]))
            print('Recall {} class {}'.format(str(np.array(reck)[ind].mean()), dict_class[d_class]))

    if args.plot:
        # Qualitative Results
        flatten_acc_fnames_sk = [item for sublist in acc_fnames_sk for item in sublist]
        flatten_acc_fnames_im = [item for sublist in acc_fnames_im for item in sublist]

        retrieved_im_fnames = []
        # Just a try
        retrieved_im_true_false = []
        for i in range(0, sim.shape[0]):
            sorted_indx = np.argsort(sim[i, :])[::-1]
            retrieved_im_fnames.append(list(np.array(flatten_acc_fnames_im)[sorted_indx][:args.num_retrieval]))
            # Just a try
            retrieved_im_true_false.append(list(np.array(str_sim[i])[sorted_indx][:args.num_retrieval]))

        with open('src/visualisation/sketches.pkl', 'wb') as f:
            pickle.dump([flatten_acc_fnames_sk], f)

        with open('src/visualisation/retrieved_im_fnames.pkl', 'wb') as f:
            pickle.dump([retrieved_im_fnames], f)

        with open('src/visualisation/retrieved_im_true_false.pkl', 'wb') as f:
            pickle.dump([retrieved_im_true_false], f)

    # Measure elapsed time
    batch_time = time.time() - end

    print('* mAP {mean_ap:.3f}; Avg Time x Batch {b_time:.3f}'.format(mean_ap=map_, b_time=batch_time))
    return map_  # , map_200, precision_200


def main():
    print('Prepare data')
    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, _], [test_sk_data, test_im_data], dict_class = load_data(args, transform)

    test_sk_loader = DataLoader(test_sk_data, batch_size=3 * args.batch_size, num_workers=args.prefetch,
                                pin_memory=True)
    test_im_loader = DataLoader(test_im_data, batch_size=3 * args.batch_size, num_workers=args.prefetch,
                                pin_memory=True)

    print('Create model')
    im_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    print('Loading model')
    checkpoint = load_checkpoint(args.load)
    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])
    print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=checkpoint['epoch'],
                                                                    mean_ap=checkpoint['best_map']))

    print('***Test***')
    map_test = test(test_im_loader, test_sk_loader, [im_net, sk_net], args, dict_class)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')

    main()
