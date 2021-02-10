import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import time
import multiprocessing

# Own modules
from src.options import Options
from src.models.utils import load_checkpoint, save_qualitative_results
from src.models.metrics import get_similarity, precak, get_map_prec_200, get_map_all
from src.models.networks.encoder import EncoderCNN
from src.data.loader_factory import load_data


def get_test_data(data_loader, model, args):
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

    acc_fnames_im, acc_im_em, acc_cls_im = get_test_data(im_loader, im_net, args)
    acc_fnames_sk, acc_sk_em, acc_cls_sk = get_test_data(sk_loader, sk_net, args)

    sim, str_sim = get_similarity(acc_sk_em, acc_im_em, acc_cls_im, acc_cls_sk)

    # Precision and recall for top k
    mpreck, reck = precak(sim, str_sim, k=5)

    num_cores = min(multiprocessing.cpu_count(), 32)
    map_200, prec_200 = get_map_prec_200(sim, str_sim, num_cores)
    ap_all, map_all = get_map_all(sim, str_sim, num_cores)

    if dict_class is not None:
        dict_class = {v: k for k, v in dict_class.items()}
        diff_class = set(acc_cls_sk)
        for d_class in diff_class:
            ind = acc_cls_sk == d_class
            print('mAP {} class {}'.format(str(np.array(ap_all)[ind].mean()), dict_class[d_class]))
            print('Recall {} class {}'.format(str(np.array(reck)[ind].mean()), dict_class[d_class]))

    if args.plot:
        save_qualitative_results(sim, str_sim, acc_fnames_sk, acc_fnames_im, args)

    # Measure elapsed time
    batch_time = time.time() - end

    print('* mAP {mean_ap:.3f}; Avg Time x Batch {b_time:.3f}'.format(mean_ap=map_all, b_time=batch_time))
    print('* mAP@200 {mean_ap_200:.3f}; Avg Time x Batch {b_time:.3f}'.format(mean_ap_200=map_200, b_time=batch_time))
    print('* Precision@200 {prec_200:.3f}; Avg Time x Batch {b_time:.3f}'.format(prec_200=prec_200, b_time=batch_time))
    return map_all, map_200, prec_200


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
    _, _, _ = test(test_im_loader, test_sk_loader, [im_net, sk_net], args, dict_class)


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
