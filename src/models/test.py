import numpy as np
import time
import multiprocessing

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data.loader_factory import load_data
from src.options import Options
from src.models.encoder import EncoderCNN
from src.models.metrics import get_similarity, compare_classes, get_map_prec_200, get_map_all
from src.models.utils import load_model, save_qualitative_results


def get_test_data(data_loader, model, args):
    '''
    Get features, paths and target class of all images (or sketches) of data loader
    Args:
        - data_loader: loader of the validation or test set
        - model: encoder from images (or sketches) to embeddings
    Return:
        - fnames: list of the path to the images (or sketches)
        - embeddings: list of the associated embeddings
        - classes: list of the associated target classes
    '''
    fnames = []
    for i, (image, fname, target) in enumerate(data_loader):
        # Data to Variable
        if args.cuda:
            image, target = image.cuda(), target.cuda()

        # Process
        out_features, _ = model(image)

        # Filename of the images for qualitative
        fnames.append(fname)

        if i == 0:
            embeddings = out_features.cpu().data.numpy()
            classes = target.cpu().data.numpy()
        else:
            embeddings = np.concatenate((embeddings, out_features.cpu().data.numpy()), axis=0)
            classes = np.concatenate((classes, target.cpu().data.numpy()), axis=0)

    return fnames, embeddings, classes


def test(im_loader, sk_loader, model, args, dict_class=None):
    '''
    Get data and computes metrics on the model
    '''
    # Start counting time
    end = time.time()

    # Switch to test mode
    im_net, sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)

    im_fnames, im_embeddings, im_class = get_test_data(im_loader, im_net, args)
    sk_fnames, sk_embeddings, sk_class = get_test_data(sk_loader, sk_net, args)

    # Similarity
    similarity = get_similarity(sk_embeddings, im_embeddings)
    class_matches = compare_classes(im_class, sk_class)

    # Precision and recall for top k
    # mprec_k, rec_k = preca_k(similarity, class_matches, k=5)

    # Mean average precision
    num_cores = min(multiprocessing.cpu_count(), 32)
    map_200, prec_200 = get_map_prec_200(similarity, class_matches, num_cores)
    ap_all, map_all = get_map_all(similarity, class_matches, num_cores)

    # Metrics for each class
    if dict_class is not None:
        dict_class = {v: k for k, v in dict_class.items()}
        diff_class = set(sk_class)
        for d_class in diff_class:
            ind = (sk_class == d_class)
            print('mAP {} class {}'.format(str(np.array(ap_all)[ind].mean()), dict_class[d_class]))
            # print('Recall {} class {}'.format(str(np.array(rec_k)[ind].mean()), dict_class[d_class]))

    if args.plot:
        save_qualitative_results(similarity, class_matches, sk_fnames, im_fnames, args)

    # Measure elapsed time
    batch_time = time.time() - end

    print('* mAP {mean_ap:.3f}; Avg Time x Batch {b_time:.3f}'.format(mean_ap=map_all, b_time=batch_time))
    print('* mAP@200 {mean_ap_200:.3f}; Avg Time x Batch {b_time:.3f}'.format(mean_ap_200=map_200, b_time=batch_time))
    print('* Precision@200 {prec_200:.3f}; Avg Time x Batch {b_time:.3f}'.format(prec_200=prec_200, b_time=batch_time))
    return map_all, map_200, prec_200


def main():
    '''
    Full testing pipeline
    '''
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
    im_net, sk_net = load_model(args.load, im_net, sk_net)

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
