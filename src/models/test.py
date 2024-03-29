import numpy as np
import time
import multiprocessing
import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from src.data.loader_factory import load_data
from src.models.encoder import EncoderCNN
from src.models.metrics import (
    get_similarity,
    compare_classes,
    get_map_prec,
    get_map_all,
)
from src.models.utils import load_model, get_parameters


def get_test_data(data_loader, model, args):
    """
    Get features, paths and target class of all images (or sketches) of data loader
    Args:
        - data_loader: loader of the validation or test set
        - model: encoder from images (or sketches) to embeddings
    Return:
        - fnames: list of the path to the images (or sketches)
        - embeddings: list of the associated embeddings
        - classes: list of the associated target classes
    """
    if args.cuda:
        model = model.cuda()

    fnames = []
    for i, (image, fname, target) in enumerate(tqdm.tqdm(data_loader)):
        # Data to Variable
        if args.cuda:
            image, target = image.cuda(), target.cuda()

        # Process
        out_features, _ = model(image)

        # Filename of the images for qualitative
        fnames.append(fname)

        if args.cuda:
            out_features = out_features.cpu().data.numpy()
            target = target.cpu().data.numpy()

        if i == 0:
            embeddings = out_features
            classes = target
        else:
            embeddings = np.concatenate((embeddings, out_features), axis=0)
            classes = np.concatenate((classes, target), axis=0)

    return fnames, embeddings, classes


def get_part_of_test_data(data_loader, model, args, part_index=0):
    """
    Get part of features, paths and target class of images (or sketches) of data loader
    Args:
        - data_loader: loader of the validation or test set
        - model: encoder from images (or sketches) to embeddings
    Return:
        - fnames: list of the path to the images (or sketches)
        - embeddings: list of the associated embeddings
        - classes: list of the associated target classes
    """
    if args.cuda:
        model = model.cuda()

    images_indexes_min = part_index * args.num_test_batches
    images_indexes_max = (part_index + 1) * args.num_test_batches

    fnames = []
    for i, (image, fname, target) in enumerate(
        tqdm.tqdm(data_loader, total=args.num_test_batches)
    ):

        if i < images_indexes_min:
            continue
        elif i >= images_indexes_max:
            break

        # Data to Variable
        if args.cuda:
            image, target = image.cuda(), target.cuda()

        # Process
        out_features, _ = model(image)

        # Filename of the images for qualitative
        fnames.append(fname)

        if args.cuda:
            out_features = out_features.cpu().data.numpy()
            target = target.cpu().data.numpy()
        else:
            out_features = out_features.detach().numpy()

        if i == images_indexes_min:
            embeddings = out_features
            classes = target
        else:
            embeddings = np.concatenate((embeddings, out_features), axis=0)
            classes = np.concatenate((classes, target), axis=0)

    return fnames, embeddings, classes


def test(args, im_loader, sk_loader, model, inference_logger=None):
    """
    Get data and computes metrics on the model
    """
    # Start counting time
    end = time.time()

    # Switch to test mode
    im_net, sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)

    im_fnames, im_embeddings, im_class = get_test_data(im_loader, im_net, args)
    _, sk_embeddings, sk_class = get_test_data(sk_loader, sk_net, args)

    # Similarity
    similarity = get_similarity(sk_embeddings, im_embeddings)
    class_matches = compare_classes(im_class, sk_class)
    del sk_embeddings, im_embeddings

    # Mean average precision
    num_cores = min(multiprocessing.cpu_count(), 32)
    map, prec = get_map_prec(similarity, class_matches, num_cores, args.metric_limit)
    map_all = get_map_all(similarity, class_matches, num_cores)

    if inference_logger:
        inference_logger.plot_inference(similarity, im_fnames, im_class)

    # Measure elapsed time
    batch_time = time.time() - end
    print("Avg Time x Batch {b_time:.3f}".format(b_time=batch_time))
    print("* mAP {mean_ap:.3f}".format(mean_ap=map_all))
    print("* mAP@200 {mean_ap_200:.3f}".format(mean_ap_200=map))
    print("* Precision@200 {prec_200:.3f}".format(prec_200=prec))

    return map_all, map, prec


def main():
    """
    Full testing pipeline
    """
    print("Prepare data")
    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, _], [test_sk_data, test_im_data], _ = load_data(args, transform)

    data_args = {
        "batch_size": 3 * args.batch_size,
        "num_workers": args.prefetch,
        "pin_memory": True,
    }
    test_sk_loader = DataLoader(test_sk_data, **data_args)
    test_im_loader = DataLoader(test_im_data, **data_args)

    print("Create model")
    im_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)

    print("Check CUDA")
    if args.cuda and args.ngpu > 1:
        print("\t* Data Parallel **NOT TESTED**")
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print("\t* CUDA")
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    print("Loading model")
    im_net, sk_net = load_model(args.load, im_net, sk_net)

    print("***Test***")
    _, _, _ = test(
        args,
        test_im_loader, test_sk_loader,
        [im_net, sk_net],
        inference_logger=None,
    )


if __name__ == "__main__":
    # Parse options
    args = get_parameters()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    main()
