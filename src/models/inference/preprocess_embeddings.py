import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.loader_factory import load_data
from src.models.test import get_test_data
from src.models.utils import get_model
from src.options import Options


def save_data(args, fnames,  embeddings, classes, dataset_type):
    embedding_path = args.load_embeddings

    df = pd.DataFrame(data=[fnames, classes]).T
    df.columns = ['fnames', 'classes']
    meta_path = embedding_path.replace('.ending', args.dataset + '_' + dataset_type + '_meta.csv')
    df.to_csv(meta_path, sep=' ', header=True)

    array_path = embedding_path.replace('.ending', args.dataset + '_' + dataset_type + '_array.npy')
    with open(array_path, 'wb') as f:
        np.save(f, embeddings)


def process_images(args, data, im_net):
    loader = DataLoader(data, batch_size=1, num_workers=args.prefetch,
                        pin_memory=args.pin_memory, drop_last=False)
    fnames, embeddings, classes = get_test_data(loader, im_net, args)
    fnames = [fname[0] for fname in fnames]
    return fnames, embeddings, classes


def save_dict(args, dict_class):
    print('Class dictionnary')
    dict_path = args.load_embeddings.replace('.ending', args.datapath + '_dict_class.json')
    dict_class = {v: k for k, v in dict_class.items()}
    with open(dict_path, 'w') as fp:
        json.dump(dict_class, fp)


def preprocess_embeddings(args):

    im_net, _ = get_model(args, args.best_model)

    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, valid_im_data], [_, test_im_data], dict_class = load_data(args, transform)
    save_dict(args, dicts_class)

    print('Valid')
    valid_fnames, valid_embeddings, valid_classes = process_images(args, valid_im_data, im_net)
    save_data(args, valid_fnames, valid_embeddings, valid_classes, '_valid')

    print('Test')
    test_fnames, test_embeddings, test_classes = process_images(args, testt_im_data, im_net)
    save_data(args, test_fnames, test_embeddings, test_classes, '_test')


def preprocess_watch_embeddings(args):

    im_net, _ = get_model(args, args.best_model)

    transform = transforms.Compose([transforms.ToTensor()])
    watch_data = Watch(args, transform)
    save_dict(args, wathc_data.dicts_class)

    print('Watch')
    fnames, embeddings, classes = process_images(args, watch_data, im_net)
    save_data(args, fnames,  embeddings, classes, '')


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    args.pin_memory = args.cuda

    # Check Test and Load
    if args.best_model is None:
        raise Exception('Cannot compute embeddins without a model.')
    if args.load_embeddings is None:
        raise Exception('No path to save embeddings')

    if args.dataset == 'watch':
        preprocess_watch_embeddings(args)
    elif args.dataset in ['sketchy_extend', 'tuberlin_extend', 'quickdraw']:
        preprocess_embeddings(args)
    elif args.dataset in ['sk+tu', 'sk+tu+qd']:
        print('\nPlease process the dataset independently, not together.')
    else:
        sys.exit()
