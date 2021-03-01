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
    meta_path = embedding_path.replace('.ending', dataset_type + '_meta.csv')
    df.to_csv(meta_path, sep=' ', header=True)

    array_path = embedding_path.replace('.ending', dataset_type + '_array.npy')
    with open(array_path, 'wb') as f:
        np.save(f, embeddings)


def preprocess_embeddings(args):

    im_net, _ = get_model(args, args.best_model)

    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, valid_im_data], [_, test_im_data], dict_class = load_data(args, transform)

    if args.cuda:
        pin_memory = True
    else:
        pin_memory = False

    print('Class dictionnary')
    dict_path = args.load_embeddings.replace('.ending', '_dict_class.json')
    dict_class = {v: k for k, v in dict_class.items()}
    with open(dict_path, 'w') as fp:
        json.dump(dict_class, fp)

    print('Valid')
    valid_im_loader = DataLoader(valid_im_data, batch_size=1, num_workers=args.prefetch,
                                 pin_memory=pin_memory, drop_last=False)
    valid_fnames, valid_embeddings, valid_classes = get_test_data(valid_im_loader, im_net, args)
    valid_fnames = [fnames[0] for fnames in valid_fnames]
    save_data(args, valid_fnames, valid_embeddings, valid_classes, '_valid')

    print('Test')
    test_im_loader = DataLoader(test_im_data, batch_size=1, num_workers=args.prefetch,
                                pin_memory=pin_memory, drop_last=False)
    test_fnames, test_embeddings, test_classes = get_test_data(test_im_loader, im_net, args)
    test_fnames = [fnames[0] for fnames in test_fnames]
    save_data(args, test_fnames, test_embeddings, test_classes, '_test')


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    # Check Test and Load
    if args.best_model is None:
        raise Exception('Cannot compute embeddins without a model.')
    if args.load_embeddings is None:
        raise Exception('No path to save embeddings')
    preprocess_embeddings(args)
