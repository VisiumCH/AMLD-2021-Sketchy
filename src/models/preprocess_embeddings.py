import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.loader_factory import load_data
from src.models.test import get_test_data
from src.models.utils import get_model
from src.options import Options


def preprocess_embeddings(args, model_path, embedding_path):

    im_net, _ = get_model(args, model_path)

    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, valid_im_data], [_, test_im_data], dict_class = load_data(args, transform)

    if args.cuda:
        pin_memory = True
    else:
        pin_memory = False

    print('Valid')
    valid_im_loader = DataLoader(valid_im_data, batch_size=1, num_workers=args.prefetch,
                                 pin_memory=pin_memory, drop_last=True)
    images_fnames, images_embeddings, images_classes = get_test_data(valid_im_loader, im_net, args)

    print('Test')
    test_im_loader = DataLoader(test_im_data, batch_size=1, num_workers=args.prefetch,
                                pin_memory=pin_memory, drop_last=True)
    fnames, embeddings, classes = get_test_data(test_im_loader, im_net, args)
    images_fnames.extend(fnames)
    images_classes = np.concatenate((images_classes, classes), axis=0)
    images_embeddings = np.concatenate((images_embeddings, embeddings), axis=0)

    df = pd.DataFrame(data=[images_fnames, images_classes]).T
    df.columns = ['fnames', 'classes']
    meta_path = embedding_path.replace('.ending', '_meta.csv')
    df.to_csv(meta_path, sep=' ', header=True)

    array_path = embedding_path.replace('.ending', '_array.npy')
    with open(array_path, 'wb') as f:
        np.save(f, images_embeddings)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')
    args.load = 'io/models/1_run-batch_size_10/checkpoint.pth'
    preprocess_embeddings(args, args.load, args.load_embeddings)
