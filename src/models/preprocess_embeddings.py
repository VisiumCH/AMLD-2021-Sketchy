import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.loader_factory import load_data
from src.data.utils import get_model
from src.options import Options


def precompute_loader_data(data_loader, model, args):
    fnames = []
    for i, (image, fname, target) in enumerate(data_loader):
        if i % 1000 == 0:
            print(f'{i} images processed on {len(data_loader)}')

        # Process
        out_features, _ = model(image)

        # Filename of the images for qualitative
        fnames.append(fname)

        if i == 0:
            embeddings = out_features.detach().numpy()
            classes = target.detach().numpy()
        else:
            embeddings = np.concatenate((embeddings, out_features.detach().numpy()), axis=0)
            classes = np.concatenate((classes, target.detach().numpy()), axis=0)

    return fnames, embeddings, classes


def preprocess_embeddings(args, model_path, embedding_path):

    im_net, _ = get_model(args, model_path)

    transform = transforms.Compose([transforms.ToTensor()])
    train_data, [_, valid_im_data], [_, test_im_data], dict_class = load_data(args, transform)

    print('Train')
    train_im_loader = DataLoader(train_data, batch_size=1)
    images_fnames, images_embeddings, images_classes = precompute_loader_data(train_im_loader, im_net, args)

    print('Valid')
    valid_im_loader = DataLoader(valid_im_data, batch_size=1)
    fnames, embeddings, classes = precompute_loader_data(valid_im_loader, im_net, args)
    images_fnames.extend(fnames)
    images_embeddings = np.concatenate((images_embeddings, embeddings), axis=0)
    images_classes = np.concatenate((images_classes, classes), axis=0)

    print('Test')
    test_im_loader = DataLoader(test_im_data, batch_size=1)
    fnames, embeddings, classes = precompute_loader_data(test_im_loader, im_net, args)
    images_fnames.extend(fnames)
    images_embeddings = np.concatenate((images_embeddings, embeddings), axis=0)
    images_classes = np.concatenate((images_classes, classes), axis=0)

    df = pd.DataFrame(data=[images_fnames, images_embeddings, images_classes]).T
    df.columns = ['fnames', 'embeddings', 'classes']
    df.to_csv(embedding_path, sep=' ', header=True)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')

    preprocess_embeddings(args, args.load, args.load_embeddings)
