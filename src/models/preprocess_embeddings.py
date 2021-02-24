import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.loader_factory import load_data
from src.models.test import get_test_data
from src.models.utils import get_model
from src.options import Options


# def precompute_embeddings(data_loader, model, args):
#     fnames = []
#     for i, (image, fname, target) in enumerate(data_loader):
#         if i % 1000 == 0:
#             print(f'{i} images processed on {len(data_loader)}')

#         # Process
#         out_features, _ = model(image)

#         # Filename of the images for qualitative
#         fnames.append(fname)

#         if i == 0:
#             embeddings = out_features.detach().numpy()
#             classes = target.detach().numpy()
#         else:
#             embeddings = np.concatenate((embeddings, out_features.detach().numpy()), axis=0)
#             classes = np.concatenate((classes, target.detach().numpy()), axis=0)

#     return fnames, embeddings, classes


def preprocess_embeddings(args, model_path, embedding_path):

    im_net, _ = get_model(args, model_path)

    transform = transforms.Compose([transforms.ToTensor()])
    _, [_, valid_im_data], [_, test_im_data], dict_class = load_data(args, transform)

    print(len(valid_im_data))

    if args.cuda:
        pin_memory = True
    else:
        pin_memory = False

    print('Valid')
    valid_im_loader = DataLoader(valid_im_data, batch_size=3*args.batch_size,
                                 num_workers=args.prefetch, pin_memory=pin_memory, drop_last=True)
    images_fnames, images_embeddings, images_classes = get_test_data(valid_im_loader, im_net, args)

    print('Test')
    test_im_loader = DataLoader(test_im_data, batch_size=3*args.batch_size,
                                num_workers=args.prefetch, pin_memory=pin_memory, drop_last=True)
    fnames, embeddings, classes = get_test_data(test_im_loader, im_net, args)
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
    args.load = 'io/models/1_run-batch_size_10/checkpoint.pth'
    preprocess_embeddings(args, args.load, args.load_embeddings)
