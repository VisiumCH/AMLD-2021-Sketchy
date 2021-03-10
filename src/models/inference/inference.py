import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from src.data.loader_factory import load_data
from src.data.utils import default_image_loader, default_image_loader_tuberlin
from src.options import Options
from src.models.utils import get_model
from src.models.metrics import get_similarity

NUM_CLOSEST = 4
NUMBER_RANDOM_IMAGES = 20


def get_loader(dataset):
    if dataset == 'TU-Berlin':
        loader = default_image_loader_tuberlin
    else:
        loader = default_image_loader
    return loader


def get_dict(dataset, dict_class):
    if isinstance(dict_class, dict):
        dict_class = dict_class
    else:
        if dataset == 'Sketchy':
            dict_class = dict_class[0]
        elif dataset == 'TU-Berlin':
            dict_class = dict_class[1]
        elif dataset == 'Quickdraw':
            dict_class = dict_class[2]
        else:
            raise(f"Error with dataset name: {dataset}.")
    return dict_class


class Inference():

    def __init__(self, args, dataset_type):

        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.loader = default_image_loader

        self.im_net, self.sk_net = get_model(args, args.best_model)
        self.im_net.eval()
        self.sk_net.eval()
        torch.set_grad_enabled(False)

        self.prediction_folder = os.path.join(args.best_model.rstrip('checkpoint.pth'), 'predictions')
        if not os.path.exists(self.prediction_folder):
            os.makedirs(self.prediction_folder)

        self.embedding_path = os.path.join(args.best_model.rstrip('checkpoint.pth'), 'precomputed_embeddings')
        if not os.path.exists(self.embedding_path):
            os.makedirs(self.embedding_path)

        self.get_data(dataset_type)

    def get_processed_images(self, args, dataset_type):

        dict_path = os.path.join(self.embedding_path, '_' + args.dataset + '_dict_class.json')
        with open(dict_path, 'r') as fp:
            dict_class = json.load(fp)

        array_path = os.path.join(self.embedding_path,  '_' + args.dataset + '_' + dataset_type + '_array.npy')
        with open(array_path, 'rb') as f:
            images_embeddings = np.load(f)

        meta_path = os.path.join(self.embedding_path, '_' + args.dataset + '_' + dataset_type + '_meta.csv')
        df = pd.read_csv(meta_path, sep=' ')

        return dict_class, df['fnames'].values, df['classes'].values, images_embeddings

    def get_data(self, dataset_type):

        dataset = self.args.dataset

        if dataset in ['sketchy', 'tuberlin', 'quickdraw']:
            self.dict_class, self.images_fnames, self.images_classes, self.images_embeddings = self.get_processed_images(
                self.args, dataset_type)
            self.sketchy_limit = None
            self.tuberlin_limit = None

        elif dataset in ['sk+tu', 'sk+tu+qd']:
            self.args.dataset = 'sketchy'
            dict_class_sk, self.images_fnames, self.images_classes, self.images_embeddings = self.get_processed_images(
                self.args, dataset_type)

            self.sketchy_limit = len(self.images_fnames)
            self.tuberlin_limit = None

            self.args.dataset = 'tuberlin'
            dict_class_tu, images_fnames, images_classes, images_embeddings = self.get_processed_images(
                self.args, dataset_type)
            self.dict_class = [dict_class_sk, dict_class_tu]

            self.images_fnames = np.concatenate((self.images_fnames, images_fnames), axis=0)
            self.images_classes = np.concatenate((self.images_classes, images_classes), axis=0)
            self.images_embeddings = np.concatenate((self.images_embeddings, images_embeddings), axis=0)

            if dataset == 'sk+tu+qd':
                self.args.dataset = 'quickdraw'
                self.tuberlin_limit = len(self.images_fnames)

                dict_class_qd, images_fnames, images_classes, images_embeddings = self.get_processed_images(
                    self.args, dataset_type)
                self.dict_class.append(dict_class_qd)

                self.images_fnames = np.concatenate((self.images_fnames, images_fnames), axis=0)
                self.images_classes = np.concatenate((self.images_classes, images_classes), axis=0)
                self.images_embeddings = np.concatenate((self.images_embeddings, images_embeddings), axis=0)
        else:
            raise Exception(args.dataset + ' not implemented.')
        self.args.dataset = dataset

    def random_images_inference(self, args, number_images):
        _, _, [test_sk_loader, _], _ = load_data(args, self.transform)
        rand_samples_sk = np.random.randint(0, high=len(test_sk_loader), size=number_images)

        for i in range(len(rand_samples_sk)):
            _, sketch_fname, _ = test_sk_loader[rand_samples_sk[i]]
            self.inference_sketch(sketch_fname)

    def inference_sketch(self, sketch_fname):
        ''' Process a sketch'''
        sketch = self.transform(self.loader(sketch_fname)).unsqueeze(0)  # unsqueeze because 1 sketch (no batch)
        if self.args.cuda:
            sketch = sketch.cuda()
        sketch_embedding, _ = self.sk_net(sketch)
        if self.args.cuda:
            sketch_embedding = sketch_embedding.cpu()
        self.get_closest_images(sketch_embedding)

        self.plot_closest(sketch_fname)

    def get_closest_images(self, sketch_embedding):
        '''
        Based on a sketch embedding, retrieve the index of the closest images
        '''
        similarity = get_similarity(sketch_embedding.detach().numpy(), self.images_embeddings)
        arg_sorted_sim = (-similarity).argsort()

        self.sorted_fnames = [self.images_fnames[i]
                              for i in arg_sorted_sim[0][0:NUM_CLOSEST + 1]]
        self.sorted_labels = [self.images_classes[i]
                              for i in arg_sorted_sim[0][0:NUM_CLOSEST + 1]]

    def plot_closest(self, sketch_fname):
        fig, axes = plt.subplots(1, NUM_CLOSEST + 1, figsize=(20, 8))

        sk = mpimg.imread(sketch_fname)
        axes[0].imshow(sk)
        axes[0].set(title='Sketch \n Label: ' + sketch_fname.split('/')[-2])
        axes[0].axis('off')

        for i in range(1, NUM_CLOSEST + 1):

            dataset = self.sorted_fnames[i-1].split('/')[-4]
            loader = get_loader(dataset)
            dict_class = get_dict(dataset, self.dict_class)

            im = loader(self.sorted_fnames[i-1])
            axes[i].imshow(im)
            axes[i].set(title='Closest image ' + str(i) +
                        '\n Label: ' + dict_class[str(self.sorted_labels[i-1])])

            axes[i].axis('off')
        plt.subplots_adjust(wspace=0.25, hspace=-0.35)

        img_name = '_'.join(sketch_fname.split('/')[-2:])
        plt.savefig(os.path.join(self.prediction_folder, img_name))


def main(args):
    inference_test = Inference(args, 'test')
    inference_test.random_images_inference(args, number_images=NUMBER_RANDOM_IMAGES)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    # Check Test and Load
    if args.best_model is None:
        raise Exception('Cannot test without loading a model.')

    main(args)
