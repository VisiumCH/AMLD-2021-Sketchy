import json
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from src.constants import (
    DICT_CLASS, EMB_ARRAY, EMBEDDINGS, METADATA,
    NUM_CLOSEST_PLOT,
    NUMBER_RANDOM_IMAGES, PREDICTION,
    SKETCHY, TUBERLIN, QUICKDRAW, SKTU, SKTUQD
)
from src.data.loader_factory import load_data
from src.data.utils import default_image_loader, get_loader, get_dict
from src.models.utils import get_model, normalise_attention, get_parameters
from src.models.metrics import get_similarity


class Inference:
    """
    Class to infer closest images of a sketch
    Parent class of
        - PlotInference called from main to plot closest images of random sketch
        - ApiInference called from the api to retrieve the closest images of a hand-drawn sketch
    """

    def __init__(self, args, dataset_type):
        """
        Initialises the inference with the trained model and precomputed embeddings
        Args:
            - args: arguments received from the command line (argparse)
            - dataset_type: 'train', 'valid' or 'test'
        """
        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.loader = default_image_loader

        self.im_net, self.sk_net = get_model(args, args.load)
        self.im_net.eval()
        self.sk_net.eval()
        torch.set_grad_enabled(False)

        self.prediction_folder = os.path.join(args.save, PREDICTION)
        os.makedirs(self.prediction_folder, exist_ok=True)

        self.embedding_path = os.path.join(args.save, EMBEDDINGS)
        os.makedirs(self.embedding_path, exist_ok=True)

        self.__get_data(dataset_type)

    def __get_processed_images(self, dataset_type):
        """
        Get the data of images to match with the sketches
        Args:
            - dataset_type: 'train', 'valid' or 'test'
        Return:
            - dict_class: dictionnary mapping numbers to classes names
            - paths to the images
            - classes of the images
            - images_embeddings: embeddings of the images
        """
        dict_path = os.path.join(
            self.embedding_path, self.args.dataset + DICT_CLASS
        )
        with open(dict_path, "r") as fp:
            dict_class = json.load(fp)

        array_path = os.path.join(
            self.embedding_path, self.args.dataset + "_" + dataset_type + EMB_ARRAY
        )
        with open(array_path, "rb") as f:
            images_embeddings = np.load(f)

        meta_path = os.path.join(
            self.embedding_path, self.args.dataset + "_" + dataset_type + METADATA
        )
        df = pd.read_csv(meta_path, sep=" ")

        return dict_class, df["fnames"].values, df["classes"].values, images_embeddings

    def __get_data(self, dataset_type):
        """
        Loads the paths, classes and embeddings of the images of different datasets
        """
        dataset = self.args.dataset

        if dataset in [SKETCHY, TUBERLIN, QUICKDRAW]:
            (
                self.dict_class,
                self.images_fnames,
                self.images_classes,
                self.images_embeddings,
            ) = self.__get_processed_images(dataset_type)
            self.sketchy_limit = None
            self.tuberlin_limit = None

        elif dataset in [SKTU, SKTUQD]:
            self.args.dataset = SKETCHY
            (
                dict_class_sk,
                self.images_fnames,
                self.images_classes,
                self.images_embeddings,
            ) = self.__get_processed_images(dataset_type)

            self.sketchy_limit = len(self.images_fnames)
            self.tuberlin_limit = None

            self.args.dataset = TUBERLIN
            (
                dict_class_tu,
                images_fnames,
                images_classes,
                images_embeddings,
            ) = self.__get_processed_images(dataset_type)
            self.dict_class = [dict_class_sk, dict_class_tu]

            self.images_fnames = np.concatenate(
                (self.images_fnames, images_fnames), axis=0
            )
            self.images_classes = np.concatenate(
                (self.images_classes, images_classes), axis=0
            )
            self.images_embeddings = np.concatenate(
                (self.images_embeddings, images_embeddings), axis=0
            )

            if dataset == SKTUQD:
                self.args.dataset = QUICKDRAW
                self.tuberlin_limit = len(self.images_fnames)

                (
                    dict_class_qd,
                    images_fnames,
                    images_classes,
                    images_embeddings,
                ) = self.__get_processed_images(dataset_type)
                self.dict_class.append(dict_class_qd)

                self.images_fnames = np.concatenate(
                    (self.images_fnames, images_fnames), axis=0
                )
                self.images_classes = np.concatenate(
                    (self.images_classes, images_classes), axis=0
                )
                self.images_embeddings = np.concatenate(
                    (self.images_embeddings, images_embeddings), axis=0
                )
        else:
            raise Exception(self.args.dataset + " not implemented.")
        self.args.dataset = dataset

    def inference_sketch(self, sk):
        """
        Find the closest images of a sketch and plot them
        """
        self.sketch = self.transform(sk).unsqueeze(0)

        if self.args.cuda:
            self.sketch = self.sketch.cuda()
        sketch_embedding, self.attn_sk = self.sk_net(self.sketch)

        if self.args.cuda:
            sketch_embedding = sketch_embedding.cpu()

        similarity = get_similarity(
            sketch_embedding.detach().numpy(), self.images_embeddings
        )
        arg_sorted_sim = (-similarity).argsort()

        self.sorted_fnames = [
            self.images_fnames[i] for i in arg_sorted_sim[0][0: NUM_CLOSEST_PLOT + 1]
        ]
        self.sorted_labels = [
            self.images_classes[i] for i in arg_sorted_sim[0][0: NUM_CLOSEST_PLOT + 1]
        ]
        return sketch_embedding

    def get_heatmap(self):
        """ Normalise the attention output of the model for heatmap plots """
        attn_sk = normalise_attention(self.attn_sk, self.sketch)
        self.heat_map = attn_sk.squeeze().detach().numpy()

    def prepare_image(self, index):
        """ Gets an an image and its label based on its index """
        dataset = self.sorted_fnames[index].split("/")[-4]

        loader = get_loader(dataset)
        image = loader(self.sorted_fnames[index])

        dict_class = get_dict(dataset, self.dict_class)
        label = dict_class[str(self.sorted_labels[index])]
        return image, label


class PlotInference(Inference):
    """ Plot inference of a random sketch with its closest images in the latent space"""

    def __init__(self, args, dataset_type):
        super().__init__(args, dataset_type)

    def random_images_inference(self, number_sketches):
        """ Selects number_sketches random sketched and find the closest images """
        _, _, [test_sk_loader, _], _ = load_data(self.args, self.transform)
        rand_samples_sk = np.random.randint(
            0, high=len(test_sk_loader), size=number_sketches
        )

        for i in range(len(rand_samples_sk)):
            _, sketch_fname, _ = test_sk_loader[rand_samples_sk[i]]
            self.sk = self.loader(sketch_fname)
            self.inference_sketch(self.sk)
            self.get_heatmap()
            self.plot_closest(sketch_fname)

    def plot_closest(self, sketch_fname):
        """
        Plots a sketch with its closest images in the embedding space.
        The images are stored in the same folder as the best model in a subfolder called 'predictions'
        """
        fig, axes = plt.subplots(1, NUM_CLOSEST_PLOT + 2, figsize=((NUM_CLOSEST_PLOT + 1) * 4, 8))

        axes[0].imshow(self.sk)
        axes[0].set(title="Sketch \n Label: " + sketch_fname.split("/")[-2])
        axes[0].axis("off")

        axes[1].imshow(self.sk)
        axes[1].imshow(255 * self.heat_map, alpha=0.7, cmap="Spectral_r")
        axes[1].set(title=sketch_fname.split("/")[-2] + "\n Attention Map")
        axes[1].axis("off")

        for i in range(2, NUM_CLOSEST_PLOT + 2):
            im, label = self.prepare_image(i - 1)
            axes[i].imshow(im)
            axes[i].set(title="Closest image " + str(i) + "\n Label: " + label)
            axes[i].axis("off")
        plt.subplots_adjust(wspace=0.25, hspace=-0.35)

        img_name = "_".join(sketch_fname.split("/")[-2:])
        plt.savefig(os.path.join(self.prediction_folder, img_name))
        plt.close(fig)


def main(args):
    """ From here, the inference is done on a random sketch and a plot with its closest images is made """
    inference = PlotInference(args, "test")
    inference.random_images_inference(number_sketches=NUMBER_RANDOM_IMAGES)


if __name__ == "__main__":

    args = get_parameters()
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    main(args)
