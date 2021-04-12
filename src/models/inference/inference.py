import json
import os
import random

import matplotlib
import matplotlib.pyplot as plt
#plt.ioff()
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from src.data.constants import DatasetName, Split
from src.data.loader_factory import load_data
from src.data.utils import default_image_loader, get_loader, get_dict
from src.models.utils import get_model, normalise_attention, get_parameters
from src.models.metrics import get_similarity

NUM_CLOSEST = 4
NUMBER_RANDOM_IMAGES = 20


class Inference:
    """ Class to infer closest images of a sketch """

    def __init__(self, args, dataset_type):
        """
        Initialises the inference with the trained model and precomputed embeddings
        Args:
            - args: arguments received from the command line (argparse)
            - dataset_type: dataset split ('train', 'valid' or 'test')
        """
        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.loader = default_image_loader

        self.im_net, self.sk_net = get_model(args, args.load)
        self.im_net.eval()
        self.sk_net.eval()
        torch.set_grad_enabled(False)

        self.prediction_folder = os.path.join(args.save, "predictions")
        os.makedirs(self.prediction_folder, exist_ok=True)

        self.embedding_path = os.path.join(args.save, "precomputed_embeddings")
        os.makedirs(self.embedding_path, exist_ok=True)

        self.__get_data(dataset_type)

    def __get_processed_images(self, dataset_type):
        """
        Get the data of images to match with the sketches
        Args:
            - dataset_type: dataset split ('train', 'valid' or 'test')
        Return:
            - dict_class: dictionnary mapping numbers to classes names
            - paths to the images
            - classes of the images
            - images_embeddings: embeddings of the images
        """
        dict_path = os.path.join(
            self.embedding_path, self.args.dataset + "_dict_class.json"
        )
        with open(dict_path, "r") as fp:
            dict_class = json.load(fp)

        array_path = os.path.join(
            self.embedding_path, self.args.dataset + "_" + dataset_type + "_array.npy"
        )
        with open(array_path, "rb") as f:
            images_embeddings = np.load(f)

        meta_path = os.path.join(
            self.embedding_path, self.args.dataset + "_" + dataset_type + "_meta.csv"
        )
        df = pd.read_csv(meta_path, sep=" ")

        return dict_class, df["fnames"].values, df["classes"].values, images_embeddings

    def __get_data(self, dataset_type):
        """
        Loads the paths, classes and embeddings of the images of different datasets
        """
        dataset = self.args.dataset

        if dataset in [
            DatasetName.sketchy,
            DatasetName.tuberlin,
            DatasetName.quickdraw,
        ]:
            (
                self.dict_class,
                self.images_fnames,
                self.images_classes,
                self.images_embeddings,
            ) = self.__get_processed_images(dataset_type)
            self.sketchy_limit = None
            self.tuberlin_limit = None

        elif dataset in [DatasetName.sktu, DatasetName.sktuqd]:
            self.args.dataset = DatasetName.sketchy
            (
                dict_class_sk,
                self.images_fnames,
                self.images_classes,
                self.images_embeddings,
            ) = self.__get_processed_images(dataset_type)

            self.sketchy_limit = len(self.images_fnames)
            self.tuberlin_limit = None

            self.args.dataset = DatasetName.tuberlin
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

            if dataset == DatasetName.sktuqd:
                self.args.dataset = DatasetName.quickdraw
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

    def random_images_inference(self, number_sketches):
        """ Selects number_sketches random sketched and find the closest images """
        _, _, [test_sk_loader, _], _ = load_data(self.args, self.transform)
        rand_samples_sk = np.random.randint(
            0, high=len(test_sk_loader), size=number_sketches
        )

        for i in range(len(rand_samples_sk)):
            _, sketch_fname, _ = test_sk_loader[rand_samples_sk[i]]
            self.inference_sketch(sketch_fname)
            self.plot_closest(sketch_fname)

    def inference_sketch(self, sketch_fname):
        """ Find the closest images of a sketch and plot it """
        sketch = self.transform(self.loader(sketch_fname)).unsqueeze(
            0
        )  # unsqueeze because 1 sketch (no batch)

        if self.args.cuda:
            sketch = sketch.cuda()
        sketch_embedding, self.attn_sk = self.sk_net(sketch)

        if self.args.cuda:
            sketch_embedding = sketch_embedding.cpu()
        self.get_closest_images(sketch_embedding)
        return sketch_embedding

    def get_attention(self, sketch_fname):
        ''' Find the closest images of a sketch and plot it '''

        sketch = self.transform(self.loader(sketch_fname)).unsqueeze(0)  # unsqueeze because 1 sketch (no batch)

        if self.args.cuda:
            sketch = sketch.cuda()

        attn_sk = normalise_attention(self.attn_sk, sketch)
        heat_map = attn_sk.squeeze().detach().numpy()

        sk = self.loader(sketch_fname)

        fig, ax = plt.subplots(frameon=False)

        ax.imshow(sk, aspect='auto')
        ax.imshow(255 * heat_map, alpha=0.7, cmap='Spectral_r', aspect='auto')
        ax.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        attention = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        
        attention = attention.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.show()
        
        fig.clf()
        plt.close('all')
        
        return attention

    def get_closest(self, number):
        images, labels = [], []
        for index in range(number):
            image, label = self.prepare_image(index)
            images.append(image)
            labels.append(label)
        return images, labels
        

    def get_closest_images(self, sketch_embedding):
        """
        Based on a sketch embedding, retrieve the index of the closest images
        """
        similarity = get_similarity(
            sketch_embedding.detach().numpy(), self.images_embeddings
        )
        arg_sorted_sim = (-similarity).argsort()

        self.sorted_fnames = [
            self.images_fnames[i] for i in arg_sorted_sim[0][0: NUM_CLOSEST + 1]
        ]
        self.sorted_labels = [
            self.images_classes[i] for i in arg_sorted_sim[0][0: NUM_CLOSEST + 1]
        ]

    def prepare_image(self, index):
        dataset = self.sorted_fnames[index].split("/")[-4]

        loader = get_loader(dataset)
        image = loader(self.sorted_fnames[index])

        dict_class = get_dict(dataset, self.dict_class)
        label = dict_class[str(self.sorted_labels[index])]
        return image, label

    def plot_closest(self, sketch_fname):
        """
        Plots a sketch with its closest images in the embedding space.
        The images are stored in the same folder as the best model in a subfolder called 'predictions'
        """
        fig, axes = plt.subplots(1, NUM_CLOSEST + 2, figsize=((NUM_CLOSEST + 1) * 4, 8))

        sk = mpimg.imread(sketch_fname)
        axes[0].imshow(sk)
        axes[0].set(title="Sketch \n Label: " + sketch_fname.split("/")[-2])
        axes[0].axis("off")

        if self.args.cuda:
            self.attn_sk = self.attn_sk.cpu()
        heat_map = self.attn_sk.squeeze().detach().numpy()

        axes[1].imshow(sk)
        axes[1].imshow(255 * heat_map, alpha=0.7, cmap="Spectral_r")
        axes[1].set(title=sketch_fname.split("/")[-2] + "\n Attention Map")
        axes[1].axis("off")

        for i in range(2, NUM_CLOSEST + 2):
            im, label = self.prepare_image(i - 2)
            axes[i].imshow(im)
            axes[i].set(title='Closest image ' + str(i) + '\n Label: ' + label)
            axes[i].axis('off')
        plt.subplots_adjust(wspace=0.25, hspace=-0.35)

        img_name = "_".join(sketch_fname.split("/")[-2:])
        plt.savefig(os.path.join(self.prediction_folder, img_name))
        plt.close(fig)



def main(args):
    inference = Inference(args, Split.test)
    inference.random_images_inference(number_sketches=NUMBER_RANDOM_IMAGES)


if __name__ == "__main__":

    args = get_parameters()
    main(args)
