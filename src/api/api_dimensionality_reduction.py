"""
Tensorboard saves the embedding labels in 3 different files which contain the ordered labels, images and embeddings.
    - sprite.png: very large image containing the sketches and images next to each others
    - metadata.csv: each row contains the label of the associated sketch or images in sprite.png
    - tensors.tsv: earch row contains an embedding of the associated sketch or images in sprite.png
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import umap

from src.api.utils import base64_encoding, get_last_epoch_number
from src.constants import (
    TENSORBOARD_CLASSES,
    TENSORBOARD_EMBEDDINGS,
    TENSORBOARD_IMAGE,
    CUSTOM_SKETCH_CLASS,
)


class DimensionalityReduction:
    def __init__(self, save_folder, nb_embeddings):
        # Retrieve data from the files generated by tensorboard (embeddings and classes)
        epoch_number = get_last_epoch_number(save_folder)
        embeddings_path = save_folder + epoch_number + "/default/"

        self.tensors = read_tensor_tsv_file(embeddings_path + TENSORBOARD_EMBEDDINGS)
        self.classes = get_classes(embeddings_path + TENSORBOARD_CLASSES)
        self.tiles = get_tiles(embeddings_path + TENSORBOARD_IMAGE, nb_embeddings)

        self._prepare_projections()

    def _prepare_projections(self):
        """ Compute projection algorithm and prepare data sent to server """
        pca_2d = PCA(n_components=2).fit(self.tensors)
        projection = pca_2d.transform(self.tensors)
        data_pca_2d = prepare_embeddings_data(projection, self.classes, 2)

        pca_3d = PCA(n_components=3).fit(self.tensors)
        projection = pca_3d.transform(self.tensors)
        data_pca_3d = prepare_embeddings_data(projection, self.classes, 3)

        projection = TSNE(n_components=2).fit_transform(self.tensors)
        data_tsne_2d = prepare_embeddings_data(projection, self.classes, 2)

        projection = TSNE(n_components=3).fit_transform(self.tensors)
        data_tsne_3d = prepare_embeddings_data(projection, self.classes, 3)

        umap_2d = umap.UMAP(n_components=2, n_neighbors=200).fit(self.tensors)
        projection = umap_2d.transform(self.tensors)
        data_umap_2d = prepare_embeddings_data(projection, self.classes, 2)

        umap_3d = umap.UMAP(n_components=3, n_neighbors=200).fit(self.tensors)
        projection = umap_3d.transform(self.tensors)
        data_umap_3d = prepare_embeddings_data(projection, self.classes, 3)

        self.class_set = sorted(list(set(data_pca_2d["classes"])))
        self.fitted_algorithm = {
            "PCA": {"2": pca_2d, "3": pca_3d},
            "UMAP": {"2": umap_2d, "3": umap_3d},
        }
        self.projected_embeddings = {
            "PCA": {"2": data_pca_2d, "3": data_pca_3d},
            "TSNE": {"2": data_tsne_2d, "3": data_tsne_3d},
            "UMAP": {"2": data_umap_2d, "3": data_umap_3d},
        }

    def get_projection(self, reduction_algo, nb_dimensions, sketch_embedding=False):
        df = self.projected_embeddings[reduction_algo][nb_dimensions]

        # Prepare data in object
        data = {}

        if reduction_algo != "TSNE" and torch.is_tensor(sketch_embedding):
            fitted_pca = self.fitted_algorithm[reduction_algo][nb_dimensions]
            embedding = fitted_pca.transform(sketch_embedding.cpu().detach().numpy())[0]
            data[CUSTOM_SKETCH_CLASS] = {
                "x": [float(embedding[0])],
                "y": [float(embedding[1])],
            }
            if nb_dimensions == "3":
                data[CUSTOM_SKETCH_CLASS]["z"] = [float(embedding[2])]

        for _class in self.class_set:
            data[_class] = {}
            data[_class]["x"] = list(df[df["classes"] == _class]["x"])
            data[_class]["y"] = list(df[df["classes"] == _class]["y"])
            if nb_dimensions == "3":
                data[_class]["z"] = list(df[df["classes"] == _class]["z"])

        return data

    def get_closest_image(self, json_data):
        df = self.projected_embeddings[json_data["reduction_algo"]][
            str(json_data["nb_dim"])
        ]

        if "z" in json_data.keys():
            point = json_data["x"], json_data["y"], json_data["z"]
            dist = np.sum((df[["x", "y", "z"]].values - point) ** 2, axis=1)
        else:
            point = json_data["x"], json_data["y"]
            dist = np.sum((df[["x", "y"]].values - point) ** 2, axis=1)

        # find index of closest image to x y z in dataframe.
        data = {"image": self.tiles[np.argmin(dist)]}

        return data


def read_tensor_tsv_file(fpath):
    """Read the tensors file:  earch row contains an embedding"""
    with open(fpath, "r") as f:
        tensor = []
        for line in f:
            line = line.rstrip("\n")
            if line:
                tensor.append(list(map(float, line.split("\t"))))
    return np.array(tensor, dtype="float32")


def get_classes(tsv_path):
    """
    Get a list of all labels
    Read the metadata file where each row contains an class
    """
    # Classes of embeddings
    with open(tsv_path, "r") as f:
        return [line.rstrip("\n") for line in f]


def get_tiles(tiles_path, embedding_number):
    """Crops the sprite.png at the appropriate locations to get the images and encode them in base 64"""
    im = Image.open(tiles_path)
    nb_images = 2 * embedding_number  # 2 because sketches and images
    nb_rows = int(np.ceil(np.sqrt(nb_images)))
    full_size = im.size[0]
    size_img = int(full_size / nb_rows)
    tiles = [
        im.crop((x, y, x + size_img, y + size_img))
        for y in range(0, full_size, size_img)
        for x in range(0, full_size, size_img)
    ]
    tiles = tiles[:nb_images]
    tiles = [base64_encoding(tile) for tile in tiles]

    return tiles


def prepare_embeddings_data(X, classes, nb_dimensions):
    """Sort the embeddings by classes"""
    # Process in dataframe
    d = {"x": list(X[:, 0]), "y": list(X[:, 1]), "classes": classes}
    if nb_dimensions == 3:
        d["z"] = list(X[:, 2])
    df = pd.DataFrame(data=d)
    df.sort_values(by=["classes"])

    return df
