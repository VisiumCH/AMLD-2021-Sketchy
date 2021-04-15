import io
import os
import random
import sys

import base64
from cairosvg import svg2png
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from src.data.utils import default_image_loader

NB_DATASET_IMAGES = 5


def get_image(folder_path, ending):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(ending)
    ]
    return np.random.choice(files, NB_DATASET_IMAGES)


def base64_encoding(image, bytes_type="PNG"):
    rawBytes = io.BytesIO()
    image.save(rawBytes, bytes_type)
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    return str(img_base64)


def prepare_dataset_data(dataset_path, image_type, category):

    if image_type == "images":
        ending = ".jpg"
        bytes_type = "JPEG"
    elif image_type == "sketches":
        ending = ".png"
        bytes_type = "PNG"
    else:
        print("Type must be either images or sketches.")
        sys.exit()

    data = {}
    images_folder_path = os.path.join(dataset_path, image_type, category)
    images_paths = get_image(images_folder_path, ending)

    for i, image_path in enumerate(images_paths):
        image = Image.open(image_path)
        data[f"{image_type}_{i}_base64"] = base64_encoding(image, bytes_type)

    return data


def svg_to_png(sketch):
    # random name
    random_number = str(random.random())
    sketch_fname = "sketch" + random_number + ".png"

    # make png
    svg2png(bytestring=sketch, write_to=sketch_fname)

    # add white background
    im = Image.open(sketch_fname)
    im = im.convert("RGBA")
    background = Image.new(im.mode[:-1], im.size, (255, 255, 255))
    background.paste(im, im.split()[-1])  # omit transparency
    background.convert("RGB").save(sketch_fname)

    sketch = default_image_loader(sketch_fname)
    os.remove(sketch_fname)

    return sketch


def prepare_images_data(images, image_labels, attention):
    data = {}
    data["images_base64"] = []
    data["images_label"] = []

    for image, image_label in zip(images, image_labels):
        data["images_base64"].append(base64_encoding(image, "PNG"))
        data["images_label"].append(" ".join(image_label.split("_")))

    im = Image.fromarray(attention.astype("uint8"))
    data["attention"] = base64_encoding(im, "PNG")

    return data


def prepare_sketch(sketch):
    sketch = svg_to_png(sketch)
    return base64_encoding(sketch, bytes_type="PNG")


def read_tensor_tsv_file(fpath):
    with open(fpath, "r") as f:
        tensor = []
        for line in f:
            line = line.rstrip("\n")
            if line:
                tensor.append(list(map(float, line.split("\t"))))
    return np.array(tensor, dtype="float32")


def read_class_tsv_file(fpath):
    with open(fpath, "r") as f:
        return [line.rstrip("\n") for line in f]


def project_embeddings(tensors_path, n_components, sketch_emb):
    tensors = read_tensor_tsv_file(tensors_path)
    if type(sketch_emb) != bool:
        # Add sketch embeddings
        sketch_emb = sketch_emb.detach().numpy()
        tensors = np.append(tensors, sketch_emb, axis=0)

    # PCA on tensors
    pca = PCA(n_components=n_components)
    pca.fit(tensors)
    return pca.transform(tensors)


def get_class(tsv_path, sketch_emb):
    # Classes of embeddings
    classes = read_class_tsv_file(tsv_path)
    if type(sketch_emb) != bool:
        classes.append("My Custom Sketch")
    return classes


def get_tiles(im_path):
    im = Image.open(im_path)
    nb_rows = 23
    nb_images = 2 * 250
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


def process_graph(embeddings_path, n_components, sketch_emb=False):
    # File names
    tensors_path = embeddings_path + "tensors.tsv"
    tsv_path = embeddings_path + "metadata.tsv"

    X = project_embeddings(tensors_path, n_components, sketch_emb)
    classes = get_class(tsv_path, sketch_emb)

    # Process in dataframe
    d = {"x": list(X[:, 0]), "y": list(X[:, 1]), "classes": classes}
    if n_components == 3:
        d["z"] = list(X[:, 2])
    df = pd.DataFrame(data=d)
    return df


def prepare_embeddings_data(df, nb_dimensions):
    df.sort_values(by=["classes"])
    class_set = sorted(list(set(df["classes"])))

    # Prepare data in object
    data = {}
    for _class in class_set:
        data[_class] = {}
        data[_class]["x"] = list(df[df["classes"] == _class]["x"])
        data[_class]["y"] = list(df[df["classes"] == _class]["y"])
        if nb_dimensions == 3:
            data[_class]["z"] = list(df[df["classes"] == _class]["z"])

    return data