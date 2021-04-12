import io
import os
import sys

import base64
from cairosvg import svg2png
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA


def get_image(folder_path, ending):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(ending)
    ]
    return np.random.choice(files, 5)


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
    print(images_paths)
    for i, image_path in enumerate(images_paths):
        image = Image.open(image_path)

        rawBytes = io.BytesIO()
        image.save(rawBytes, bytes_type)
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())

        data[f"{image_type}_{i}_base64"] = str(img_base64)

    return data


def svg_to_png(sketch, sketch_fname):
    # make png
    svg2png(bytestring=sketch, write_to=sketch_fname)

    # add white background
    im = Image.open(sketch_fname)
    im = im.convert("RGBA")
    background = Image.new(im.mode[:-1], im.size, (255, 255, 255))
    background.paste(im, im.split()[-1])  # omit transparency
    im = background
    im.convert("RGB").save(sketch_fname)


def prepare_images_data(images, image_labels, attention):
    data = {}
    data["images_base64"] = []
    data["images_label"] = []

    for image, image_label in zip(images, image_labels):
        rawBytes = io.BytesIO()
        image.save(rawBytes, "PNG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())

        data["images_base64"].append(str(img_base64))
        data["images_label"].append(" ".join(image_label.split("_")))

    im = Image.fromarray(attention.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    attention_base64 = base64.b64encode(rawBytes.read())
    data["attention"] = str(attention_base64)

    return data


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


def process_embeddings(embeddings_path, n_components, sketch_emb):
    # File names
    tensors_path = embeddings_path + "tensors.tsv"
    tsv_path = embeddings_path + "metadata.tsv"

    tensors = read_tensor_tsv_file(tensors_path)
    if type(sketch_emb) != bool:
        # Add sketch embeddings
        sketch_emb = sketch_emb.detach().numpy()
        tensors = np.append(tensors, sketch_emb, axis=0)

    # PCA on tensors
    pca = PCA(n_components=n_components)
    pca.fit(tensors)
    X = pca.transform(tensors)

    # Classes of embeddings
    classes = read_class_tsv_file(tsv_path)
    if type(sketch_emb) != bool:
        classes.append("My Custom Sketch")

    # Process in dataframe
    if n_components == 3:
        d = {
            "x": list(X[:, 0]),
            "y": list(X[:, 1]),
            "z": list(X[:, 2]),
            "classes": classes,
        }
    else:
        d = {"x": list(X[:, 0]), "y": list(X[:, 1]), "classes": classes}
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
