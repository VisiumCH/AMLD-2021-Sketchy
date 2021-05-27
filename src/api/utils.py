import io
import os
import random
import sys

import base64
from cairosvg import svg2png
import numpy as np
from PIL import Image

from src.constants import NB_DATASET_IMAGES, PARAMETERS
from src.data.utils import default_image_loader


def get_image(folder_path, ending):
    """ Get a list of all images or sketches in a folder """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(ending)
    ]
    return np.random.choice(files, NB_DATASET_IMAGES)


def base64_encoding(image, bytes_type="PNG"):
    """ Encode image in base 64 encoding """
    rawBytes = io.BytesIO()
    image.save(rawBytes, bytes_type)
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    return str(img_base64)


def prepare_dataset(dataset_path, image_type, category):
    """ Base 64 encoding of all images in dataset_path """
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
    """ Convert a sketch in svg format to an image array """
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
    os.remove(sketch_fname)  # remove saved sketch from machine

    return sketch


def prepare_images_data(images, image_labels, attention):
    """ Load the images, labels and attention into dictionnary to send to the web app """
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
    """ Prepare the sketch: convert it from svg to a base64 encoding """
    sketch = svg_to_png(sketch)
    return base64_encoding(sketch, bytes_type="PNG")


def get_parameters(fpath):
    
    param = {}
    with open(fpath + PARAMETERS, "r") as f:
        data = [line.rstrip("\n") for line in f]
    
    for line in data:
        key, val = line.split(' ')
        param[key] = val
        
    return param["dataset"], int(param["emb_size"]), int(param["embedding_number"])