import io

import base64
from cairosvg import svg2png
import pandas as pd
from PIL import Image


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


def prepare_data(images, image_labels, attention):
    data = {}
    data['images_base64'] = []
    data['images_label'] = []

    for image, image_label in zip(images, image_labels):
        rawBytes = io.BytesIO()
        image.save(rawBytes, "PNG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())

        data['images_base64'].append(str(img_base64))
        data['images_label'].append(' '.join(image_label.split('_')))

    rawBytes = io.BytesIO()
    attention.save(rawBytes, "PNG")
    rawBytes.seek(0)
    attention_base64 = base64.b64encode(rawBytes.read())
    data['attention'] = str(attention_base64)

    return data


def prepare_embeddings(df):
    data = {}
    data['x'] = list(df['embeddings_1'])
    data['y'] = list(df['embeddings_2'])
    data['z'] = list(df['embeddings_1'])
    data['classes'] = list(df['classes'])

    return data
