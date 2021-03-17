import io

import base64
from cairosvg import svg2png
from PIL import Image


class Args:
    dataset = "sketchy"
    emb_size = 256
    cuda = False
    best_model = 'io/models/sk_training/checkpoint.pth'


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


def prepare_data(images, image_labels):
    data = {}
    data['images_base64'] = []
    data['images_label'] = []

    for image, image_label in zip(images, image_labels):
        rawBytes = io.BytesIO()
        image.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())

        data['images_base64'].append(str(img_base64))
        data['images_label'].append(' '.join(image_label.split('_')))
    return data
