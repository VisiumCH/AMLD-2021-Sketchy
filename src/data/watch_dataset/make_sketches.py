import os
from glob import glob
import cv2 as cv

from src.data.watch_dataset.hed import HED


# GLOBAL VARIABLE DECLARATION
HED_MODEL_PATH = 'io/models/watch_pretrained_models/hed'


def make_sketches(args):

    hed = HED(HED_MODEL_PATH)

    watches_folder = os.path.join(args.data_path, 'Watch')

    watches_images_path = glob(watches_folder + '/*/*/*/*image.png')
    number_images = len(watches_images_path)

    for i, image_path in enumerate(watches_images_path):
        # Predict sketch from image
        img = cv.imread(image_path)
        pred = hed.predict(img)
        cv.imwrite(image_path.replace('image.png', 'sketch.png'), 255*pred)

        if i % 100 == 0:
            print(f'{i} on {number_images}')


if __name__ == '__main__':
    from src.options import Options

    # Parse options
    args = Options().parse()
    print("Parameters:\t" + str(args))

    make_sketches(args)
