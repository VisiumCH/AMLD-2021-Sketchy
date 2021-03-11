import os
from glob import glob
import cv2 as cv
import numpy as np

from src.data.watch_dataset.hed import HED


# GLOBAL VARIABLE DECLARATION
HED_MODEL_PATH = 'io/models/watch_pretrained_models/hed'


def make_sketches():

    hed = HED(HED_MODEL_PATH)

    watches_folder = 'io/data/raw/Watch'

    watches_images_path = glob(watches_folder + '/*/*/*/*image.png')
    number_images = len(watches_images_path)

    for i, image_path in enumerate(watches_images_path):
        if i > 11388:
            if i % 10 == 0:
                print(f'{i} on {number_images}: {image_path}')

            # Predict sketch from image
            img = cv.imread(image_path)

            if isinstance(img, np.ndarray):
                pred = hed.predict(img)
                cv.imwrite(image_path.replace('image.png', 'sketch.png'), 255*pred)
            else:
                print(f'\nImg {image_path} is not a numpy array.')
                os.remove(image_path)


if __name__ == '__main__':

    make_sketches()
