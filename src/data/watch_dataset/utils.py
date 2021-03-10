from src.data.utils import create_dict_texts


def get_watch_dict(images_path):
    labels = [get_watch_label(image) for image in images_path]
    return create_dict_texts(labels)


def get_watch_label(watch_image_path):
    ''' Returns metadata containing the brand, name and variant '''

    splitted_path = watch_image_path.split('/')

    brand = splitted_path[-4]
    model = splitted_path[-3]
    variant = splitted_path[-2]

    return f'{brand}_{model}_{variant}'


def watch_dataset_split(images_path):

    train_images = images_path[: int(0.8 * len(images_path))]
    valid_images = images_path[int(0.8 * len(images_path)):int(0.9 * len(images_path))]
    test_images = images_path[int(0.9 * len(images_path)):]

    def sketches_from_images(images_path):
        return [image.replace('image.png', 'sketch.png') for image in images_path]

    train_sketches = sketches_from_images(train_images)
    valid_sketches = sketches_from_images(valid_images)
    test_sketches = sketches_from_images(test_images)

    return [train_images, train_sketches], [valid_images, valid_sketches], [test_images, test_sketches]
