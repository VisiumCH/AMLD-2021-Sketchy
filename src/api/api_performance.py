"""
Tensorboard saves the scalar and images (other then the embeddings) in a tfevent s file
"""
import io
from PIL import Image
from glob import glob
import random
from tensorboard.backend.event_processing import event_accumulator

from src.api.utils import base64_encoding
from src.constants import INFERENCE_CROP, ATTENTION_CROP


class ModelPerformance():
    def __init__(self, model_path):
        event_path = glob(model_path + "events.out.tfevents*")[0]

        self.ea = event_accumulator.EventAccumulator(
            event_path,
            size_guidance={
                event_accumulator.IMAGES: 0,
                event_accumulator.SCALARS: 0,
            }
        )
        self.ea.Reload()  # loads events from file

        self.__preload_scalar()
        self.__prepare_images()

    def __preload_scalar(self):
        self.loss_domain_values = [value[2] for value in self.ea.Scalars('loss_domain')]
        self.loss_triplet_values = [value[2] for value in self.ea.Scalars('loss_triplet')]
        self.loss_values = [value[2] for value in self.ea.Scalars('loss_train')]

        self.map = [value[2] for value in self.ea.Scalars('map_valid')]
        self.map_200 = [value[2] for value in self.ea.Scalars('map_valid_200')]
        self.prec_valid = [value[2] for value in self.ea.Scalars('prec_valid_200')]

    def __prepare_images(self):
        image_tags = self.ea.Tags()['images']

        self.inference_tags = [tag for tag in image_tags if tag.startswith('Inference')]
        self.image_attention_tags = [tag for tag in image_tags if tag.startswith('im')]
        self.sketch_attention_tags = [tag for tag in image_tags if tag.startswith('sk')]

    def get_scalars(self):
        return {
            "domain_loss": self.loss_domain_values,
            "triplet_loss": self.loss_triplet_values,
            "total_loss": self.loss_values,
            "map": self.map,
            "map_200": self.map_200,
            "prec_valid": self.prec_valid
        }

    def get_image(self, image_type):
        if image_type == 'inference':
            tags = self.inference_tags
            left, top, right, bottom = INFERENCE_CROP
        elif image_type == 'attention_image':
            tags = self.image_attention_tags
            left, top, right, bottom = ATTENTION_CROP
        elif image_type == 'attention_sketch':
            tags = self.sketch_attention_tags
            left, top, right, bottom = ATTENTION_CROP
        else:
            raise Exception(f"Image type {image_type} not implemented.")

        image_list = self.ea.Images(random.choice(tags))

        images_processed = [Image.open(io.BytesIO(image[2])).crop(
            (left, top, right, bottom)) for image in image_list]
        if image_type != 'inference':
            images_processed = [image.resize((int(image.size[0]/1.5), int(image.size[1]/1.5)))
                                for image in images_processed]

        data = {str(epoch[1]): base64_encoding(image) for epoch, image in zip(image_list, images_processed)}
        data['length'] = len(data)
        return data
