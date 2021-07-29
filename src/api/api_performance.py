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


class ModelPerformance:
    def __init__(self, model_path):
        event_path = glob(model_path + "events.out.tfevents*")[0]

        self.ea = event_accumulator.EventAccumulator(
            event_path,
            size_guidance={
                event_accumulator.IMAGES: 0,
                # event_accumulator.SCALARS: 0,
            },
        )
        self.ea.Reload()  # loads events from file

        # self.__preload_scalar()
        self.__prepare_images()

    # def __preload_scalar(self):
    #     print("preload_scalar")
    #     self.loss_domain_values = [value[2] for value in self.ea.Scalars("loss_domain")]
    #     self.loss_triplet_values = [
    #         value[2] for value in self.ea.Scalars("loss_triplet")
    #     ]
    #     self.loss_values = [value[2] for value in self.ea.Scalars("loss_train")]

    #     self.map = [value[2] for value in self.ea.Scalars("map_valid")]
    #     self.map_200 = [value[2] for value in self.ea.Scalars("map_valid_200")]
    #     self.prec_valid = [value[2] for value in self.ea.Scalars("prec_valid_200")]

    def __prepare_images(self):
        tag_list = self.ea.Tags()["images"]

        self.inference_dict = self.__prepare_image_type(tag_list, "Inference")
        self.im_attention_dict = self.__prepare_image_type(tag_list, "im")
        self.sk_attention_dict = self.__prepare_image_type(tag_list, "sk")

        # self.inference_tags = [tag for tag in image_tags if tag.startswith('Inference')]
        # self.image_attention_tags = [tag for tag in image_tags if tag.startswith('im')]
        # self.sketch_attention_tags = [tag for tag in image_tags if tag.startswith('sk')]

    def __prepare_image_type(self, tag_list, keyword):
        tags = [tag for tag in tag_list if tag.startswith(keyword)]
        tags = tags[::2]

        if keyword == "Inference":
            left, top, right, bottom = INFERENCE_CROP
        else:
            left, top, right, bottom = ATTENTION_CROP

        prepared_data = {}
        for tag in tags:
            image_list = self.ea.Images(tag)
            image_list = image_list[::3]
            images_processed = [
                Image.open(io.BytesIO(image[2])).crop((left, top, right, bottom))
                for image in image_list
            ]

            if keyword != "Inference":  # resize attention
                images_processed = [
                    image.resize((int(image.size[0] / 1.5), int(image.size[1] / 1.5)))
                    for image in images_processed
                ]

            prepared_data[tag] = {
                str(epoch[1]): base64_encoding(image)
                for epoch, image in zip(image_list, images_processed)
            }
        return prepared_data

    def get_scalars(self):
        return {
            "domain_loss": self.loss_domain_values,
            "triplet_loss": self.loss_triplet_values,
            "total_loss": self.loss_values,
            "map": self.map,
            "map_200": self.map_200,
            "prec_valid": self.prec_valid,
        }

    def get_image(self, image_type):
        if image_type == "Inference":
            images_processed = self.inference_dict
        elif image_type == "image_attention":
            images_processed = self.im_attention_dict
        elif image_type == "sketch_attention":
            images_processed = self.sk_attention_dict
        else:
            raise Exception(f"Image type {image_type} not implemented.")

        keys = list(images_processed.keys())
        data = images_processed[random.choice(keys)]
        data["length"] = len(data)
        return data
