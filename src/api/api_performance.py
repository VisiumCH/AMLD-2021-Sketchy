"""
Tensorboard saves the scalar and images (other then the embeddings) in a tfevent s file
"""
import base64
from glob import glob
import random
from tensorboard.backend.event_processing import event_accumulator


class ModelPerformance():
    def __init__(self, model_path):
        event_path = glob(model_path + "events.out.tfevents*" )[0]

        self.ea = event_accumulator.EventAccumulator(
            event_path,
            size_guidance={
                event_accumulator.IMAGES: 0,
                event_accumulator.SCALARS: 0,
            }
        )
        self.ea.Reload() # loads events from file

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
        elif image_type == 'attention_image':
            tags = self.image_attention_tags
        elif image_type == 'attention_sketch':
            tags = self.sketch_attention_tags
        else:
            raise Exception(f"Image type {image_type} not implemented.")
        
        image_list = self.ea.Images(random.choice(tags))

        data = {}
        for image in image_list:
            with open("tmp.png", "wb") as f:
                f.write(image[2])
            encoded = base64.b64encode(open("tmp.png", "rb").read())
            data[]

        data = {str(image[1]): base64.b64encode(image[2]) for image in image_list}
        data['length'] = len(data)
        return data



