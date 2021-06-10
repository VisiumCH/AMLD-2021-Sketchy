from src.constants import CUSTOM_SKETCH_CLASS, DATA_PATH, MODELS_PATH
from src.api.utils import (
    svg_to_png,
    prepare_images_data,
    prepare_dataset,
    prepare_sketch,
    get_parameters,
)
from src.api.api_dimensionality_reduction import DimensionalityReduction
from src.api.api_options import ApiOptions
from src.api.api_inference import ApiInference
from src.api.api_performance import ModelPerformance
import torch
import json
from flask_restful import Resource, Api
from flask import Flask, request, make_response
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func


app = Flask(__name__)
api = Api(app)


class APIList(Resource):
    """
    Return a list of available command and their description
    """

    def get(self):
        api_json = {
            "cmd: /api_list": "Input: None. \
                Return the list of available commands.",
            "cmd: /find_images": "Input: an svg base 64 string of a sketch. \
                Returns base 64 string of its closest images, the associated labels and the attention map.",
            "cmd: /get_embeddings": "Inputs: a dimension number (2 or 3) and optinally a sketch. \
                Returns the projected points of the embeddings.",
            "cmd: /get_dataset_images": "Input: a category name. \
                Returns 5 random images and sketches of this category.",
            "cmd: /get_embedding_images": "Input: a class and clicked position. \
                Returns the closest image to the clicked position.",
            "cmd: /scalar_perf": "Input: None. \
                Returns the values of the loss and metrics during training.",
            "cmd: /image_perf": "Input: an image type (inference, sketch or image attention). \
                Returns how the model performed at each epoch of the training.",
        }
        return {
            "available-apis": [
                {"api-key": k, "description": v} for k, v in api_json.items()
            ]
        }


class Inferrence(Resource):
    """ Receives a sketch and returns its closest images. """

    def post(self):
        json_data = request.get_json()

        # Verify the data
        if "sketch" not in json_data.keys():
            return {"ERROR": "No sketch provided"}, 400

        sketch = svg_to_png(json_data["sketch"])

        inference.inference_sketch(sketch)
        images, image_labels = inference.get_closest(2)
        attention = inference.get_attention(sketch)

        data = prepare_images_data(images, image_labels, attention)

        return make_response(json.dumps(data), 200)


class Dataset(Resource):
    """ Receives a category and returns associated images. """

    def post(self):
        json_data = request.get_json()

        if "category" not in json_data.keys():
            return {"ERROR": "Category not provided"}, 400
        if "dataset" not in json_data.keys():
            return {"ERROR": "Dataset not provided"}, 400
        category = json_data["category"]
        dataset_folder = json_data["dataset"]

        dataset_path = DATA_PATH + dataset_folder

        data_sketches = prepare_dataset(dataset_path, "sketches", category, dataset_folder)
        data_images = prepare_dataset(dataset_path, "images", category, dataset_folder)

        data = {**data_sketches, **data_images}

        return make_response(json.dumps(data), 200)


class Embeddings(Resource):
    """ Receives a sketch and returns its closest images. """

    def post(self):
        json_data = request.get_json()

        # Verify the data
        if "nb_dim" not in json_data.keys():
            return {"ERROR": "Number of dimensions not provided"}, 400
        if "reduction_algo" not in json_data.keys():
            return {"ERROR": "Reduction algorithm not provided"}, 400

        if "sketch" not in json_data.keys():
            data = dim_red.get_projection(json_data["reduction_algo"], json_data["nb_dim"])
        else:
            sketch = svg_to_png(json_data["sketch"])
            sketch_embedding = inference.inference_sketch(sketch)
            data = dim_red.get_projection(json_data["reduction_algo"], json_data["nb_dim"], sketch_embedding)

        return make_response(json.dumps(data), 200)


class ShowEmbeddingImage(Resource):
    """Return the custom sketch or the image selected on the embedding graph"""

    def post(self):
        json_data = request.get_json()

        if "class" not in json_data.keys():
            return {"ERROR": "Class not provided"}, 400

        if json_data["class"] == CUSTOM_SKETCH_CLASS:
            if "sketch" not in json_data.keys():
                return {"ERROR": "sketch not provided"}, 400
            sketch = prepare_sketch(json_data["sketch"])
            data = {"image": sketch}
        else:
            if "nb_dim" not in json_data.keys():
                return {"ERROR": "Number of dimensions not provided"}, 400
            if "reduction_algo" not in json_data.keys():
                return {"ERROR": "Reduction algorithm not provided"}, 400
            if "x" not in json_data.keys() or "y" not in json_data.keys():
                return {"ERROR": "Pointnumber not provided"}, 400

            data = dim_red.get_closest_image(json_data)

        return make_response(json.dumps(data), 200)


class ScalarPerformance(Resource):
    """Return the scalar values (loss and metrics) of the model """

    def post(self):
        data = performance.get_scalars()
        return make_response(json.dumps(data), 200)


class ImagePerformance(Resource):
    """Return the custom sketch or the image selected on the embedding graph"""

    def post(self):
        json_data = request.get_json()

        if "image_type" not in json_data.keys():
            return {"ERROR": "Image Type not provided"}, 400

        data = performance.get_image(json_data["image_type"])
        return make_response(json.dumps(data), 200)


api.add_resource(APIList, "/api_list")
api.add_resource(Inferrence, "/find_images")
api.add_resource(Embeddings, "/get_embeddings")
api.add_resource(Dataset, "/get_dataset_images")
api.add_resource(ShowEmbeddingImage, "/get_embedding_images")
api.add_resource(ScalarPerformance, "/scalar_perf")
api.add_resource(ImagePerformance, "/image_perf")


if __name__ == "__main__":

    args = ApiOptions().parse()
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    print("Cuda:\t" + str(args.cuda))
    args.cuda = False

    args.save = MODELS_PATH + args.name + '/'
    args.dataset, args.emb_size, nb_embeddings = get_parameters(args.save)
    args.load = args.save + "checkpoint.pth"

    inference = ApiInference(args, "test")
    performance = ModelPerformance(args.save)
    dim_red = DimensionalityReduction(args.save, nb_embeddings)

    app.run(host="0.0.0.0", port="5000", debug=True)
