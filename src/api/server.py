from flask import Flask, request, make_response
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import flask_restful
from flask_restful import Resource, Api
import json
import numpy as np
import pandas as pd
import torch

from src.api.api_inference import ApiInference
from src.api.api_options import ApiOptions
from src.api.utils import (
    svg_to_png,
    prepare_images_data,
    prepare_dataset,
    prepare_sketch,
    get_parameters,
    get_last_epoch_number
)
from src.api.embeddings_utils import (
    prepare_embeddings_data,
    process_graph,
    get_tiles,
)
from src.api.api_performance import ModelPerformance
from src.constants import CUSTOM_SKETCH_CLASS, DATA_PATH, MODELS_PATH, TENSORBOARD_IMAGE

app = Flask(__name__)
api = Api(app)


class APIList(Resource):
    """
    Return a list of available command and their description
    """

    def get(self):
        api_json = {
            "cmd: /api_list": "return the list of available commands.",
            "cmd: /find_images": "receives a sketch and returns its closest images.",
        }
        print("In api list function of server")
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
        nb_dimensions = json_data["nb_dim"]

        global df
        if "sketch" not in json_data.keys():
            df = process_graph(
                args.embeddings_path, n_components=nb_dimensions, sketch_emb=False
            )
        else:
            sketch = svg_to_png(json_data["sketch"])
            sketch_embedding = inference.inference_sketch(sketch)

            df = process_graph(
                args.embeddings_path,
                n_components=nb_dimensions,
                sketch_emb=sketch_embedding,
            )

        data = prepare_embeddings_data(df, nb_dimensions)

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
            if "x" not in json_data.keys() or "y" not in json_data.keys():
                return {"ERROR": "Pointnumber not provided"}, 400

            if "z" in json_data.keys():
                point = json_data["x"], json_data["y"], json_data["z"]
                dist = np.sum((df[["x", "y", "z"]].values - point) ** 2, axis=1)
            else:
                point = json_data["x"], json_data["y"]
                dist = np.sum((df[["x", "y"]].values - point) ** 2, axis=1)

            # find index of closest image to x y z in dataframe.
            data = {"image": tiles[np.argmin(dist)]}

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
        print(data)
        print(type(data["0"]))
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
    args.cuda=False

    args.save = MODELS_PATH + args.name + '/'
    args.dataset, args.emb_size, embedding_number = get_parameters(args.save)
    args.load = args.save + "checkpoint.pth"
    
    # Precompute the images from the large tensorboard sprite
    epoch_number = get_last_epoch_number(args.save)
    args.embeddings_path = args.save + epoch_number  + "/default/"
    tiles = get_tiles(args.embeddings_path + TENSORBOARD_IMAGE, embedding_number)

    # Global dataframe 
    # Gets data when opening embedding graphs ("/get_embeddings")
    # Is later called when image are clicked in another api ("/get_embedding_images")
    df = pd.DataFrame() 

    inference = ApiInference(args, "test")
    performance = ModelPerformance(args.save)
    app.run(host="0.0.0.0", port="5000", debug=True)
