from flask import Flask, request, make_response
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import flask_restful
from flask_restful import Resource, Api
import json
import numpy as np
import pandas as pd

from src.api.api_inference import ApiInference
from src.api.api_options import ApiOptions
from src.api.utils import (
    svg_to_png,
    prepare_images_data,
    prepare_dataset,
    prepare_sketch,
    get_parameters
)
from src.api.embeddings_utils import (
    prepare_embeddings_data,
    process_graph,
    get_tiles,
)

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
        category = json_data["category"]

        dataset_path = "io/data/raw/Quickdraw/"

        data_sketches = prepare_dataset(dataset_path, "sketches", category)
        data_images = prepare_dataset(dataset_path, "images", category)

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

        if json_data["class"] == "My Custom Sketch":
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


api.add_resource(APIList, "/api_list")
api.add_resource(Inferrence, "/find_images")
api.add_resource(Embeddings, "/get_embeddings")
api.add_resource(Dataset, "/get_dataset_images")
api.add_resource(ShowEmbeddingImage, "/get_embedding_images")

    
if __name__ == "__main__":

    args = ApiOptions().parse()
    args.save = args.log + args.name + '/'
    args.dataset, args.emb_size = get_parameters(args.save)
    args.load = args.save + "checkpoint.pth"
    args.embeddings_path = args.save + args.epoch  + "/default/"
    args.cuda = False

    tiles = get_tiles(args.embeddings_path + "sprite.png")
    df = pd.DataFrame()

    inference = ApiInference(args, "test")
    app.run(host="0.0.0.0", port="5000", debug=True)
