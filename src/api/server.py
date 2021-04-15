import os

from flask import Flask, request, make_response
from flask_restful import Resource, Api
import json

from src.api.api_inference import ApiInference
from src.api.utils import (
    svg_to_png,
    prepare_images_data,
    prepare_embeddings_data,
    process_embeddings,
    prepare_dataset_data,
    map_curvenumber_image,
    prepare_sketch,
)
from src.data.constants import Split

app = Flask(__name__)
api = Api(app)


class Args:
    dataset = "quickdraw"
    emb_size = 256
    cuda = False
    save = "io/models/quickdraw/"
    load = save + "checkpoint.pth"
    embeddings_path = save + "00012/default/"


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


class Embeddings(Resource):
    """ Receives a sketch and returns its closest images. """

    def post(self):
        json_data = request.get_json()
        args = Args()

        if "nb_dim" not in json_data.keys():
            return {"ERROR": "Number of dimensions not provided"}, 400
        nb_dimensions = json_data["nb_dim"]

        # Verify the data
        if "sketch" not in json_data.keys():
            df = process_embeddings(
                args.embeddings_path, n_components=nb_dimensions, sketch_emb=False
            )
        else:
            sketch = svg_to_png(json_data["sketch"])
            sketch_embedding = inference.inference_sketch(sketch)

            df = process_embeddings(
                args.embeddings_path,
                n_components=nb_dimensions,
                sketch_emb=sketch_embedding,
            )

        data = prepare_embeddings_data(df, nb_dimensions)

        return make_response(json.dumps(data), 200)


class Dataset(Resource):
    """ Receives a category and returns associated images. """

    def post(self):
        json_data = request.get_json()

        if "category" not in json_data.keys():
            return {"ERROR": "Category not provided"}, 400
        category = json_data["category"]

        dataset_path = "io/data/raw/Quickdraw/"

        data_sketches = prepare_dataset_data(dataset_path, "sketches", category)
        data_images = prepare_dataset_data(dataset_path, "images", category)

        data = {**data_sketches, **data_images}

        return make_response(json.dumps(data), 200)


class ShowEmbeddingImage(Resource):
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
            if "pointnumber" not in json_data.keys():
                return {"ERROR": "Pointnumber not provided"}, 400

            key = (json_data["class"], json_data["pointnumber"])
            data = {"image": class_curvenumber_to_image[key]}

        return make_response(json.dumps(data), 200)


api.add_resource(APIList, "/api_list")
api.add_resource(Inferrence, "/find_images")
api.add_resource(Embeddings, "/get_embeddings")
api.add_resource(Dataset, "/get_dataset_images")
api.add_resource(ShowEmbeddingImage, "/get_embedding_images")

if __name__ == "__main__":

    args = Args()
    inference = ApiInference(args, Split.test)
    class_curvenumber_to_image = map_curvenumber_image(args)
    app.run(host="0.0.0.0", port="5000", debug=True)
