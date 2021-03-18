import os

from flask import Flask, request, make_response
from flask_restful import Resource, Api
import json
import random

from src.api.utils import Args, svg_to_png, prepare_data
from src.data.constants import Split
from src.models.inference.inference import Inference

app = Flask(__name__)
api = Api(app)


class APIList(Resource):
    """
    Return a list of available command and their description
    """

    def get(self):
        api_json = {"cmd: /api_list": "return the list of available commands.",
                    "cmd: /find_images": "receives a sketch and returns its closest images."
                    }
        print('In api list function of server')
        return {"available-apis": [{"api-key": k, "description": v} for k, v in api_json.items()]}


class InferImages(Resource):
    """ Receives a sketch and returns its closest images. """

    def post(self):
        json_data = request.get_json()

        # Verify the data
        if "sketch" not in json_data.keys():
            return {"ERROR": "No sketch provided"}, 400

        sketch_fname = 'sketch' + str(random.random()) + '.png'
        svg_to_png(json_data["sketch"], sketch_fname)

        inference.inference_sketch(sketch_fname)
        images, image_labels = inference.return_closest_images(2)
        data = prepare_data(images, image_labels)
        os.remove(sketch_fname)

        return make_response(json.dumps(data), 200)


api.add_resource(APIList, "/api_list")
api.add_resource(InferImages, "/find_images")

if __name__ == "__main__":

    args = Args()
    inference = Inference(args, Split.test)
    app.run(host="0.0.0.0", port="5000", debug=True)
