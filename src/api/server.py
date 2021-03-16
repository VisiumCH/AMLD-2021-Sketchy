import io

import base64
from cairosvg import svg2png
from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
import json

from src.data.constants import Split
from src.models.inference.inference import Inference

app = Flask(__name__)
api = Api(app)


class Args:
    dataset = "sketchy"
    emb_size = 256
    cuda = False
    best_model = 'io/models/best_model/checkpoint.pth'


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

        sketch_fname = 'sketch.png'
        svg2png(bytestring=json_data["sketch"], write_to=sketch_fname)
        inference.inference_sketch(sketch_fname)

        image, image_label = inference.return_closest_image()

        rawBytes = io.BytesIO()
        image.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())

        data = {'image_string': str(img_base64),
                'image_label': image_label}

        return make_response(json.dumps(data), 200)


api.add_resource(APIList, "/api_list")
api.add_resource(InferImages, "/find_images")

if __name__ == "__main__":

    args = Args()
    inference = Inference(args, Split.test)
    app.run(host="0.0.0.0", port="5000", debug=True)
