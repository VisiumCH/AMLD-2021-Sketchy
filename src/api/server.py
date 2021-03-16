from flask import Flask, request, jsonify
from flask_restful import Resource, Api

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
        print('in here')

        # Verify the data
        if "sketch" not in json_data.keys():
            return jsonify({"ERROR": "No sketch provided"}, 400)

        return jsonify({"OK": f"Sketch received {0} problem."}, 200)


api.add_resource(APIList, "/api_list")
api.add_resource(InferImages, "/find_images")

if __name__ == "__main__":

    args = Args()
    inference = Inference(args, Split.test)
    app.run(host="0.0.0.0", port="5000", debug=True)
