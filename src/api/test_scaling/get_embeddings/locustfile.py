from locust import task
import json

from src.api.test_scaling.default_locust import APIUser


# Loading the test JSON data
with open("../mock_svg/test.json") as f:
    test_data = json.loads(f.read())
test_data["nb_dim"] = 3
test_data["reduction_algo"] = "TSNE"


class getEmbedding(APIUser):
    @task()
    def predict_endpoint(self):
        self.client.post("/get_embeddings", json=test_data)
