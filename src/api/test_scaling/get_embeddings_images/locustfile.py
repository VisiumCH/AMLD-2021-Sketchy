from locust import task

from src.api.test_scaling.default_locust import APIUser


test_data = {
    "class": "butterfly",
    "nb_dim": 3,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "reduction_algo": "TSNE",
}


class getEmbeddingImages(APIUser):
    @task()
    def predict_endpoint(self):
        self.client.post("/get_embedding_images", json=test_data)
