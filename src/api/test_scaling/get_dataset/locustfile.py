from locust import task
from src.api.test_scaling.default_locust import APIUser


test_data = {"category": "pineapple", "dataset": "quickdraw"}


class getDatasetTest(APIUser):
    @task()
    def predict_endpoint(self):
        self.client.post("/get_dataset_images", json=test_data)
