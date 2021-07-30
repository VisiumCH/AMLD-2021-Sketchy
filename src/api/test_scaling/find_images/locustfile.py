from locust import task
import json

from src.api.test_scaling.default_locust import APIUser


# Loading the test JSON data
with open("../mock_svg/test.json") as f:
    test_data = json.loads(f.read())


class findImagesTest(APIUser):
    @task()
    def predict_endpoint(self):
        self.client.post("/find_images", json=test_data)
