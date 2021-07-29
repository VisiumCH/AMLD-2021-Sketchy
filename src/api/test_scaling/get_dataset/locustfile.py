from locust import HttpUser, task, between
import json


test_data = {"category": "pineapple", "dataset": "quickdraw"}


class APIUser(HttpUser):
    # Setting the host name and wait_time
    host = "http://localhost:5000"
    wait_time = between(3, 5)

    # Defining the post task using the JSON test data
    @task()
    def predict_endpoint(self):
        self.client.post("/get_dataset_images", json=test_data)