from locust import HttpUser, task, between
import json


# Loading the test JSON data
with open("mock_svg/test1.json") as f:
    test_data = json.loads(f.read())
test_data['nb_dim'] = 3


class APIUser(HttpUser):
    # Setting the host name and wait_time
    host = "http://localhost:5000"
    wait_time = between(3, 5)

    # Defining the post task using the JSON test data
    @task()
    def predict_endpoint(self):
        self.client.post("/get_embeddings", json=test_data)
