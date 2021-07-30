"""
File to test web-scaling with locust (http://localhost:8089/)
See this post: https://towardsdatascience.com/performance-testing-an-ml-serving-api-with-locust-ecd98ab9b7f7
It will make the virtual machine crash at the limit.
"""
from locust import HttpUser, task, between
import json


# Loading the test JSON data
with open("../mock_svg/test.json") as f:
    test_data = json.loads(f.read())


class APIUser(HttpUser):
    # Setting the host name and wait_time
    host = "http://localhost:5000"
    wait_time = between(3, 5)

    # Defining the post task using the JSON test data
    @task()
    def predict_endpoint(self):
        self.client.post("/find_images", json=test_data)
