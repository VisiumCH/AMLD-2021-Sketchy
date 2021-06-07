from locust import HttpUser, task, between

test_data = {
    "category": "pineapple"
}


class APIUser(HttpUser):
    # Setting the host name and wait_time
    host = "http://localhost:5000"
    wait_time = between(3, 5)

    # Defining the post task using the JSON test data
    @task()
    def predict_endpoint(self):
        self.client.post("/get_embeddings", json=test_data)
