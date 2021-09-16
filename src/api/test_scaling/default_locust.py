from abc import abstractmethod
from locust import HttpUser, task, between


class APIUser(HttpUser):
    # Setting the host name: the default host is 5000,
    # but you can change it in the web app UI
    # before launching the stress testing.
    host = "http://localhost:5000"
    wait_time = between(3, 5)

    @abstractmethod
    @task()
    def predict_endpoint(self):
        pass
