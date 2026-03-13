from locust import HttpUser, task, between

class HelloUser(HttpUser):
    # Temps d'attente entre 2 requêtes d'un même user virtuel
    wait_time = between(0.5, 1.5)

    @task
    def call_hello(self):
        # Appelle ton endpoint /hello
        self.client.get("/hello", params={"name": "Bob"})
    
    @task
    def call_diamond_price(self):
        # Appelle ton endpoint /predict_one
        self.client.post("/predict_one", params={"carat": 1, "cut": "Ideal", "color": "E", "clarity": "SI2", "depth": 1, "table": 1, "x": 1, "y": 1, "z": 1})
    
