import random
from locust import HttpUser
from locust import between
from locust import task

SAMPLE_SENTENCES = [
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
    "The city awakens as morning light filters through the streets.",
    "Stars twinkle in the velvety blanket of the night sky.",
    "The aroma of fresh coffee fills the cozy kitchen.",
    "A curious kitten pounces on a fluttering butterfly.",
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
    "The city awakens as morning light filters through the streets.",
    "Stars twinkle in the velvety blanket of the night sky.",
    "The aroma of fresh coffee fills the cozy kitchen.",
    "A curious kitten pounces on a fluttering butterfly."
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
]


class BentoHttpUser(HttpUser):
    """
        Start locust load testing client with:

            locust --class-picker -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """
    
    # @task
    # def assist_sentence_embedding(self):
    #     random_number = random.randint(1, len(SAMPLE_SENTENCES))
    #     self.client.post("/api/assist/sentence-embedding", json={"query": SAMPLE_SENTENCES[random_number]})

    @task
    def assist_sentence_embedding_batch(self):
        self.client.post("/api/assist/sentence-embedding-batch", json={"data": SAMPLE_SENTENCES})

    # wait_time = between(0.01, 2)
