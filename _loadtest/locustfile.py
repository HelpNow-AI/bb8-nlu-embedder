from locust import HttpUser, FastHttpUser, task

class TritonInfer(HttpUser):
    @task
    def inference(self):
        self.client.post(
            "/triton/bi-encoder/infer",
            json = {
                "queries": [
                    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
                    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
                    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
                    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
                    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla"                    
                ]
            }
        )

# run "$ locust" in terminal