from __future__ import annotations

import typing as t
import traceback

import numpy as np
import bentoml


SAMPLE_SENTENCES = [
    "The sun dips below the horizon, painting the sky orange.",
    "A gentle breeze whispers through the autumn leaves.",
    "The moon casts a silver glow on the tranquil lake.",
    "A solitary lighthouse stands guard on the rocky shore.",
]

MODEL_ID = "BAAI/bge-base-en-v1.5"

@bentoml.service(
    traffic={"timeout": 60},
    resources={"gpu": 1},
)
class SentenceTransformers:

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer, models
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_ID, device=self.device)
        print(f"Model '{MODEL_ID}' loaded on device: '{self.device}'.")

    @bentoml.api(batchable=True)
    def encode(
        self,
        sentences: t.List[str]
    ) -> np.ndarray:
        
        return self.model.encode(sentences, normalize_embeddings=True, device=self.device)

""" Request example

curl -X 'POST' \
  'http://localhost:3000/encode' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla", 
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
    "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla"
  ]
}'
"""
    