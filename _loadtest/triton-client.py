from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from pytriton.client import ModelClient

app = FastAPI()

# Triton client 
nlu_client = ModelClient("http://34.64.221.13:8000", "bb8-embedder-nlu", init_timeout_s=600.0)
assist_client = ModelClient("http://34.64.221.13:8000", "bb8-embedder-assist-biencoder-passage", init_timeout_s=600.0)


class Query(BaseModel):
    queries: List[str]

@app.post("/triton/nlu/bi-encoder/infer")
async def infer(query: Query):
    try:
        queries = [[q] for q in query.queries]
        sequence = np.array(queries)
        sequence = np.char.encode(sequence, "utf-8")
        
        result_dict = nlu_client.infer_batch(sequence)
        embed_vector = np.array([row.astype(float) for row in result_dict['embed_vectors']])

        return {"embed_vectors": embed_vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/triton/assist/bi-encoder/infer")
async def infer(query: Query):
    try:
        queries = [[q] for q in query.queries]
        sequence = np.array(queries)
        sequence = np.char.encode(sequence, "utf-8")
        
        result_dict = assist_client.infer_batch(sequence)
        embed_vector = np.array([row.astype(float) for row in result_dict['embed_vectors']])

        return {"embed_vectors": embed_vector.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



"""example

$ curl --location 'http://34.171.159.167:8000/triton/nlu/bi-encoder/infer' \
--header 'Content-Type: application/json' \
--data '{
    "queries": [
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla"                    
    ]
}'

$ curl --location 'http://34.171.159.167:8000/triton/assist/bi-encoder/infer' \
--header 'Content-Type: application/json' \
--data '{
    "queries": [
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla",
        "hello, my name is jaehyeong.", "I really want to go home.", "Sorry.. But, I hate you.", "hahahahahah, blablablabla"                    
    ]
}'
"""