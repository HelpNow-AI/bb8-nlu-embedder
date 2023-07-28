import sys
import os
import time
import datetime
import traceback
import requests
# from concurrent import futures

import numpy as np
from sentence_transformers import SentenceTransformer


# fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from _config import logger


os.environ["TOKENIZERS_PARALLELISM"] = "true"
output_model_dir = "./_output" # trained model


## FastAPI & CORS (Cross-Origin Resource Sharing) ##
app = FastAPI(
    title="bb8-embedder-gpu",
    version="0.2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## SentenceBERT ##
# nlu_embedder = SentenceTransformer('./transformer-models/nlu-sentence-embedder', device='cpu')
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device='cpu')
# pool = nlu_embedder.start_multi_process_pool()

# assist_embedder = SentenceTransformer('./transformer-models/assist-sentence-embedder', device='cpu')
assist_bi_encoder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device='cpu')
# pool = assist_embedder.start_multi_process_pool()


@app.get('/health')
def health_check():
    '''
    Health Check
    '''
    return JSONResponse({'status':"bb8-embedder-gpu is listening..", "timestamp":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})


class EmbeddingItem(BaseModel):
    data = []

@app.get("/api/nlu/sentence-embedding")
def sentence_embedding(query):
    try:
        embed_vector = nlu_embedder.encode(query)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vector = None


    embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': embed_vector})


@app.post("/api/nlu/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    item = item.dict()
    data = item['data']
    query_list = [r['text'] for r in data]

    try:
        embed_vectors = nlu_embedder.encode(query_list)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = [float(v) for v in embed_vectors[i]]

    return JSONResponse(data)


@app.get("/api/assist/sentence-embedding")
def sentence_embedding(query):
    try:
        embed_vector = assist_bi_encoder.encode(query)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vector = None


    embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': embed_vector})


@app.post("/api/assist/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    item = item.dict()
    data = item['data']
    query_list = [r['text'] for r in data]

    try:
        embed_vectors = assist_bi_encoder.encode(query_list)
        # embed_vectors = assist_bi_encoder.encode_multi_process(query_list, pool)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = [float(v) for v in embed_vectors[i]]

    return JSONResponse(data)
