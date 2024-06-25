import sys
import os
import time
import datetime
import traceback
import gc
import requests
# from concurrent import futures

import numpy as np
import numba as nb
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import FlagModel, FlagReranker


# fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from _config import logger


os.environ["TOKENIZERS_PARALLELISM"] = "true"
output_model_dir = "./_output" # trained model


## FastAPI & CORS (Cross-Origin Resource Sharing) ##
app = FastAPI(
    title="bb8-nlu-embedder",
    version="0.2.5"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@nb.jit(nopython=True)
def numpy_to_list(vector: np.ndarray) -> list:
    n = vector.shape[0]  # Get the length of the 1D array
    result = [0.0] * n  # Initialize a list with float elements

    for i in range(n):
        result[i] = float(vector[i])  # Convert each element to float and assign to result list

    return result  # Return the result as a list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_properties = torch.cuda.get_device_properties(0)


## Load Models ##
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device=device)
assist_bi_encoder = FlagModel('BAAI/bge-base-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
assist_cross_encoder = FlagReranker('BAAI/bge-reranker-base', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation


@app.get('/health')
def health_check():
    '''
    Health Check
    '''
    return JSONResponse({'status':"helpnow-embedder is listening..", "timestamp":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})


class EmbeddingItem(BaseModel):
    data = []

@app.get("/api/nlu/sentence-embedding")
def sentence_embedding(query):
    try:
        embed_vector = nlu_embedder.encode(query, device=device)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vector = None

    #embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': numpy_to_list(embed_vector)})


@app.post("/api/nlu/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    item = item.dict()
    data = item['data']
    query_list = [r['text'] for r in data]

    try:
        embed_vectors = nlu_embedder.encode(query_list, device=device)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = [float(v) for v in embed_vectors[i]]

    return JSONResponse(data)


@app.get("/api/assist/sentence-embedding")
def sentence_embedding(query: str):
    try:
        embed_vector = assist_bi_encoder.encode_queries(query) # query_instruction_for_retrieval + query
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vector = None

    #embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': numpy_to_list(embed_vector)})


@app.post("/api/assist/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    item = item.dict()
    data = item['data']
    query_list = [r['text'] for r in data]

    try:
        embed_vectors = assist_bi_encoder.encode(query_list)
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = numpy_to_list(embed_vectors[i])

    return JSONResponse(data)


@app.post("/api/assist/cross-encoder/similarity-scores")
def sentence_embedding_batch(item: EmbeddingItem):
    s = time.time()
    item = item.dict()
    data = item['data']

    query_doc_list = [[r['query'], r['passage']] for r in data]
    similarity_scores = assist_cross_encoder.compute_score(query_doc_list)
    print(f'⏱️ process time of cross-encoder: {time.time() - s}')

    return JSONResponse({"similarity_scores": similarity_scores})


#===========================
# CUDA Memory Check 스케쥴러 입니다.
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

def check_cuda_memory():
    if device.type == "cuda":
        current_memory = round(torch.cuda.memory_allocated() / (1024 ** 3), 4)
        total_memory = round(gpu_properties.total_memory / (1024 ** 3), 4)
        print(f'>> Usage of Current Memory: {current_memory} GB / {total_memory} GB')

        gc.collect()
        torch.cuda.empty_cache()
    else:
        print('>> Not using CUDA.')

scheduler = BackgroundScheduler()

# 스케줄러에 작업 추가 (예: 10초마다 실행)
scheduler.add_job(check_cuda_memory, IntervalTrigger(seconds=30))
# 스케줄러 시작
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()