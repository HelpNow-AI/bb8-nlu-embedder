import sys
import os
import time
import datetime
import traceback
import gc
import requests
# from concurrent import futures

import numpy as np
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_properties = torch.cuda.get_device_properties(0)


## Load Models ##
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device=device)
nlu_embedder.to("cuda")
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
        return JSONResponse({'embed_vector': [float(nlu_embedder.encode(query, device=device))]})
    except:
        logger.error(f'{traceback.format_exc()}')
        return JSONResponse({'embed_vector': None})


@app.post("/api/nlu/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    try:
        return JSONResponse({'embed_vector': nlu_embedder.encode([r['text'] for r in item.dict()['data']].astype(float).tolist(), device=device)})
    except:
        logger.error(f'{traceback.format_exc()}')
        return JSONResponse({"embed_vector" : [None for _ in item.dict()['data']]})


@app.get("/api/assist/sentence-embedding")
def sentence_embedding(query: str):
    try:
        return JSONResponse({'embed_vector': assist_bi_encoder.encode_queries(query)}) # query_instruction_for_retrieval + query
    except:
        logger.error(f'{traceback.format_exc()}')
        return JSONResponse({'embed_vector': None})


@app.post("/api/assist/sentence-embedding-batch")
def sentence_embedding_batch(item: EmbeddingItem):
    try:
        JSONResponse({'embed_vector' : assist_bi_encoder.encode([r['text'] for r in item.dict()['data']]).astype(float).tolist()})
    except:
        logger.error(f'{traceback.format_exc()}')
        return JSONResponse({"embed_vector": [None for _ in item.dict()['data']]})

@app.post("/api/assist/cross-encoder/similarity-scores")
def sentence_embedding_batch(item: EmbeddingItem):
    return JSONResponse({"similarity_scores": assist_cross_encoder.compute_score([[r['query'], r['passage']] for r in item.dict()['data']])})


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