import sys
import os
import time
import datetime
import traceback
import requests
# from concurrent import futures

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from FlagEmbedding import FlagModel, FlagReranker


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
    title="helpnow-embedder",
    version="0.2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## SentenceBERT ##
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device=device)
# assist_bi_encoder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device=device)
# assist_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)



from FlagEmbedding import FlagModel
assist_bi_encoder = FlagModel('BAAI/bge-base-en-v1.5', 
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=True
            ) # Setting use_fp16 to True speeds up computation with a slight performance degradation

assist_cross_encoder = FlagReranker('BAAI/bge-reranker-base', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation


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

    embed_vector = [float(v) for v in embed_vector]

    return JSONResponse({'embed_vector': embed_vector})


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
        embed_vector = assist_bi_encoder.encode_queries(query)
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
    except:
        logger.error(f'{traceback.format_exc()}')
        embed_vectors = None

    for i, row in enumerate(data):
        row['embed_vector'] = [float(v) for v in embed_vectors[i]]

    return JSONResponse(data)


@app.post("/api/assist/cross-encoder/similarity-scores")
def sentence_embedding_batch(item: EmbeddingItem):
    s = time.time()
    item = item.dict()
    data = item['data']

    query_doc_list = [r['query'], r['passage'] for r in data]
    similarity_scores = assist_cross_encoder.compute_score(query_doc_list)

    print(f'⏱️ process time of cross-encoder: {time.time() - s}')

    return JSONResponse({"similarity_scores": similarity_scores})